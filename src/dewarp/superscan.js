// Superscan: fuse several captures of the same document into one clean
// scan. Shadows and wrinkle shading MOVE between frames while the paper
// content doesn't, so:
//   text (high frequency)  <- single sharpest frame, untouched
//   illumination (low freq) <- per-pixel 70th percentile across frames
//                              (brighter-biased: rejects moving shadows
//                               AND glare outliers)
//   result = ref * fusedLow / blur(ref)
// Validated offline on the 1818xx burst corpus: wrinkles flatten, text
// sharpness identical to the best single frame.

import { computeHomography, warpBilinear } from "./homography.js";

const MAX_EDGE = 1400;   // canonical size cap (memory + fuse time)

function quadSize(quad) {
  const d = (i, j) => Math.hypot(quad[i * 2] - quad[j * 2], quad[i * 2 + 1] - quad[j * 2 + 1]);
  return { w: Math.max(d(0, 1), d(3, 2)), h: Math.max(d(0, 3), d(1, 2)) };
}

function toGray(rgb, n) {
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3;
  return g;
}

function lapVar(g, W, H) {
  let sum = 0, sum2 = 0, n = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const v = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - W] - g[i + W];
      sum += v; sum2 += v * v; n++;
    }
  }
  const m = sum / n;
  return sum2 / n - m * m;
}

// Integer (dx,dy) minimizing SAD vs ref on 1/4-res gray, window ±6 (=±24 full-res).
function bestShift(refG, g, W, H) {
  const s = 4, sw = (W / s) | 0, sh = (H / s) | 0;
  const a = new Float32Array(sw * sh), b = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      a[y * sw + x] = refG[(y * s) * W + x * s];
      b[y * sw + x] = g[(y * s) * W + x * s];
    }
  let best = Infinity, bx = 0, by = 0;
  for (let dy = -6; dy <= 6; dy++) {
    for (let dx = -6; dx <= 6; dx++) {
      let sad = 0;
      for (let y = 8; y < sh - 8; y += 2) {
        const yy = y + dy;
        for (let x = 8; x < sw - 8; x += 2) {
          sad += Math.abs(a[y * sw + x] - b[yy * sw + x + dx]);
        }
      }
      if (sad < best) { best = sad; bx = dx; by = dy; }
    }
  }
  return { dx: bx * s, dy: by * s };
}

// Separable box blur, `iters` passes (3 ≈ gaussian). radius r, float32.
function boxBlur3(src, W, H, r, iters = 3) {
  let cur = Float32Array.from(src);
  const tmp = new Float32Array(W * H);
  for (let it = 0; it < iters; it++) {
    for (let y = 0; y < H; y++) {           // horizontal
      const row = y * W;
      let acc = 0;
      for (let x = -r; x <= r; x++) acc += cur[row + Math.min(W - 1, Math.max(0, x))];
      for (let x = 0; x < W; x++) {
        tmp[row + x] = acc / (2 * r + 1);
        acc += cur[row + Math.min(W - 1, x + r + 1)] - cur[row + Math.max(0, x - r)];
      }
    }
    for (let x = 0; x < W; x++) {           // vertical
      let acc = 0;
      for (let y = -r; y <= r; y++) acc += tmp[Math.min(H - 1, Math.max(0, y)) * W + x];
      for (let y = 0; y < H; y++) {
        cur[y * W + x] = acc / (2 * r + 1);
        acc += tmp[Math.min(H - 1, y + r + 1) * W + x] - tmp[Math.max(0, y - r) * W + x];
      }
    }
  }
  return cur;
}

// ── Cosmetic geometry: deskew + baseline curl flatten ───────────────────
// Rotation first (physical order): find the angle that maximizes the
// sharpness (variance) of horizontal ink-row sums — text rows become
// razor peaks when horizontal; price columns can't fool it. Then flatten
// residual curl by fitting each text line's baseline (glyph bottoms,
// descenders rejected by an asymmetric refit) and remapping a smooth
// dy(x,y) field interpolated between baselines. Both stages are gated
// and no-op on already-straight scans.
function straighten(rgb, W, H) {
  const n = W * H;
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) gray[i] = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3;
  // ink mask: darker than local mean
  const mu = boxBlur3(gray, W, H, 12, 1);
  const ink = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (gray[i] < mu[i] - 12) ink[i] = 1;

  // 1. deskew: coarse-to-fine angle search on subsampled ink
  const sharp = (deg) => {
    const rad = deg * Math.PI / 180, cos = Math.cos(rad), sin = Math.sin(rad);
    const rows = new Float64Array(H + 1);
    for (let y = 0; y < H; y += 2) {
      for (let x = 0; x < W; x += 2) {
        if (!ink[y * W + x]) continue;
        const ry = Math.round((x - W / 2) * sin + (y - H / 2) * cos + H / 2);
        if (ry >= 0 && ry <= H) rows[ry]++;
      }
    }
    let s = 0, s2 = 0;
    for (let y = 0; y <= H; y++) { s += rows[y]; s2 += rows[y] * rows[y]; }
    const m = s / (H + 1);
    return s2 / (H + 1) - m * m;
  };
  let best = 0, bestV = -1;
  for (let a = -8; a <= 8; a += 0.5) { const v = sharp(a); if (v > bestV) { bestV = v; best = a; } }
  for (let a = best - 0.5; a <= best + 0.5; a += 0.1) { const v = sharp(a); if (v > bestV) { bestV = v; best = a; } }

  let cur = rgb, curGray = gray, curInk = ink;
  if (Math.abs(best) >= 1.0) {
    const rad = -best * Math.PI / 180, cos = Math.cos(rad), sin = Math.sin(rad);
    const rot = new Uint8ClampedArray(n * 3);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const sx = Math.min(W - 1, Math.max(0, Math.round((x - W / 2) * cos - (y - H / 2) * sin + W / 2)));
        const sy = Math.min(H - 1, Math.max(0, Math.round((x - W / 2) * sin + (y - H / 2) * cos + H / 2)));
        const si = (sy * W + sx) * 3, di = (y * W + x) * 3;
        rot[di] = cur[si]; rot[di + 1] = cur[si + 1]; rot[di + 2] = cur[si + 2];
      }
    }
    cur = rot;
    curGray = new Float32Array(n);
    for (let i = 0; i < n; i++) curGray[i] = (cur[i * 3] + cur[i * 3 + 1] + cur[i * 3 + 2]) / 3;
    const mu2 = boxBlur3(curGray, W, H, 12, 1);
    curInk = new Uint8Array(n);
    for (let i = 0; i < n; i++) if (curGray[i] < mu2[i] - 12) curInk[i] = 1;
  }

  // 2. baseline curl: glyph components -> line clusters -> quadratic fits
  const comps = components(curInk, W, H, 4000);
  const glyphs = []; // [cx, bottomY, h]
  for (const c of comps) {
    if (c.area < 6) break;
    let x0 = W, x1 = 0, y0 = H, y1 = 0;
    for (let i = 0; i < c.pixels.length; i += 2) {
      const x = c.pixels[i], y = c.pixels[i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const h = y1 - y0 + 1, w = x1 - x0 + 1;
    if (h < 3 || h > H * 0.1 || w > W * 0.5) continue;
    glyphs.push([(x0 + x1) / 2, y1, h]);
  }
  if (glyphs.length < 40) return cur;
  const medH = glyphs.map(g => g[2]).sort((a, b) => a - b)[glyphs.length >> 1];
  glyphs.sort((a, b) => a[1] - b[1]);
  const lines = [];
  let curLine = [glyphs[0]], curY = glyphs[0][1];
  for (let i = 1; i < glyphs.length; i++) {
    if (glyphs[i][1] - curY <= 0.7 * medH) { curLine.push(glyphs[i]); curY = 0.7 * curY + 0.3 * glyphs[i][1]; }
    else { lines.push(curLine); curLine = [glyphs[i]]; curY = glyphs[i][1]; }
  }
  lines.push(curLine);

  const rowsFit = []; // {ymid, dy: Float32Array over grid, x0, x1}
  const GRID = 48;
  for (const ln of lines) {
    if (ln.length < 6) continue;
    ln.sort((a, b) => a[0] - b[0]);
    const x0 = ln[0][0], x1 = ln[ln.length - 1][0];
    if (x1 - x0 < W * 0.25) continue;
    // asymmetric quadratic fit: drop points >2px BELOW the fit (descenders)
    let pts = ln;
    let coef = null;
    for (let pass = 0; pass < 3; pass++) {
      coef = quadFit(pts);
      if (!coef) break;
      const kept = pts.filter(([x, y]) => {
        const f = coef[0] + coef[1] * x + coef[2] * x * x;
        return y - f < 2 && y - f > -3 * medH;
      });
      if (kept.length < 5 || kept.length === pts.length) { pts = kept.length >= 5 ? kept : pts; break; }
      pts = kept;
    }
    if (!coef || pts.length < 5) continue;
    let resid = 0;
    for (const [x, y] of pts) resid += Math.abs(coef[0] + coef[1] * x + coef[2] * x * x - y);
    if (resid / pts.length > 3) continue;
    const ymid = coef[0] + coef[1] * (W / 2) + coef[2] * (W / 2) ** 2;
    const dy = new Float32Array(GRID);
    for (let g = 0; g < GRID; g++) {
      const x = (g / (GRID - 1)) * (W - 1);
      const xc = Math.min(x1 + 20, Math.max(x0 - 20, x)); // clamp outside span
      dy[g] = coef[0] + coef[1] * xc + coef[2] * xc * xc - ymid;
    }
    rowsFit.push({ ymid, dy });
  }
  if (rowsFit.length < 5) return cur;
  rowsFit.sort((a, b) => a.ymid - b.ymid);
  let amp = 0;
  for (const r of rowsFit) for (let g = 0; g < GRID; g++) amp = Math.max(amp, Math.abs(r.dy[g]));
  if (amp < 5 || amp > H * 0.12) return cur;

  // remap with dy(x,y) interpolated between baselines
  const out = new Uint8ClampedArray(n * 3);
  for (let y = 0; y < H; y++) {
    // find bracketing baselines
    let lo = 0;
    while (lo < rowsFit.length - 1 && rowsFit[lo + 1].ymid < y) lo++;
    const hi = Math.min(rowsFit.length - 1, lo + 1);
    const yA = rowsFit[lo].ymid, yB = rowsFit[hi].ymid;
    const t = yB > yA ? Math.min(1, Math.max(0, (y - yA) / (yB - yA))) : 0;
    for (let x = 0; x < W; x++) {
      const gf = (x / (W - 1)) * (GRID - 1);
      const g0 = gf | 0, g1 = Math.min(GRID - 1, g0 + 1), ft = gf - g0;
      const dA = rowsFit[lo].dy[g0] * (1 - ft) + rowsFit[lo].dy[g1] * ft;
      const dB = rowsFit[hi].dy[g0] * (1 - ft) + rowsFit[hi].dy[g1] * ft;
      const sy = Math.min(H - 1, Math.max(0, Math.round(y + dA * (1 - t) + dB * t)));
      const si = (sy * W + x) * 3, di = (y * W + x) * 3;
      out[di] = cur[si]; out[di + 1] = cur[si + 1]; out[di + 2] = cur[si + 2];
    }
  }
  return out;
}

// 8-connected components of a 0/1 mask, largest-first, capped.
function components(mask, W, H, maxN) {
  const label = new Int32Array(W * H);
  const stack = new Int32Array(W * H);
  const out = [];
  let next = 1;
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || label[start]) continue;
    let top = 0, area = 0;
    const pixels = [];
    stack[top++] = start; label[start] = next;
    while (top > 0) {
      const i = stack[--top];
      const x = i % W, y = (i / W) | 0;
      pixels.push(x, y); area++;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= W) continue;
          const j = ny * W + nx;
          if (mask[j] && !label[j]) { label[j] = next; stack[top++] = j; }
        }
      }
    }
    next++;
    out.push({ pixels, area });
    if (out.length >= maxN) break;
  }
  out.sort((a, b) => b.area - a.area);
  return out;
}

// least-squares quadratic y = a + bx + cx^2 via normal equations
function quadFit(pts) {
  let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, t0 = 0, t1 = 0, t2 = 0;
  for (const [x, y] of pts) {
    s0++; s1 += x; s2 += x * x; s3 += x * x * x; s4 += x * x * x * x;
    t0 += y; t1 += x * y; t2 += x * x * y;
  }
  const M = [[s0, s1, s2, t0], [s1, s2, s3, t1], [s2, s3, s4, t2]];
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r2 = c + 1; r2 < 3; r2++) if (Math.abs(M[r2][c]) > Math.abs(M[p][c])) p = r2;
    if (Math.abs(M[p][c]) < 1e-9) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r2 = 0; r2 < 3; r2++) {
      if (r2 === c) continue;
      const f = M[r2][c] / M[c][c];
      for (let k = c; k < 4; k++) M[r2][k] -= f * M[c][k];
    }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

// Hard Sauvola binarization — the fax/scanner-app "B&W document" look.
// Maximum cosmetic cleanliness, ~10x smaller files, but costs OCR
// (0.39 -> 0.30 on the 20260722 burst: antialiasing and thin Thai
// diacritics shatter). Cosmetic/archival only; never feed to OCR.
function binarize(rgb, W, H) {
  const n = W * H;
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) gray[i] = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3;
  // SINGLE box pass: Sauvola stats need a sharp window; the 3-pass
  // gaussian-ish blur spreads mu so far that whole text rows binarize black.
  const win = 12;
  const mu = boxBlur3(gray, W, H, win, 1);
  const g2 = new Float32Array(n);
  for (let i = 0; i < n; i++) g2[i] = gray[i] * gray[i];
  const mu2 = boxBlur3(g2, W, H, win, 1);
  const out = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const sd = Math.sqrt(Math.max(0, mu2[i] - mu[i] * mu[i]));
    const v = gray[i] < mu[i] * (1 + 0.18 * (sd / 128 - 1)) ? 0 : 255;
    out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = v;
  }
  return out;
}

/**
 * frames: [{rgb: Uint8ClampedArray (h*w*3), w, h, quad}] — full-res frames
 * with their detected (already padded) quads. Returns {warped, w, h}.
 * opts.bw: true -> hard-binarized output instead of whitened.
 */
export function fuseScans(frames, opts = {}) {
  if (!frames.length) return null;
  // canonical rect from the largest quad, capped
  let W = 0, H = 0;
  for (const f of frames) {
    const s = quadSize(f.quad);
    if (s.w * s.h > W * H) { W = Math.round(s.w); H = Math.round(s.h); }
  }
  const cap = MAX_EDGE / Math.max(W, H);
  if (cap < 1) { W = Math.round(W * cap); H = Math.round(H * cap); }
  if (W < 40 || H < 40) return null;

  const dst = [0, 0, W - 1, 0, W - 1, H - 1, 0, H - 1];
  const warps = [];
  for (const f of frames) {
    const Hm = computeHomography(f.quad, dst);
    const w = warpBilinear(f.rgb, f.h, f.w, 3, Hm, H, W);
    const g = toGray(w, W * H);
    warps.push({ rgb: w, g, sharp: lapVar(g, W, H) });
  }

  // reference = sharpest
  let ref = warps[0];
  for (const w of warps) if (w.sharp > ref.sharp) ref = w;

  // per-frame aligned low-frequency fields
  const r = Math.max(3, (Math.max(W, H) / 90) | 0);
  const lowsR = [], lowsG = [], lowsB = [];
  for (const w of warps) {
    let dx = 0, dy = 0;
    if (w !== ref) {
      ({ dx, dy } = bestShift(ref.g, w.g, W, H));
      if (Math.abs(dx) > 0.1 * W || Math.abs(dy) > 0.1 * H) continue; // outlier frame
    }
    // shifted per-channel planes
    const cr = new Float32Array(W * H), cg = new Float32Array(W * H), cb = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      const sy = Math.min(H - 1, Math.max(0, y + dy));
      for (let x = 0; x < W; x++) {
        const sx = Math.min(W - 1, Math.max(0, x + dx));
        const si = (sy * W + sx) * 3, di = y * W + x;
        cr[di] = w.rgb[si]; cg[di] = w.rgb[si + 1]; cb[di] = w.rgb[si + 2];
      }
    }
    lowsR.push(boxBlur3(cr, W, H, r));
    lowsG.push(boxBlur3(cg, W, H, r));
    lowsB.push(boxBlur3(cb, W, H, r));
  }
  const N = lowsR.length;
  if (!N) return null;

  // per-pixel 70th percentile of the illumination fields
  const k = Math.min(N - 1, Math.floor(0.7 * (N - 1) + 0.5));
  const vals = new Float32Array(N);
  const pick = (lows, i) => {
    for (let f = 0; f < N; f++) vals[f] = lows[f][i];
    // insertion sort — N <= ~10
    for (let a = 1; a < N; a++) {
      const v = vals[a]; let b = a - 1;
      while (b >= 0 && vals[b] > v) { vals[b + 1] = vals[b]; b--; }
      vals[b + 1] = v;
    }
    return vals[k];
  };

  const refLowR = boxBlur3(toChannel(ref.rgb, W * H, 0), W, H, r);
  const refLowG = boxBlur3(toChannel(ref.rgb, W * H, 1), W, H, r);
  const refLowB = boxBlur3(toChannel(ref.rgb, W * H, 2), W, H, r);
  const fused = new Uint8ClampedArray(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    fused[i * 3]     = ref.rgb[i * 3]     * (pick(lowsR, i) / Math.max(1, refLowR[i]));
    fused[i * 3 + 1] = ref.rgb[i * 3 + 1] * (pick(lowsG, i) / Math.max(1, refLowG[i]));
    fused[i * 3 + 2] = ref.rgb[i * 3 + 2] * (pick(lowsB, i) / Math.max(1, refLowB[i]));
  }
  // Cosmetic-only final passes (ocrSource stays the raw fusion: measured
  // repeatedly, EVERY pixel intervention before Apple Vision is net-
  // negative for OCR — Vision wants raw pixels):
  //   1. straighten: deskew (projection-sharpness angle) + baseline curl
  //      flatten (per-text-line glyph-bottom fits), both gated.
  //   2. whiten creases or binarize.
  const straight = straighten(fused, W, H);
  const out = opts.bw ? binarize(straight, W, H) : whiten(straight, W, H);
  return { warped: out, ocrSource: fused, w: W, h: H, frames: N };
}

function toChannel(rgb, n, c) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = rgb[i * 3 + c];
  return out;
}

// Separable grayscale max filter (dilation), radius r.
function maxFilter(src, W, H, r) {
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      let m = 0;
      for (let k = Math.max(0, x - r); k <= Math.min(W - 1, x + r); k++)
        if (src[row + k] > m) m = src[row + k];
      tmp[row + x] = m;
    }
  }
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) {
      let m = 0;
      for (let k = Math.max(0, y - r); k <= Math.min(H - 1, y + r); k++)
        if (tmp[k * W + x] > m) m = tmp[k * W + x];
      out[y * W + x] = m;
    }
  }
  return out;
}

// Crease/shading whitening: creases are LOW-contrast dark structure while
// ink is HIGH-contrast, so a Sauvola-style local threshold separates them.
// Pixels classified paper are replaced by a local paper estimate
// (max-filter kills dark strokes, blur smooths); ink pixels pass through.
// Validated on the 20260722 burst: creases vanish, OCR conf unchanged.
function whiten(rgb, W, H) {
  const n = W * H;
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) gray[i] = (rgb[i * 3] + rgb[i * 3 + 1] + rgb[i * 3 + 2]) / 3;
  const win = 12; // ~2x text stroke spacing at receipt scale
  const mu = boxBlur3(gray, W, H, win);
  const g2 = new Float32Array(n);
  for (let i = 0; i < n; i++) g2[i] = gray[i] * gray[i];
  const mu2 = boxBlur3(g2, W, H, win);
  const ink = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const sd = Math.sqrt(Math.max(0, mu2[i] - mu[i] * mu[i]));
    const thr = mu[i] * (1 + 0.18 * (sd / 128 - 1));
    ink[i] = Math.min(1, Math.max(0, (thr - gray[i]) / Math.max(0.35 * sd + 6, 1)));
  }
  const inkSoft = boxBlur3(maxFilter(ink, W, H, 1), W, H, 1); // keep glyph edges
  const out = new Uint8ClampedArray(n * 3);
  for (let c = 0; c < 3; c++) {
    const paper = boxBlur3(maxFilter(toChannel(rgb, n, c), W, H, 8), W, H, 8);
    for (let i = 0; i < n; i++) {
      const a = inkSoft[i];
      out[i * 3 + c] = rgb[i * 3 + c] * a + paper[i] * (1 - a);
    }
  }
  return out;
}
