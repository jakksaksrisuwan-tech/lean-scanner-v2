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
  // final pass: whiten crease shading (static, so the percentile can't
  // reject it — it moves WITH the paper, unlike shadows). Whitening is
  // cosmetic and costs a sliver of OCR confidence (0.39 -> 0.37 on the
  // 20260722 burst), so the pre-whiten fusion is returned alongside —
  // display/store `warped`, feed `ocrSource` to any text recognizer.
  const out = opts.bw ? binarize(fused, W, H) : whiten(fused, W, H);
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
