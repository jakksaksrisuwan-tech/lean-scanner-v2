// v3 detector: whiteness-based. Paper is desaturated (R≈G≈B all high),
// wood/bg is saturated (blue channel low). So min(R,G,B) separates paper
// from wood far better than luma — validated 29/29 on the real phone
// corpus in captures-debug/ (v2 luma approach scored 11/29).
//
// Pipeline: min-channel → Otsu threshold → 5×5 open → biggest
// 8-connected component → min-area rotated rect (hull + rotating
// calipers). No tuned constants except sanity limits.

import { orderCorners } from "./quad.js";

// Downscale-then-split in one pass: nearest-neighbour sample the RGBA
// frame at the target size and emit min/max channel planes. Never touches
// full-res pixels beyond the sampled ones, so a 4K camera frame costs the
// same as a VGA one.
function downscaleMinMax(imageData, longEdge) {
  const W = imageData.width, H = imageData.height, d = imageData.data;
  const scale = Math.min(1.0, longEdge / Math.max(W, H));
  const nW = Math.max(1, Math.round(W * scale)), nH = Math.max(1, Math.round(H * scale));
  const mn = new Uint8Array(nW * nH), mx = new Uint8Array(nW * nH);
  for (let y = 0; y < nH; y++) {
    const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < nW; x++) {
      const si = (sy * W + Math.min(W - 1, Math.floor(x / scale))) * 4;
      const r = d[si], g = d[si + 1], b = d[si + 2];
      const i = y * nW + x;
      mn[i] = r < g ? (r < b ? r : b) : (g < b ? g : b);
      mx[i] = r > g ? (r > b ? r : b) : (g > b ? g : b);
    }
  }
  return { mn, mx, W: nW, H: nH, scale };
}

function otsu(ch) {
  const hist = new Float64Array(256);
  for (let i = 0; i < ch.length; i++) hist[ch[i]]++;
  const total = ch.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, best = 0, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

// Separable box morphology via prefix sums — O(N) regardless of kernel
// size. A box erode/dilate decomposes exactly into a row pass then a
// column pass, so results are identical to the naive window scan.
function _morph(mask, W, H, k, isErode) {
  const r = (k - 1) >> 1;
  const tmp = new Uint8Array(W * H);
  const preR = new Int32Array(W + 1);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) preR[x + 1] = preR[x] + mask[row + x];
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r);
      const s = preR[x1 + 1] - preR[x0];
      tmp[row + x] = isErode ? (s === x1 - x0 + 1 ? 1 : 0) : (s > 0 ? 1 : 0);
    }
  }
  const out = new Uint8Array(W * H);
  const preC = new Int32Array(H + 1);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) preC[y + 1] = preC[y] + tmp[y * W + x];
    for (let y = 0; y < H; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r);
      const s = preC[y1 + 1] - preC[y0];
      out[y * W + x] = isErode ? (s === y1 - y0 + 1 ? 1 : 0) : (s > 0 ? 1 : 0);
    }
  }
  return out;
}
function morphOpen(mask, W, H, k) {
  return _morph(_morph(mask, W, H, k, true), W, H, k, false);
}

// Windowed sum of a 0/1 mask via prefix sums (radius r box).
function boxSum(mask, W, H, r) {
  const pre = new Int32Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let rowAcc = 0;
    for (let x = 0; x < W; x++) {
      rowAcc += mask[y * W + x];
      pre[(y + 1) * (W + 1) + x + 1] = pre[y * (W + 1) + x + 1] + rowAcc;
    }
  }
  const out = new Int32Array(W * H);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r) + 1;
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r) + 1;
      out[y * W + x] = pre[y1 * (W + 1) + x1] - pre[y0 * (W + 1) + x1]
                     - pre[y1 * (W + 1) + x0] + pre[y0 * (W + 1) + x0];
    }
  }
  return out;
}

// 8-connected components, largest first (top `max`); each { pixels, area }.
function components(mask, W, H, max = 5) {
  const label = new Int32Array(W * H); // 0 = unvisited
  const stack = new Int32Array(W * H);
  const out = [];
  let next = 1;
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || label[start]) continue;
    let top = 0, area = 0;
    const pixels = [];
    stack[top++] = start;
    label[start] = next;
    while (top > 0) {
      const i = stack[--top];
      const x = i % W, y = (i / W) | 0;
      pixels.push(x, y);
      area++;
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
  }
  out.sort((a, b) => b.area - a.area);
  return out.slice(0, max);
}
function biggestComponent(mask, W, H) {
  const c = components(mask, W, H, 1);
  return c.length ? c[0] : null;
}

// Andrew monotone chain convex hull. pts = flat [x,y,...]; returns [[x,y],...] CCW.
function convexHull(flat) {
  const pts = [];
  for (let i = 0; i < flat.length; i += 2) pts.push([flat[i], flat[i + 1]]);
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

// Max-area quadrilateral inscribed in the convex hull. A receipt seen at
// an angle is a trapezoid, not a rectangle — minAreaRect circumscribes and
// pushes corners into the bg. The max-area 4-vertex subset of the hull
// lands on the true corners instead.
// ponytail: brute force O(n^4) over decimated hull (≤40 verts → ≤92k combos)
function maxAreaQuad(hull) {
  let h = hull;
  if (h.length > 40) {
    const step = Math.ceil(h.length / 40);
    h = h.filter((_, i) => i % step === 0);
  }
  const n = h.length;
  if (n < 4) return null;
  const tri = (a, b, c) =>
    Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
  let best = null, bestA = -1;
  for (let i = 0; i < n - 3; i++)
    for (let j = i + 1; j < n - 2; j++)
      for (let k = j + 1; k < n - 1; k++)
        for (let l = k + 1; l < n; l++) {
          const a = tri(h[i], h[j], h[k]) + tri(h[i], h[k], h[l]);
          if (a > bestA) { bestA = a; best = [h[i], h[j], h[k], h[l]]; }
        }
  return { quad: best, area: bestA / 2 };
}

// Refine each quad side with a least-squares line over nearby boundary
// points (crumpled edges make single hull vertices noisy), then intersect
// adjacent lines for the final corners.
function refineQuad(quad, boundary) {
  const lines = [];
  for (let k = 0; k < 4; k++) {
    const p0 = quad[k], p1 = quad[(k + 1) % 4];
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    const L = Math.hypot(dx, dy);
    if (L < 1) return null;
    const ux = dx / L, uy = dy / L;         // side direction
    const nx = -uy, ny = ux;                // side normal
    let sx = 0, sy = 0, cnt = 0;
    const sel = [];
    for (let i = 0; i < boundary.length; i += 2) {
      const rx = boundary[i] - p0[0], ry = boundary[i + 1] - p0[1];
      const t = rx * ux + ry * uy;
      if (t < 0.1 * L || t > 0.9 * L) continue;
      if (Math.abs(rx * nx + ry * ny) > 5) continue;
      sel.push(boundary[i], boundary[i + 1]);
      sx += boundary[i]; sy += boundary[i + 1]; cnt++;
    }
    if (cnt < 8) { lines.push({ nx, ny, d: nx * p0[0] + ny * p0[1] }); continue; }
    const mx = sx / cnt, my = sy / cnt;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < sel.length; i += 2) {
      const ex = sel[i] - mx, ey = sel[i + 1] - my;
      sxx += ex * ex; sxy += ex * ey; syy += ey * ey;
    }
    // principal direction of the scatter = refined side direction
    const t = (sxx + syy) / 2;
    const disc = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
    let vx, vy;
    if (Math.abs(sxy) < 1e-9) { [vx, vy] = sxx >= syy ? [1, 0] : [0, 1]; }
    else { vx = sxy; vy = (t + disc) - sxx; const l2 = Math.hypot(vx, vy); vx /= l2; vy /= l2; }
    const rnx = -vy, rny = vx;
    lines.push({ nx: rnx, ny: rny, d: rnx * mx + rny * my });
  }
  const out = [];
  for (let k = 0; k < 4; k++) {
    const A = lines[k], B = lines[(k + 1) % 4];
    const det = A.nx * B.ny - B.nx * A.ny;
    if (Math.abs(det) < 1e-9) return null;
    out.push((A.d * B.ny - B.d * A.ny) / det, (A.nx * B.d - B.nx * A.d) / det);
  }
  return out;
}

export function detectWhiteness(imageData, cfg) {
  const ds = downscaleMinMax(imageData, cfg.longEdge);
  const { mn: ch, mx, W, H, scale } = ds;

  const thr = otsu(ch);
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (ch[i] > thr) mask[i] = 1;

  const opened = morphOpen(mask, W, H, 5);
  let totMn = 0;
  for (let i = 0; i < W * H; i++) totMn += ch[i];

  // Score every sizable candidate blob and pick the most paper-like —
  // NEVER just the biggest. A sun-glare patch on polished wood is bright,
  // desaturated, and often bigger than the receipt; what tells them apart
  // is the boundary: paper is a step edge (sharp ≈ 1.0), glare fades in
  // (sharp ≈ 0.6). Measured on the glare corpus of 2026-07-21.
  let best = null;
  for (const comp of components(opened, W, H, 5)) {
    if (comp.area < 0.02 * W * H) break;      // sorted desc — rest are smaller
    if (comp.area > 0.90 * W * H) continue;   // whole-frame blob: nothing to align

    // Paperness: desaturated + clearly brighter than bg. Thresholds from
    // 35 receipt-free negative crops (34 → 0 false fires) + 35 positives.
    let fgMn = 0, fgSat = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const p = comp.pixels[i + 1] * W + comp.pixels[i];
      fgMn += ch[p];
      fgSat += mx[p] > 0 ? (mx[p] - ch[p]) / mx[p] : 0;
    }
    fgMn /= comp.area; fgSat /= comp.area;
    const bgN = W * H - comp.area;
    const bgMn = bgN > 0 ? (totMn - fgMn * comp.area) / bgN : 0;
    const sep = fgMn - bgMn;
    if (fgSat > 0.30 || sep < 50) continue;

    // Boundary pixels (pixel with an off- or out-of-frame neighbor).
    const boundary = [];
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const x = comp.pixels[i], y = comp.pixels[i + 1];
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1 ||
          !opened[(y - 1) * W + x] || !opened[(y + 1) * W + x] ||
          !opened[y * W + x - 1] || !opened[y * W + x + 1]) {
        boundary.push(x, y);
      }
    }
    // Sharpness on the OUTER silhouette only (row/col extremes). The
    // boundary list above also contains interior hole edges around text
    // glyphs, which have no gradient and would dilute the measure.
    const rowMin = new Int16Array(H).fill(-1), rowMax = new Int16Array(H).fill(-1);
    const colMin = new Int16Array(W).fill(-1), colMax = new Int16Array(W).fill(-1);
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const x = comp.pixels[i], y = comp.pixels[i + 1];
      if (rowMin[y] < 0 || x < rowMin[y]) rowMin[y] = x;
      if (x > rowMax[y]) rowMax[y] = x;
      if (colMin[x] < 0 || y < colMin[x]) colMin[x] = y;
      if (y > colMax[x]) colMax[x] = y;
    }
    let sharpN = 0, sharpT = 0;
    const step = (x, y) => {
      const xl = Math.max(0, x - 3), xr = Math.min(W - 1, x + 3);
      const yu = Math.max(0, y - 3), yd = Math.min(H - 1, y + 3);
      return Math.max(Math.abs(ch[y * W + xr] - ch[y * W + xl]),
                      Math.abs(ch[yd * W + x] - ch[yu * W + x]));
    };
    for (let y = 0; y < H; y++) {
      if (rowMin[y] < 0) continue;
      sharpT += 2;
      if (step(rowMin[y], y) > 25) sharpN++;
      if (step(rowMax[y], y) > 25) sharpN++;
    }
    for (let x = 0; x < W; x++) {
      if (colMin[x] < 0) continue;
      sharpT += 2;
      if (step(x, colMin[x]) > 25) sharpN++;
      if (step(x, colMax[x]) > 25) sharpN++;
    }
    const sharp = sharpT ? sharpN / sharpT : 0;
    // Small blobs must have a crisp boundary: every false-fire blob in the
    // negatives corpus is small AND soft (sharp <= 0.72); real receipts
    // measure >= 0.75. Big blobs are exempt — SRD white-on-white receipts
    // fill the frame with soft boundaries and are still real.
    if (comp.area < 0.15 * W * H && sharp < 0.74) continue;
    let cmx = 0, cmy = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) { cmx += comp.pixels[i]; cmy += comp.pixels[i + 1]; }
    cmx /= comp.area; cmy /= comp.area;
    const dc = Math.hypot(cmx - W / 2, cmy - H / 2) / (0.5 * Math.hypot(W, H));
    const score = (2 * sharp + Math.min(1.5, sep / 100)) * (1 - 0.4 * dc);
    if (!best || score > best.score) best = { comp, boundary, score };
  }
  if (!best) return null;
  const { comp, boundary } = best;
  const hull = convexHull(boundary);
  if (hull.length < 4) return null;
  const maq = maxAreaQuad(hull);
  if (!maq) return null;
  const refined = refineQuad(maq.quad, boundary) ??
    maq.quad.flat();

  // shoelace area of the refined quad
  let qa = 0;
  for (let k = 0; k < 4; k++) {
    const k2 = (k + 1) % 4;
    qa += refined[k * 2] * refined[k2 * 2 + 1] - refined[k2 * 2] * refined[k * 2 + 1];
  }
  qa = Math.abs(qa / 2);

  // Solidity: blob fills its quad ⇒ it's really a paper sheet,
  // not a sprawl of merged highlights.
  const solidity = comp.area / Math.max(1, qa);
  if (solidity < 0.6) return null;

  // No edge-touch filter: a receipt running past the frame edge is a valid
  // live-scanner detection (the quality scorer penalizes framing). The
  // merged-blob failure it used to guard is covered by the whole-frame and
  // solidity checks — validated on the SRD 200-receipt corpus (26% → 93%).

  const flat = refined.slice();
  if (scale < 1.0) for (let i = 0; i < 8; i++) flat[i] /= scale;
  const quad = orderCorners(flat);

  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += quad[i * 2]; cy += quad[i * 2 + 1]; }
  cx /= 4; cy /= 4;

  // 24×24 segmentation grid for UI (fraction of ON pixels per cell)
  const G = 24;
  const seg = new Float64Array(G * G);
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const y0 = Math.floor(gy / G * H), y1 = Math.min(H, Math.floor((gy + 1) / G * H));
      const x0 = Math.floor(gx / G * W), x1 = Math.min(W, Math.floor((gx + 1) / G * W));
      let t = 0, n = 0;
      for (let yy = y0; yy < y1; yy++)
        for (let xx = x0; xx < x1; xx++) { t += opened[yy * W + xx]; n++; }
      seg[gy * G + gx] = n ? t / n : 0;
    }
  }

  return { quad, confidence: Math.min(1, solidity), segmentation: seg, cx, cy };
}

// Ink-density fallback for light backgrounds (white wall, steel table):
// there the paper isn't brighter or less saturated than the bg, so the
// whiteness path has nothing to grab — but the document is packed with
// text/grid strokes while the bg is smooth. Gradient-dense blob = doc.
// Runs at 320px (7×7 morphology is too slow at 640 in JS) and ONLY on
// globally desaturated scenes, so gradient-rich wood grain never enters.
function detectInk(imageData, cfg) {
  const ds = downscaleMinMax(imageData, 320);
  const { mn, mx, W, H, scale } = ds;
  // gradient magnitude |dx|+|dy| on the min channel
  const mag = new Float32Array(W * H);
  let magMax = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const m = Math.abs(mn[i + 1] - mn[i - 1]) + Math.abs(mn[i + W] - mn[i - W]);
      mag[i] = m;
      if (m > magMax) magMax = m;
    }
  }
  // 92nd percentile via histogram
  const bins = new Int32Array(256);
  for (let i = 0; i < mag.length; i++) bins[Math.min(255, mag[i] | 0)]++;
  let acc = 0, p92 = 30;
  for (let b = 0; b < 256; b++) { acc += bins[b]; if (acc >= 0.92 * mag.length) { p92 = b; break; } }
  const thr = Math.max(30, p92);
  const ink = new Uint8Array(W * H);
  for (let i = 0; i < mag.length; i++) if (mag[i] > thr) ink[i] = 1;

  // Density filter: text is gradient-DENSE (packed glyphs), while tile
  // grout, cables, and table edges are isolated high-gradient LINES.
  // Drop ink pixels whose 17px neighborhood is mostly empty — otherwise
  // dilation welds grout lines into the doc blob and the hull runs away
  // along the seams (20260723 battery-pack-on-tile corpus).
  // Wide window: a grout/cable NETWORK stays sparse at 33px scale while a
  // document region (glyphs + rules mixed) stays moderately dense. Small
  // windows can't tell a grout line from a form's grid line — both are
  // thin — so the context radius is what separates them.
  const dens = boxSum(ink, W, H, 16);
  const area33 = 33 * 33;
  for (let i = 0; i < ink.length; i++) {
    if (ink[i] && dens[i] < 0.09 * area33) ink[i] = 0;
  }

  const big = _morph(_morph(ink, W, H, 7, false), W, H, 7, false);   // dilate x2
  const core = _morph(_morph(big, W, H, 7, true), W, H, 7, true);    // erode x2
  // Candidate selection with a soft center prior: the user aims the
  // subject roughly at the frame center; a same-sized blob at the edge
  // (tile seams, table edge) loses to a centered one, but a decisively
  // bigger off-center blob still wins (prior, not constraint).
  let comp = null, bestScore = -1;
  for (const c of components(core, W, H, 5)) {
    if (c.area < 0.02 * W * H) break;
    if (c.area > 0.90 * W * H) continue;
    let mx = 0, my = 0;
    for (let i = 0; i < c.pixels.length; i += 2) { mx += c.pixels[i]; my += c.pixels[i + 1]; }
    mx /= c.area; my /= c.area;
    const dc = Math.hypot(mx - W / 2, my - H / 2) / (0.5 * Math.hypot(W, H));
    const score = c.area * (1 - 0.7 * dc);
    if (score > bestScore) { bestScore = score; comp = c; }
  }
  if (!comp) return null;
  // Mahalanobis trim: a grout/cable tail welded to the doc blob is a
  // small pixel mass far from the blob's center — clip >2.5 sigma before
  // the hull so maxAreaQuad can't pick tail points as corners.
  // Iterated: a heavy tail inflates the covariance on pass 1, hiding
  // itself; re-estimating on the trimmed set converges in 2-3 passes.
  for (let pass = 0; pass < 3; pass++) {
    let mx = 0, my = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) { mx += comp.pixels[i]; my += comp.pixels[i + 1]; }
    mx /= comp.area; my /= comp.area;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const ex = comp.pixels[i] - mx, ey = comp.pixels[i + 1] - my;
      sxx += ex * ex; sxy += ex * ey; syy += ey * ey;
    }
    sxx /= comp.area; sxy /= comp.area; syy /= comp.area;
    const det = sxx * syy - sxy * sxy;
    if (det <= 1e-6) break;
    const i11 = syy / det, i12 = -sxy / det, i22 = sxx / det;
    const kept = [];
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const ex = comp.pixels[i] - mx, ey = comp.pixels[i + 1] - my;
      const d2 = ex * (i11 * ex + i12 * ey) + ey * (i12 * ex + i22 * ey);
      if (d2 <= 2.5 * 2.5) kept.push(comp.pixels[i], comp.pixels[i + 1]);
    }
    if (kept.length < 16 || kept.length === comp.pixels.length) break;
    comp = { pixels: kept, area: kept.length / 2 };
  }

  // hull over ORIGINAL ink pixels inside the core component (undilated
  // points keep steel-scratch bloat out of the corners)
  const sel = [];
  for (let i = 0; i < comp.pixels.length; i += 2) {
    const x = comp.pixels[i], y = comp.pixels[i + 1];
    if (ink[y * W + x]) sel.push(x, y);
  }
  const hull = convexHull(sel.length >= 8 ? sel : comp.pixels);
  if (hull.length < 4) return null;
  const maq = maxAreaQuad(hull);
  if (!maq) return null;
  const quad = maq.quad;
  // doc-ness: the dilated ink must fill most of the quad
  const solidity = comp.area / Math.max(1, maq.area);
  if (solidity < 0.45) return null;

  // additive margin: erosion radius (7*2/2 per side ≈ 7px core loss after
  // symmetric open, measured) + paper border beyond the outermost strokes
  const MARGIN = 22;
  let cx = 0, cy = 0;
  for (const [x, y] of quad) { cx += x; cy += y; }
  cx /= 4; cy /= 4;
  const fw = imageData.width, fh = imageData.height;
  const flat = [];
  for (const [x, y] of quad) {
    const vx = x - cx, vy = y - cy;
    const L = Math.max(1, Math.hypot(vx, vy));
    flat.push(
      Math.min(fw - 1, Math.max(0, (cx + vx * (1 + MARGIN / L)) / scale)),
      Math.min(fh - 1, Math.max(0, (cy + vy * (1 + MARGIN / L)) / scale)));
  }
  const ordered = orderCorners(flat);
  let ocx = 0, ocy = 0;
  for (let i = 0; i < 4; i++) { ocx += ordered[i * 2]; ocy += ordered[i * 2 + 1]; }
  return { quad: ordered, confidence: Math.min(1, solidity), segmentation: null, cx: ocx / 4, cy: ocy / 4 };
}

export function detectV3(imageData, cfg) {
  // Scene gate: on globally desaturated scenes (steel table, white wall)
  // the whiteness signal is meaningless — paper is no brighter or whiter
  // than the bg — so the ink-density path leads and whiteness is the
  // fallback. On colored scenes (wood) whiteness leads and ink NEVER runs,
  // because gradient-rich grain would false-fire it.
  const ds = downscaleMinMax(imageData, 64);
  let sat = 0;
  for (let i = 0; i < ds.mn.length; i++) sat += ds.mx[i] > 0 ? (ds.mx[i] - ds.mn[i]) / ds.mx[i] : 0;
  // Two-axis scene gate (measured on the phone corpus):
  //   saturated bg (wood, sat>0.30)              -> whiteness only
  //   desaturated + BRIGHT bg (white wall/steel,
  //     median-mn>95: paper no brighter than bg) -> ink first
  //   desaturated + DARK bg (charcoal, median<95:
  //     whiteness signal is huge and stable)     -> whiteness first
  // Ink-first on dark scenes caused 180px corner jitter (20260722_142x):
  // text gradients merged with table scratches into runaway hulls.
  const desaturated = sat / ds.mn.length <= 0.30;
  const sorted = Array.from(ds.mn).sort((a, b) => a - b);
  const bgBright = sorted[sorted.length >> 1] > 95;
  if (desaturated && bgBright) {
    return detectInk(imageData, cfg) ?? detectWhiteness(imageData, cfg);
  }
  return detectWhiteness(imageData, cfg) ?? (desaturated ? detectInk(imageData, cfg) : null);
}
