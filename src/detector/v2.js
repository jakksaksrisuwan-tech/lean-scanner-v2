// Boundary by row/column scanning — works for convex blobs.
//
// For each ROW, find leftmost and rightmost ON pixel (the left and right
// edges of the doc at that y). For each COLUMN, find topmost and
// bottommost ON pixel (the top and bottom edges at that x). That gives
// us 4 polylines (one per side). Fit a line to each. Intersect the 4
// lines pairwise to get the 4 corners.
//
// This avoids Moore-neighbour tracing entirely and is robust on convex
// polygons (which the doc silhouette always is under our leam-in).
//
// One subtle point: doc is at most 320x240 in the simulator (or 640x480
// in real capture), so the per-row/column scan is cheap.

import { orderCorners, isConvex, quadAspect } from "./quad.js";

// ── 1. Image utils ──────────────────────────────────────────────────────
function toGray(imageData) {
  const W = imageData.width, H = imageData.height;
  const data = imageData.data;
  const gray = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3.0;
  }
  return { gray, W, H };
}

function downscaleGray(gray, W, H, longEdge) {
  const scale = longEdge / Math.max(W, H);
  if (scale >= 1.0) return { gray, W, H, scale: 1.0 };
  const newW = Math.max(1, Math.round(W * scale));
  const newH = Math.max(1, Math.round(H * scale));
  const out = new Float64Array(newW * newH);
  for (let y = 0; y < newH; y++) {
    const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < newW; x++) {
      const sx = Math.min(W - 1, Math.floor(x / scale));
      out[y * newW + x] = gray[sy * W + sx];
    }
  }
  return { gray: out, W: newW, H: newH, scale };
}

function median(values) {
  const s = values.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

// ── 2. Binary morphology ────────────────────────────────────────────────
function _morph(mask, W, H, k, mode) {
  const out = new Uint8Array(W * H);
  const r = (k - 1) >> 1;
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r);
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r);
      let v;
      if (mode === "erode") {
        v = 1;
        outer: for (let yy = y0; yy <= y1; yy++)
          for (let xx = x0; xx <= x1; xx++)
            if (!mask[yy * W + xx]) { v = 0; break outer; }
      } else {
        v = 0;
        outer: for (let yy = y0; yy <= y1; yy++)
          for (let xx = x0; xx <= x1; xx++)
            if (mask[yy * W + xx]) { v = 1; break outer; }
      }
      out[y * W + x] = v;
    }
  }
  return out;
}
function erode(mask, W, H, k)  { return _morph(mask, W, H, k, "erode"); }
function dilate(mask, W, H, k) { return _morph(mask, W, H, k, "dilate"); }
function morphOpen(mask, W, H, k)  { return dilate(erode(mask, W, H, k), W, H, k); }
function morphClose(mask, W, H, k) { return erode(dilate(mask, W, H, k), W, H, k); }
// For each ROW, find left+right edge (the leftmost and rightmost ON
// pixels). For each COL, find top+bot edge. For the col case we scan
// past internal "holes" (title-bar segments) — keep tracking lastY
// even after an OFF.
function sampleEdges(mask, W, H, minRun = 4) {
  // Per row: leftmost/rightmost ON pixel
  const left = [];
  const right = [];
  for (let y = 0; y < H; y++) {
    let firstX = -1, lastX = -1, run = 0;
    for (let x = 0; x < W; x++) {
      if (mask[y * W + x]) {
        if (firstX < 0) firstX = x;
        lastX = x;
        run++;
      }
    }
    if (firstX >= 0 && run >= minRun) {
      left.push([firstX, y]);
      right.push([lastX, y]);
    }
  }
  // Per col: topmost/bottommost ON pixel
  const top = [];
  const bot = [];
  for (let x = 0; x < W; x++) {
    let firstY = -1, lastY = -1;
    for (let y = 0; y < H; y++) {
      if (mask[y * W + x]) {
        if (firstY < 0) firstY = y;
        lastY = y;
      }
    }
    if (firstY >= 0 && lastY - firstY >= minRun) {
      top.push([x, firstY]);
      bot.push([x, lastY]);
    }
  }
  return { left, right, top, bot };
}

// ── 4. Line fit (perpendicular least squares via SVD of centered) ─────
function fitLine(pts) {
  const n = pts.length;
  if (n < 2) return null;
  let cx = 0, cy = 0;
  for (const [x, y] of pts) { cx += x; cy += y; }
  cx /= n; cy /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const [x, y] of pts) {
    const dx = x - cx, dy = y - cy;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  // Special case: axis-aligned cluster. The smaller-variance direction is
  // the LINE direction (because perpendicular variance is what line fitting
  // actually minimizes, and tightly-clustered coords means the fit goes
  // along the OTHER axis).
  //   sxx small → x tight → line is vertical (x = cx), normal (1, 0)
  //   syy small → y tight → line is horizontal (y = cy), normal (0, 1)
  if (Math.abs(sxy) < 1e-9) {
    if (sxx <= syy) return { a: 1, b: 0, c: -cx };
    return { a: 0, b: 1, c: -cy };
  }
  // Diagonal case: eigenvector of [[sxx, sxy], [sxy, syy]] with larger eigenvalue.
  const t = (sxx + syy) / 2;
  const disc = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
  let eig1x = sxy, eig1y = (t + disc) - sxx;
  const len = Math.hypot(eig1x, eig1y);
  if (len < 1e-9) return null;
  const dx = eig1x / len, dy = eig1y / len;
  // Normal = (dy, -dx)
  const a = dy, b = -dx;
  const c = -(a * cx + b * cy);
  return { a, b, c };
}

function intersect(L1, L2) {
  const a1 = L1.a, b1 = L1.b, c1 = L1.c;
  const a2 = L2.a, b2 = L2.b, c2 = L2.c;
  const det = a1 * b2 - a2 * b1;
  if (Math.abs(det) < 1e-9) return null;
  const x = (b1 * c2 - b2 * c1) / det;
  const y = (a2 * c1 - a1 * c2) / det;
  return [x, y];
}

// ── Public entry ────────────────────────────────────────────────────────
export function detectHighContrast(imageData, cfg) {
  let { gray, W, H } = toGray(imageData);
  const ds = downscaleGray(gray, W, H, cfg.longEdge);
  gray = ds.gray; W = ds.W; H = ds.H;
  const scale = ds.scale;

  // bg luma via sampled median
  const sample = Math.min(4000, W * H);
  const stride = Math.max(1, Math.floor((W * H) / sample));
  const sampleVals = [];
  for (let i = 0; i < W * H; i += stride) sampleVals.push(gray[i]);
  const bgLuma = median(sampleVals);
  if (bgLuma > 180) return null;

  const lthr = bgLuma + 30;
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (gray[i] >= lthr) mask[i] = 1;
  let lit = 0;
  for (let i = 0; i < W * H; i++) if (mask[i]) lit++;
  if (lit < 200) return null;

  let m1 = morphOpen(mask, W, H, 5);
  let m2 = morphClose(m1, W, H, 9);

  let count = 0;
  for (let i = 0; i < W * H; i++) if (m2[i]) count++;
  if (count < 500) return null;

  // Check overall fill ratio too — we don't want false positives
  // on a whole-frame-bright input that snuck through.
  if (count > 0.95 * W * H) return null;

  // Sample edges per row/column with a min-run threshold to avoid
  // edges from tiny noise components.
  const edges = sampleEdges(m2, W, H, 8);
  if (edges.left.length < 8 || edges.right.length < 8 ||
      edges.top.length < 8 || edges.bot.length < 8) {
    return null;
  }
  // Take a STRATIFIED SUBSAMPLE of each side (every ~5th point) so the
  // line fit doesn't get hung up on a few outliers.
  function sub(p, k = 5) {
    const out = [];
    for (let i = 0; i < p.length; i += k) out.push(p[i]);
    return out;
  }
  const L_top = fitLine(sub(edges.top));
  const L_bot = fitLine(sub(edges.bot));
  const L_left = fitLine(sub(edges.left));
  const L_right = fitLine(sub(edges.right));
  if (!L_top || !L_bot || !L_left || !L_right) return null;

  // Intersect consecutive lines (clockwise: TL = L_top ∩ L_left,
  // TR = L_top ∩ L_right, BR = L_bot ∩ L_right, BL = L_bot ∩ L_left)
  const p_TL = intersect(L_left, L_top);
  const p_TR = intersect(L_top, L_right);
  const p_BR = intersect(L_right, L_bot);
  const p_BL = intersect(L_bot, L_left);
  if (!p_TL || !p_TR || !p_BR || !p_BL) return null;

  const ordered = [p_TL, p_TR, p_BR, p_BL];
  const flat = [];
  for (const [x, y] of ordered) flat.push(x, y);

  if (!isConvex(flat)) return null;
  const aspect = quadAspect(flat);
  if (!(cfg.minAspect < aspect && aspect < cfg.maxAspect)) return null;

  // Edge-touching filter: real receipts don't touch 2+ image edges.
  // The JS algorithm is fundamentally different (line-fit on per-row/col
  // samples) but the bounding box can still span the frame edge when the
  // receipt edge is near the frame border. Reject such cases.
  // Translate the flat 8-array → bounding box. ordered is flat 8 in
  // TL/TR/BR/BL order.
  let bx0 = flat[0], by0 = flat[1], bx1 = flat[0], by1 = flat[1];
  for (let i = 0; i < 4; i++) {
    const x = flat[i * 2], y = flat[i * 2 + 1];
    if (x < bx0) bx0 = x; if (x > bx1) bx1 = x;
    if (y < by0) by0 = y; if (y > by1) by1 = y;
  }
  let edgeTouch = 0;
  if (bx0 <= 4 * scale) edgeTouch++;
  if (by0 <= 4 * scale) edgeTouch++;
  if (bx1 >= (W - 4) * scale) edgeTouch++;
  if (by1 >= (H - 4) * scale) edgeTouch++;
  if (edgeTouch >= 2) return null;

  // Scale back to original resolution
  if (scale < 1.0) {
    for (let i = 0; i < 8; i++) flat[i] = flat[i] / scale;
  }
  const ordRes = orderCorners(flat);

  // Compute centroid as mean of corners
  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += ordRes[i * 2]; cy += ordRes[i * 2 + 1]; }
  cx /= 4; cy /= 4;

  // 24x24 segmentation mask (downsampled binarized luma for UI)
  const G = 24;
  const seg = new Float64Array(G * G);
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const y0 = Math.floor(gy / G * H);
      const y1 = Math.min(H, Math.floor((gy + 1) / G * H));
      const x0 = Math.floor(gx / G * W);
      const x1 = Math.min(W, Math.floor((gx + 1) / G * W));
      let total = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) {
        for (let xx = x0; xx < x1; xx++) {
          total += m2[yy * W + xx];
          n++;
        }
      }
      seg[gy * G + gx] = n ? total / n : 0;
    }
  }

  // Confidence: 1 if the lines fit cleanly (low residual), less if noisy.
  // Rough proxy: fraction of edge samples within 4 px of their line fit.
  function residual(line, pts) {
    let bad = 0;
    for (const [x, y] of pts) {
      const d = Math.abs(line.a * x + line.b * y + line.c);
      if (d > 4) bad++;
    }
    return 1 - (bad / pts.length);
  }
  const confs = [
    residual(L_top, edges.top),
    residual(L_bot, edges.bot),
    residual(L_left, edges.left),
    residual(L_right, edges.right),
  ];
  const conf = confs.reduce((a, b) => a + b, 0) / 4;
  return {
    quad: ordRes,
    confidence: Math.max(0, Math.min(1, conf)),
    segmentation: seg,
    cx, cy,
  };
}

export function detectV2(imageData, cfg) {
  return detectHighContrast(imageData, cfg);
}
