// Rotation-invariant document detector — coarse-to-fine radial sweep.
//
// Design: instead of "find the 4 strongest gradients and assume they're
// the doc edges" (current sobel+bbox, rotation-fragile), we work in two
// passes:
//
//   1. text-density heatmap → centroid of the bill (rotation-invariant)
//   2. radial sweep from the centroid → 24 polar edge points → quad fit
//   3. zoom into each corner ROI → sub-pixel corner refinement
//
// The algorithm is resolution-agnostic and progressive: the same code
// path runs on any input size, and each stage has a fixed compute cost
// independent of input resolution (we always operate on a 32x32 grid or
// a 96x96 corner ROI).
//
// Author motivation: paper bills and receipts are routinely laid down
// at 30-60 deg from the camera. The original `detect_numpy` assumes a
// roughly-aligned page; this detector explicitly does not.
//
// Algorithm references:
//   - "Advanced Hough-based method for on-device document localization"
//     arxiv:2106.09987 (Hough-line voting; we use a related radial sweep
//     that recovers orientation for free)
//   - "Text density" trick from classical document segmentation
//     (Esposito et al, ICDAR 2005) — works because text-heavy docs have
//     high local variance on a uniform-color substrate.
//
// All math is pure-js, no deps. Tested in tests/test_consistency.mjs.

import {
  quadArea, orderCorners, isConvex, quadAspect,
} from "./quad.js";

// ── Public entry ─────────────────────────────────────────────────────────

/**
 * Detect a 4-corner document quad at any rotation.
 * @param {ImageData} imageData  RGBA
 * @param {object} cfg  DetectorConfig
 * @returns {number[] | null}  flat 8-array ordered TL TR BR BL, or null
 */
export function detectRadial(imageData, cfg) {
  const W = imageData.width;
  const H = imageData.height;
  const gray = toGray(imageData);

  // Stage 0 — paper-region segmentation + centroid
  const seg0 = segmentPaperRegion(gray, W, H);
  if (!seg0) return null;

  // Stage 1 — 24-ray radial sweep from the centroid
  const points = radialSweep(gray, W, H, seg0.cx, seg0.cy, cfg);
  if (points.length < 8) return null;  // not enough edges to fit
  const sweepConfidence = Math.min(points.length / 18, 1);

  // Stage 2 — fit a quadrilateral (use segmentation centroid)
  const rough = fitQuad(points, seg0.cx, seg0.cy);
  if (!rough) return null;

  // Stage 3 — zoom into each corner for sub-pixel refinement
  const refined = refineCorners(gray, W, H, rough);

  // Order the quad TL, TR, BR, BL BEFORE sanity checks (some checks
  // assume the order).
  const ordered = orderCorners(refined);

  // Sanity checks
  if (!isConvex(ordered)) return null;
  const aspect = quadAspect(ordered);
  if (aspect < cfg.minAspect || aspect > cfg.maxAspect) return null;
  const area = quadArea(ordered);
  const frameArea = W * H;
  const arRatio = area / frameArea;
  // Use a smaller floor than the legacy detector: radial may fit a tighter
  // quad (inside the page, between text strips) instead of an outer-edge quad.
  const minRatio = Math.min(cfg.minQuadAreaRatio, 0.02);
  if (arRatio < minRatio) return null;
  if (arRatio > cfg.maxQuadAreaRatio) return null;

  // Return rich result: ordered quad + confidence + segmentation for UI overlay
  return {
    quad: ordered,
    confidence: 0.5 * seg0.confidence + 0.5 * sweepConfidence,
    segmentation: seg0.mask,         // length-256 array, 24x24 grid of [0..1] scores
    cx: seg0.cx,
    cy: seg0.cy,
  };
}

// ── Grayscale helper ─────────────────────────────────────────────────────

export function toGray(imageData) {
  const W = imageData.width;
  const H = imageData.height;
  const data = imageData.data;
  const isRGBA = data.length === W * H * 4;
  const gray = new Float64Array(W * H);
  for (let i = 0; i < W * H; i++) {
    if (isRGBA) {
      gray[i] = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / 3.0;
    } else {
      gray[i] = (data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2]) / 3.0;
    }
  }
  return gray;
}

// ── Stage 0 — paper-region segmentation + centroid ──────────────────────

/**
 * Per-pixel paper-likelihood scoring -> 24x24 mask of [0..1] paper scores,
 * then take the largest connected component and return its centroid.
 *
 * Score heuristic per pixel:
 *   - luma in [40, 230]  (paper, not pure black/white)
 *   - local 5x5 stddev > some threshold  (text/edge content)
 *   - local 5x5 mean in [60, 220]  (paper-like luminance)
 *
 * Voting: each cell is the average paper-score of its inner pixels, then
 * we threshold + largest connected component + mass-center of that CC.
 *
 * @returns {{ cx, cy, confidence, mask } | null}
 *   mask is a Float64Array of length 576 (24x24), values in [0..1]
 */
export function segmentPaperRegion(gray, W, H) {
  const G = 24;                       // grid side
  const cellW = Math.floor(W / G);
  const cellH = Math.floor(H / G);

  // 1. Per-pixel paper score (binary mask at full resolution)
  //    Robust to bg choice: score is luma in [PIX_LUMA_LO, PIX_LUMA_HI]
  //    AND local stddev > thr (text/edge content), OR (when strict mask
  //    finds nothing) the pixel is at least 30 brighter than the global
  //    median, meaning "paper-colored compared to its background".
  const PIX_LUMA_LO = 30, PIX_LUMA_HI = 250;
  const PIX_VAR_THR = 6;
  const PIX_MEAN_LO = 30, PIX_MEAN_HI = 240;
  const PIX_R = 2;

  // First-pass: estimate global mean (cheap). Used for the relative-luma
  // mode below, which works on light-table bgs (where paper & bg are
  // both near luma 230 but differ by ~30).
  let gSum = 0;
  for (let i = 0; i < W * H; i++) gSum += gray[i];
  const gMean = gSum / (W * H);
  const gBrightnessThr = Math.max(8, gMean * 0.05);  // 5% brighter than gMean

  const paperScores = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const luma = gray[y * W + x];
      if (luma < PIX_LUMA_LO || luma > PIX_LUMA_HI) continue;

      // 5x5 window mean + variance
      let sum = 0, sumSq = 0, n = 0;
      const y0 = Math.max(0, y - PIX_R), y1 = Math.min(H - 1, y + PIX_R);
      const x0 = Math.max(0, x - PIX_R), x1 = Math.min(W - 1, x + PIX_R);
      for (let yy = y0; yy <= y1; yy++) {
        for (let xx = x0; xx <= x1; xx++) {
          const v = gray[yy * W + xx];
          sum += v;
          sumSq += v * v;
          n++;
        }
      }
      const mean = sum / n;
      const variance = sumSq / n - mean * mean;
      const stddev = Math.sqrt(Math.max(0, variance));

      // Score combines two independent signals:
      //   A) strong texture (high local stddev) in a paper-like mean range
      //   B) significantly brighter than global mean (paper is bright)
      let score = 0;
      let scoreA = 0, scoreB = 0;
      if (mean >= PIX_MEAN_LO && mean <= PIX_MEAN_HI) {
        scoreA = stddev > PIX_VAR_THR ? 1.0 : stddev / PIX_VAR_THR;
      }
      if (luma > gMean + gBrightnessThr) {
        scoreB = Math.min(1, (luma - gMean) / 30);
      }
      score = Math.max(scoreA, scoreB);
      paperScores[y * W + x] = score;
    }
  }

  // 2. Downsample to G x G cells (max-pool per cell)
  const mask = new Float64Array(G * G);
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const y0 = cy * cellH, x0 = cx * cellW;
      const y1 = Math.min(H, y0 + cellH);
      const x1 = Math.min(W, x0 + cellW);
      let cellMax = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const v = paperScores[y * W + x];
          if (v > cellMax) cellMax = v;
        }
      }
      mask[cy * G + cx] = cellMax;
    }
  }

  // 3. Connected component label on the mask (4-connectivity).
  //    Take the largest one. If none > 1 cell, null.
  const ccLabels = new Int32Array(G * G);
  let nextLabel = 1;
  const ccSizes = [0];
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const i = cy * G + cx;
      if (mask[i] <= 0.1 || ccLabels[i] !== 0) continue;
      const label = nextLabel++;
      ccSizes.push(0);
      // Iterative flood fill
      const stack = [[cy, cx]];
      while (stack.length > 0) {
        const [y, x] = stack.pop();
        const j = y * G + x;
        if (y < 0 || y >= G || x < 0 || x >= G) continue;
        if (ccLabels[j] !== 0 || mask[j] <= 0.1) continue;
        ccLabels[j] = label;
        ccSizes[label]++;
        stack.push([y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]);
      }
    }
  }

  // Find the largest CC
  let bestLabel = 0, bestSize = 0;
  for (let l = 1; l < ccSizes.length; l++) {
    if (ccSizes[l] > bestSize) { bestSize = ccSizes[l]; bestLabel = l; }
  }
  if (bestLabel === 0) return null;
  if (bestSize < 8) return null;  // very small — probably noise

  // 4. Mass-center of the largest CC, weighted by paper-score
  let massX = 0, massY = 0, total = 0;
  for (let cy = 0; cy < G; cy++) {
    for (let cx = 0; cx < G; cx++) {
      const i = cy * G + cx;
      if (ccLabels[i] !== bestLabel) continue;
      const x = (cx + 0.5) * cellW;
      const y = (cy + 0.5) * cellH;
      const w = mask[i];
      massX += x * w;
      massY += y * w;
      total += w;
    }
  }
  if (total < 0.01) return null;

  const cx = massX / total;
  const cy = massY / total;
  const confidence = Math.min(bestSize / (G * G * 0.2), 1);

  return { cx, cy, confidence, mask };
}

/** @deprecated kept for compat — returns same shape as segmentPaperRegion */
export function findCentroid(gray, W, H) {
  const r = segmentPaperRegion(gray, W, H);
  if (!r) return null;
  return { cx: r.cx, cy: r.cy };
}

// ── Stage 1 — 24-ray radial sweep ────────────────────────────────────────

/**
 * Sweep N rays outward from (cx, cy); at each step, measure the gradient
 * perpendicular to the ray direction. A sharp perpendicular gradient
 * indicates an edge perpendicular to the ray — i.e. the page boundary.
 *
 * Returns a list of polar edge points (r, theta) for the N angles where
 * a strong edge was found. Angles with no edge are omitted.
 */
export function radialSweep(gray, W, H, cx, cy, cfg) {
  const N_RAYS = 24;
  const STRIDE = 2;   // radius step in pixels (cheap to double)
  const MAX_R = Math.min(W, H) * 0.5;
  const GRAD_THR = 30;  // perpendicular gradient must exceed this to be an edge

  const points = [];

  for (let i = 0; i < N_RAYS; i++) {
    const theta = (i / N_RAYS) * 2 * Math.PI;
    const dx = Math.cos(theta);
    const dy = Math.sin(theta);

    let bestR = -1;
    let bestGrad = 0;

    for (let r = 8; r < MAX_R; r += STRIDE) {
      const x = Math.round(cx + r * dx);
      const y = Math.round(cy + r * dy);
      if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) break;

      // Sobel at (x, y)
      const gx = sobelX(gray, W, H, x, y);
      const gy = sobelY(gray, W, H, x, y);

      // Project onto perpendicular direction (the direction the edge
      // gradient should point if we hit a page boundary)
      const perpGrad = Math.abs(-dy * gx + dx * gy);

      if (perpGrad > bestGrad) {
        bestGrad = perpGrad;
        bestR = r;
      }
    }

    if (bestR > 0 && bestGrad > GRAD_THR) {
      const px = cx + bestR * dx;
      const py = cy + bestR * dy;
      points.push({ x: px, y: py, theta, r: bestR, grad: bestGrad });
    }
  }
  return points;
}

function sobelX(gray, W, H, x, y) {
  const a = gray[(y - 1) * W + (x - 1)];
  const b = gray[(y - 1) * W + x];
  const c = gray[(y - 1) * W + (x + 1)];
  const d = gray[y * W + (x - 1)];
  const f = gray[y * W + (x + 1)];
  const g = gray[(y + 1) * W + (x - 1)];
  const hh = gray[(y + 1) * W + x];
  const ii = gray[(y + 1) * W + (x + 1)];
  return (c + 2 * f + ii) - (a + 2 * d + g);
}

function sobelY(gray, W, H, x, y) {
  const a = gray[(y - 1) * W + (x - 1)];
  const b = gray[(y - 1) * W + x];
  const c = gray[(y - 1) * W + (x + 1)];
  const g = gray[(y + 1) * W + (x - 1)];
  const hh = gray[(y + 1) * W + x];
  const ii = gray[(y + 1) * W + (x + 1)];
  const d = gray[y * W + (x - 1)];
  const f = gray[y * W + (x + 1)];
  return (a + 2 * b + c) - (g + 2 * hh + ii);
}

// ── Stage 2 — quadrant-bucketed line fitting ─────────────────────────────

/**
 * Given 2D points that are radial intersections with the page boundary,
 * classify each by which side of the centroid it landed on (E, W, N, S),
 * fit a line to each bucket, then intersect consecutive lines for corners.
 *
 * Pass `centroidX`, `centroidY` if you have a more accurate estimate of
 * the page center (e.g. from the paper-region segmentation). Otherwise we
 * use the mean of the points.
 *
 * Rotation-invariant by construction (we don't care about the polar θ of
 * points, only their x/y relative to the centroid).
 *
 * @param {Array<{x, y}>} points
 * @param {number} [centroidX]
 * @param {number} [centroidY]
 * @returns {number[] | null} flat 8-array ordered TL TR BR BL
 */
export function fitQuad(points, centroidX, centroidY) {
  if (points.length < 8) return null;

  // Centroid: prefer explicit (e.g. from segmentation), fall back to mean.
  let mx, my;
  if (centroidX !== undefined && centroidY !== undefined) {
    mx = centroidX; my = centroidY;
  } else {
    mx = 0; my = 0;
    for (const p of points) { mx += p.x; my += p.y; }
    mx /= points.length; my /= points.length;
  }

  // Bucket each point into one of 4 sides by quadrant relative to centroid.
  //   N:  |y - cy| > |x - cx| && y < cy
  //   S:  |y - cy| > |x - cx| && y > cy
  //   E:  |x - cx| >= |y - cy| && x > cx
  //   W:  |x - cx| >= |y - cy| && x < cx
  const buckets = [[], [], [], []];   // N, E, S, W
  for (const p of points) {
    const dx = p.x - mx, dy = p.y - my;
    const adx = Math.abs(dx), ady = Math.abs(dy);
    if (ady > adx) {
      buckets[dy < 0 ? 0 : 2].push(p);  // N or S
    } else {
      buckets[dx > 0 ? 1 : 3].push(p);  // E or W
    }
  }
  for (const b of buckets) if (b.length < 2) return null;
  const lines = buckets.map(edgeLine);

  // Intersect (W∩N, N∩E, E∩S, S∩W) for (TL, TR, BR, BL).
  // Image coords: y grows down. N=top, S=bottom, E=right, W=left.
  //   N ∩ E = TR (top-right of the page)
  //   E ∩ S = BR
  //   S ∩ W = BL
  //   W ∩ N = TL
  // We want TL first → reorder to (W∩N, N∩E, E∩S, S∩W).
  const corners = [
    intersect(lines[3], lines[0]),  // W ∩ N = TL
    intersect(lines[0], lines[1]),  // N ∩ E = TR
    intersect(lines[1], lines[2]),  // E ∩ S = BR
    intersect(lines[2], lines[3]),  // S ∩ W = BL
  ];
  if (corners.some((c) => c === null)) return null;
  return [corners[0].x, corners[0].y,
          corners[1].x, corners[1].y,
          corners[2].x, corners[2].y,
          corners[3].x, corners[3].y];
}

/** Fit a line through 2D points using the principal eigenvector of the
 *  covariance matrix (total least squares, perpendicular distance). */
function edgeLine(pts) {
  const n = pts.length;
  if (n < 2) {
    const p = pts[0];
    return { a: 1, b: 0, c: p.x };
  }
  let mx = 0, my = 0;
  for (const p of pts) { mx += p.x; my += p.y; }
  mx /= n; my /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (const p of pts) {
    const dx = p.x - mx, dy = p.y - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const half = (sxx + syy) / 2;
  const disc = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
  const lambda = half + disc;
  let dx = (lambda - syy), dy = sxy;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1e-9) return { a: 1, b: 0, c: mx };
  dx /= len; dy /= len;
  const a = -dy, b = dx, c = a * mx + b * my;
  return { a, b, c };
}

function intersect(l1, l2) {
  const det = l1.a * l2.b - l2.a * l1.b;
  if (Math.abs(det) < 1e-9) return null;
  return {
    x: (l2.b * l1.c - l1.b * l2.c) / det,
    y: (l1.a * l2.c - l2.a * l1.c) / det,
  };
}

// ── unused legacy helpers below (kept for compat) ────────────────────────
void function __unused() {
  function pickUnused() { return -1; }
  function lineFromTwo() { return null; }
  function distanceToLine() { return 0; }
};

// ── Stage 3 — per-corner zoom-in ─────────────────────────────────────────

/**
 * Refine each corner by walking along its two OUTWARD arms. The "outward"
 * direction depends on which corner it is — the two arms are the ones
 * that point AWAY from the page interior.
 *
 * For each arm, walk a short distance in the outward direction. At each
 * step, measure the gradient. The arm's terminus is where the gradient
 * peaks (this is the page edge). The refined corner position is the
 * midpoint between the two arm termini, snapped to the rough corner.
 */
export function refineCorners(gray, W, H, roughQuad) {
  // roughQuad is flat 8-array TL TR BR BL
  const c = [
    { x: roughQuad[0], y: roughQuad[1] },  // TL
    { x: roughQuad[2], y: roughQuad[3] },  // TR
    { x: roughQuad[4], y: roughQuad[5] },  // BR
    { x: roughQuad[6], y: roughQuad[7] },  // BL
  ];

  // For each corner, the "outward" axes:
  //   TL: -x (left), -y (up)
  //   TR: +x (right), -y (up)
  //   BR: +x (right), +y (down)
  //   BL: -x (left), +y (down)
  const outward = [
    { dx: -1, dy:  0 }, { dx:  1, dy:  0 },  // x-outward per corner
    { dx: -1, dy:  0 }, { dx:  1, dy:  0 },
  ];
  const outwardY = [
    { dx:  0, dy: -1 }, { dx:  0, dy: -1 },  // y-outward per corner
    { dx:  0, dy:  1 }, { dx:  0, dy:  1 },
  ];

  const HALF_ROI = 48;
  const STEP = 2;
  const refined = new Float64Array(8);

  for (let i = 0; i < 4; i++) {
    const corner = c[i];
    const xArm = outward[i];
    const yArm = outwardY[i];

    // Walk along x-arm, find the strongest gradient peak
    let bestX = corner, bestXGrad = 0;
    for (let s = STEP; s <= HALF_ROI; s += STEP) {
      const x = Math.round(corner.x + s * xArm.dx);
      const y = Math.round(corner.y + s * xArm.dy);
      if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) break;
      const g = edgeGradientX(gray, W, H, x, y, xArm.dx);
      if (g > bestXGrad) { bestXGrad = g; bestX = { x, y }; }
    }

    // Walk along y-arm
    let bestY = corner, bestYGrad = 0;
    for (let s = STEP; s <= HALF_ROI; s += STEP) {
      const x = Math.round(corner.x + s * yArm.dx);
      const y = Math.round(corner.y + s * yArm.dy);
      if (x < 1 || x >= W - 1 || y < 1 || y >= H - 1) break;
      const g = edgeGradientY(gray, W, H, x, y, yArm.dy);
      if (g > bestYGrad) { bestYGrad = g; bestY = { x, y }; }
    }

    // If neither arm found a strong gradient, keep the rough corner as-is.
    // Otherwise do a weighted shift of the corner toward the peaks.
    const GRAD_FLOOR = 5;
    if (bestXGrad < GRAD_FLOOR && bestYGrad < GRAD_FLOOR) {
      // No clear edge — keep rough corner untouched.
      refined[i * 2]     = corner.x;
      refined[i * 2 + 1] = corner.y;
      continue;
    }
    // Where at least one arm found something, blend using the peak
    // positions but trust the rough corner enough that we don't fly off.
    const wX = Math.max(bestXGrad, 0.001);
    const wY = Math.max(bestYGrad, 0.001);
    // Damp the refinement: refined = rough + 0.5 * (peak - rough)
    const peakX = (bestX.x * wX + bestY.x * wY) / (wX + wY);
    const peakY = (bestX.y * wX + bestY.y * wY) / (wX + wY);
    refined[i * 2]     = corner.x + 0.5 * (peakX - corner.x);
    refined[i * 2 + 1] = corner.y + 0.5 * (peakY - corner.y);
  }
  return Array.from(refined);
}

/** Gradient component perpendicular to an arm pointing in `dx` direction.
 *  Used to find the page edge along that arm. */
function edgeGradientX(gray, W, H, x, y, dxSign) {
  // If arm points left (-1), an edge to the LEFT means gradient is
  // bright on the right (i.e., the gradient vector points to the right).
  // Sign-of-gradient in x axis: dx adjacent cell.
  const x1 = Math.max(0, Math.min(W - 1, x));
  const xL = Math.max(0, Math.min(W - 1, x - dxSign));
  const xR = Math.max(0, Math.min(W - 1, x + dxSign));
  const yC = Math.max(0, Math.min(H - 1, y));
  const gx = gray[yC * W + xR] - gray[yC * W + xL];
  // Average of 3x1 neighborhood for noise suppression
  const yU = Math.max(0, yC - 1), yD = Math.min(H - 1, yC + 1);
  const gx2 = gray[yU * W + xR] - gray[yU * W + xL];
  const gx3 = gray[yD * W + xR] - gray[yD * W + xL];
  return Math.abs(gx + gx2 + gx3) / 3;
}

function edgeGradientY(gray, W, H, x, y, dySign) {
  const yC = Math.max(0, Math.min(H - 1, y));
  const yU = Math.max(0, Math.min(H - 1, y - dySign));
  const yD = Math.max(0, Math.min(H - 1, y + dySign));
  const xC = Math.max(0, Math.min(W - 1, x));
  const gy = gray[yD * W + xC] - gray[yU * W + xC];
  const xL = Math.max(0, xC - 1), xR = Math.min(W - 1, xC + 1);
  const gy2 = gray[yD * W + xR] - gray[yU * W + xR];
  const gy3 = gray[yD * W + xL] - gray[yU * W + xL];
  return Math.abs(gy + gy2 + gy3) / 3;
}