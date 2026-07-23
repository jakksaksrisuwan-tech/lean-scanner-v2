// Quad utilities — ordering, area, IoU, drift.
//
// Direct port of autocapture/detector/quad.py. The python version is a
// pure-numpy module; here we use plain typed arrays since the data is
// always (4, 2) flat floats.

/** Signed (absolute) area of a 4-corner polygon via the shoelace formula. */
export function quadArea(quad) {
  // quad: array-like of length 8 (4 corners × xy)
  const x0 = quad[0], y0 = quad[1];
  const x1 = quad[2], y1 = quad[3];
  const x2 = quad[4], y2 = quad[5];
  const x3 = quad[6], y3 = quad[7];
  return 0.5 * Math.abs(
    x0 * y1 - x1 * y0 +
    x1 * y2 - x2 * y1 +
    x2 * y3 - x3 * y2 +
    x3 * y0 - x0 * y3
  );
}

/**
 * Reorder 4 corners into TL, TR, BR, BL.
 *
 * Trick: TL has min(x+y), BR has max(x+y), TR has min(x-y), BL has
 * max(x-y). Robust to almost-arbitrary input order.
 *
 * @param {number[]} q length-8 flat array
 * @returns {number[]} length-8 reordered flat array
 */
export function orderCorners(q) {
  const c = [
    [q[0], q[1]],
    [q[2], q[3]],
    [q[4], q[5]],
    [q[6], q[7]],
  ];
  const sums = c.map((p) => p[0] + p[1]);
  // Python uses np.diff which yields y - x (subtracts column 0 from column 1).
  const diffs = c.map((p) => p[1] - p[0]);

  const argmin = (arr) => arr.indexOf(Math.min(...arr));
  const argmax = (arr) => arr.indexOf(Math.max(...arr));

  const tl = c[argmin(sums)];
  const br = c[argmax(sums)];
  const tr = c[argmin(diffs)];
  const bl = c[argmax(diffs)];

  return [
    tl[0], tl[1],
    tr[0], tr[1],
    br[0], br[1],
    bl[0], bl[1],
  ];
}

/** Quick convexity test: cross products at all 4 vertices must share sign. */
export function isConvex(quad) {
  const q = [
    [quad[0], quad[1]],
    [quad[2], quad[3]],
    [quad[4], quad[5]],
    [quad[6], quad[7]],
  ];
  const signs = [];
  for (let i = 0; i < 4; i++) {
    const a = q[i];
    const b = q[(i + 1) % 4];
    const c = q[(i + 2) % 4];
    const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
    signs.push(cross);
  }
  let allPos = true, allNeg = true;
  for (const s of signs) {
    if (s <= 0) allPos = false;
    if (s >= 0) allNeg = false;
  }
  return allPos || allNeg;
}

/** Bounding-box aspect ratio (w / h). 0 if h == 0. */
export function quadAspect(quad) {
  const xs = [quad[0], quad[2], quad[4], quad[6]];
  const ys = [quad[1], quad[3], quad[5], quad[7]];
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const w = xMax - xMin, h = yMax - yMin;
  return h > 0 ? w / h : 0.0;
}

/** Approximate IoU of two quads using their bounding boxes. */
export function quadIou(a, b) {
  const aXs = [a[0], a[2], a[4], a[6]];
  const aYs = [a[1], a[3], a[5], a[7]];
  const bXs = [b[0], b[2], b[4], b[6]];
  const bYs = [b[1], b[3], b[5], b[7]];
  const ax0 = Math.min(...aXs), ax1 = Math.max(...aXs);
  const ay0 = Math.min(...aYs), ay1 = Math.max(...aYs);
  const bx0 = Math.min(...bXs), bx1 = Math.max(...bXs);
  const by0 = Math.min(...bYs), by1 = Math.max(...bYs);
  const ix0 = Math.max(ax0, bx0);
  const iy0 = Math.max(ay0, by0);
  const ix1 = Math.min(ax1, bx1);
  const iy1 = Math.min(ay1, by1);
  const iw = Math.max(0, ix1 - ix0);
  const ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  const union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
  return union > 0 ? inter / union : 0.0;
}

/** Mean Euclidean distance between corresponding corners (length-8 arrays). */
export function meanCornerDrift(a, b) {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const dx = a[i * 2] - b[i * 2];
    const dy = a[i * 2 + 1] - b[i * 2 + 1];
    s += Math.sqrt(dx * dx + dy * dy);
  }
  return s / 4;
}