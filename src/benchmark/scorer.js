// Scoring a detected quadrilateral against the ground-truth one.
//
// We need a fair rotation-tolerant IoU. Axis-aligned bbox IoU (which I
// already have in quad.js) works for axis-aligned docs but wildly
// under-reports for rotated docs. The proper metric is polygon
// intersection / polygon union, with the Sutherland-Hodgman algorithm
// for clipping one convex polygon against another. Each quad is convex
// with 4 sides, so this is straightforward.

/**
 * @param {number[]} a flat 8-array TL TR BR BL
 * @param {number[]} b flat 8-array TL TR BR BL
 * @returns {number} IoU in [0, 1]
 */
export function polygonIoU(a, b) {
  const areaA = shoelaceArea(a);
  const areaB = shoelaceArea(b);
  if (areaA < 1 || areaB < 1) return 0;
  const interArea = polygonIntersectionArea(a, b);
  const unionArea = areaA + areaB - interArea;
  if (unionArea < 1) return 0;
  return interArea / unionArea;
}

function shoelaceArea(quad) {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    s += quad[i * 2] * quad[j * 2 + 1] - quad[j * 2] * quad[i * 2 + 1];
  }
  return Math.abs(s) / 2;
}

/** Build a polygon as an array of [x, y] points from a flat 8-array. */
function toPoly(quad) {
  return [
    [quad[0], quad[1]],
    [quad[2], quad[3]],
    [quad[4], quad[5]],
    [quad[6], quad[7]],
  ];
}

/** Polygon intersection area using Sutherland-Hodgman.
 *  Subject is clipped against clip. Both must be convex CCW (or both CW).
 *  Returns the intersection area (0 if none). */
function polygonIntersectionArea(subject, clip) {
  const subj = toPoly(subject);
  const clp = toPoly(clip);
  // Make sure both CCW
  let outputList = ensureCCW(subj);
  const clipPoly = ensureCCW(clp);
  for (let i = 0; i < clipPoly.length; i++) {
    const inputList = outputList;
    outputList = [];
    const a = clipPoly[i];
    const b = clipPoly[(i + 1) % clipPoly.length];
    if (inputList.length === 0) break;
    let s = inputList[inputList.length - 1];
    for (let j = 0; j < inputList.length; j++) {
      const e = inputList[j];
      if (inside(e, a, b)) {
        if (!inside(s, a, b)) {
          outputList.push(intersect(s, e, a, b));
        }
        outputList.push(e);
      } else if (inside(s, a, b)) {
        outputList.push(intersect(s, e, a, b));
      }
      s = e;
    }
  }
  if (outputList.length < 3) return 0;
  return polyArea(outputList);
}

function ensureCCW(poly) {
  if (polyArea(poly) > 0) return poly;
  return poly.slice().reverse();
}

function polyArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const j = (i + 1) % poly.length;
    s += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1];
  }
  return s / 2;
}

function inside(p, a, b) {
  // CCW polygon: inside if p is to the left of edge a -> b
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0;
}

function intersect(s, e, a, b) {
  const x1 = s[0], y1 = s[1], x2 = e[0], y2 = e[1];
  const x3 = a[0], y3 = a[1], x4 = b[0], y4 = b[1];
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 1e-12) {
    return [x1, y1];
  }
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
  return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
}

/**
 * Corner-distance metric: mean Euclidean distance between corresponding
 * corners of detected vs ground-truth quad. Lower is better.
 *
 * Requires both quads ordered TL TR BR BL.
 */
export function meanCornerError(detected, gt) {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    const dx = detected[i * 2] - gt[i * 2];
    const dy = detected[i * 2 + 1] - gt[i * 2 + 1];
    s += Math.sqrt(dx * dx + dy * dy);
  }
  return s / 4;
}

/**
 * Center offset: distance between detected centroid and GT centroid.
 */
export function centerError(detected, gt) {
  let dcx = 0, dcy = 0, gcx = 0, gcy = 0;
  for (let i = 0; i < 4; i++) {
    dcx += detected[i * 2]; dcy += detected[i * 2 + 1];
    gcx += gt[i * 2];     gcy += gt[i * 2 + 1];
  }
  dcx /= 4; dcy /= 4; gcx /= 4; gcy /= 4;
  return Math.sqrt((dcx - gcx) ** 2 + (dcy - gcy) ** 2);
}
