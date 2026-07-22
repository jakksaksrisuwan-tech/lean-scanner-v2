// 3x3 homography + bilinear inverse-warp.
//
// Direct port of autocapture/dewarp/homography.py.
//
// Math: a homography H maps (x,y) -> (x',y') in projective coords.
// Solving for H from 4 correspondences uses the Direct Linear Transform
// (DLT) — 8 equations in 8 unknowns, solved by SVD on an 8x9 matrix.
// The right-singular vector with the smallest singular value is H.

import { svdRightVectors } from "../math/svd.js";
import { det2x2 } from "../math/ndarray.js";

/**
 * Compute the 3x3 homography mapping src_pts (4,2) -> dst_pts (4,2).
 * @returns {Float64Array} length-9 row-major matrix
 */
export function computeHomography(srcPts, dstPts) {
  if (srcPts.length !== 8 || dstPts.length !== 8) {
    throw new Error(`computeHomography: expected flat length-8 arrays, got ${srcPts.length} / ${dstPts.length}`);
  }
  const A = new Float64Array(8 * 9);
  for (let i = 0; i < 4; i++) {
    const sx = srcPts[i * 2], sy = srcPts[i * 2 + 1];
    const dx = dstPts[i * 2], dy = dstPts[i * 2 + 1];
    // row 2i
    A[(2 * i) * 9 + 0] = 0;      A[(2 * i) * 9 + 1] = 0;      A[(2 * i) * 9 + 2] = 0;
    A[(2 * i) * 9 + 3] = -sx;    A[(2 * i) * 9 + 4] = -sy;    A[(2 * i) * 9 + 5] = -1;
    A[(2 * i) * 9 + 6] = dy * sx; A[(2 * i) * 9 + 7] = dy * sy; A[(2 * i) * 9 + 8] = dy;
    // row 2i+1
    A[(2 * i + 1) * 9 + 0] = -sx; A[(2 * i + 1) * 9 + 1] = -sy; A[(2 * i + 1) * 9 + 2] = -1;
    A[(2 * i + 1) * 9 + 3] = 0;   A[(2 * i + 1) * 9 + 4] = 0;   A[(2 * i + 1) * 9 + 5] = 0;
    A[(2 * i + 1) * 9 + 6] = -dx * sx; A[(2 * i + 1) * 9 + 7] = -dx * sy; A[(2 * i + 1) * 9 + 8] = -dx;
  }

  // SVD: smallest singular value's right-singular vector is the last row of Vt
  const Vt = svdRightVectors(A, 8, 9);
  // Our svdRightVectors sorts rows of V ascending by sigma, so Vt[0]
  // is the right-singular vector for the smallest singular value
  // (matches the DLT requirement). np.linalg.svd sorts descending,
  // which is why the python implementation uses Vt[-1] — same row.
  const H = new Float64Array(9);
  for (let i = 0; i < 9; i++) H[i] = Vt[0 * 9 + i];  // first row

  // Normalize so H[2,2] = 1
  if (Math.abs(H[8]) > 1e-12) {
    for (let i = 0; i < 9; i++) H[i] /= H[8];
  }

  // Disambiguate the 4 projective solutions: try H, -H, and the two
  // single-axis row reflections. The DLT gives H up to scale and up to
  // a 4-element Klein-4 ambiguity {I, -I, diag(-1,1,1), diag(1,-1,1)}
  // acting on the LEFT (negates a row). Pick the one whose 2x2 linear
  // part has positive determinant and minimal centroid error.
  const candidates = [
    H,
    H.map((v) => -v),
    reflectRow(H, 0, -1),  // diag(-1, 1, 1) @ H: negate row 0
    reflectRow(H, 1, -1),  // diag(1, -1, 1) @ H: negate row 1
  ];

  const srcCx = (srcPts[0] + srcPts[2] + srcPts[4] + srcPts[6]) / 4;
  const srcCy = (srcPts[1] + srcPts[3] + srcPts[5] + srcPts[7]) / 4;
  const dstCx = (dstPts[0] + dstPts[2] + dstPts[4] + dstPts[6]) / 4;
  const dstCy = (dstPts[1] + dstPts[3] + dstPts[5] + dstPts[7]) / 4;

  let best = null, bestScore = -Infinity;
  for (const c of candidates) {
    // c @ [srcCx, srcCy, 1]
    const px = c[0] * srcCx + c[1] * srcCy + c[2];
    const py = c[3] * srcCx + c[4] * srcCy + c[5];
    const pw = c[6] * srcCx + c[7] * srcCy + c[8];
    if (Math.abs(pw) < 1e-12) continue;
    const cx = px / pw;
    const cy = py / pw;
    // det of top-left 2x2
    const detTopLeft = det2x2([c[0], c[1], c[3], c[4]]);
    const score = detTopLeft * 1e6 - ((cx - dstCx) ** 2 + (cy - dstCy) ** 2);
    if (score > bestScore) { bestScore = score; best = c; }
  }
  return best;
}

/** Multiply on the left by diag(...,1,...) but with sign flip on row k. */
function reflectRow(H, k, sign) {
  const out = new Float64Array(H);
  for (let j = 0; j < 3; j++) out[k * 3 + j] *= sign;
  return out;
}

/**
 * Inverse-warp src through H to produce an (outH, outW) image.
 *
 * For each output pixel (u, v) we apply H^-1 to get the source (x, y),
 * then sample with bilinear interpolation. Pixels outside src fill with 0.
 *
 * @param {Uint8ClampedArray|Uint8Array} src length H_src*W_src*C
 * @param {number} H_src
 * @param {number} W_src
 * @param {number} C channels (1 or 3)
 * @param {Float64Array} H length-9, src->dst
 * @param {number} outH
 * @param {number} outW
 * @returns {Uint8ClampedArray} length outH*outW*C
 */
export function warpBilinear(src, H_src, W_src, C, H, outH, outW) {
  // Inverse H
  const Hi = invert3x3(H);

  const out = new Uint8ClampedArray(outH * outW * C);
  const srcH = H_src, srcW = W_src;

  for (let v = 0; v < outH; v++) {
    for (let u = 0; u < outW; u++) {
      // Hi @ [u, v, 1]
      const hx = Hi[0] * u + Hi[1] * v + Hi[2];
      const hy = Hi[3] * u + Hi[4] * v + Hi[5];
      const hw = Hi[6] * u + Hi[7] * v + Hi[8];
      const denom = Math.abs(hw) < 1e-12 ? 1e-12 : hw;
      const x = hx / denom;
      const y = hy / denom;

      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      if (x0 < 0 || x0 >= srcW || y0 < 0 || y0 >= srcH) continue;  // leaves 0
      const x1 = Math.min(srcW - 1, x0 + 1);
      const y1 = Math.min(srcH - 1, y0 + 1);
      const wx = x - x0;
      const wy = y - y0;
      const base = (v * outW + u) * C;
      for (let c = 0; c < C; c++) {
        const ia = src[(y0 * srcW + x0) * C + c];
        const ib = src[(y0 * srcW + x1) * C + c];
        const ic = src[(y1 * srcW + x0) * C + c];
        const id = src[(y1 * srcW + x1) * C + c];
        const top = ia * (1 - wx) + ib * wx;
        const bot = ic * (1 - wx) + id * wx;
        out[base + c] = Math.round(top * (1 - wy) + bot * wy);
      }
    }
  }
  return out;
}

function invert3x3(M) {
  const a = M[0], b = M[1], c = M[2];
  const d = M[3], e = M[4], f = M[5];
  const g = M[6], h = M[7], i = M[8];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const Hc = -(a * f - c * d);
  const Ic = a * e - b * d;
  const det = a * A + b * B + c * C;
  const invDet = 1.0 / det;
  return new Float64Array([
    A * invDet, D * invDet, G * invDet,
    B * invDet, E * invDet, Hc * invDet,
    C * invDet, F * invDet, Ic * invDet,
  ]);
}

/**
 * Warp src so the 4-corner quad becomes a rectangle.
 *
 * @param {Uint8ClampedArray} src
 * @param {number} srcH
 * @param {number} srcW
 * @param {number} C
 * @param {number[]} quad flat 8-array, TL TR BR BL
 * @param {string} targetAspect "A4" | "US_LETTER" | "MAX_EDGE" | "AUTO"
 * @returns {{ data: Uint8ClampedArray, width: number, height: number }}
 */
export function warpQuad(src, srcH, srcW, C, quad, targetAspect = "A4") {
  // Edge lengths
  const top    = dist(quad[0], quad[1], quad[2], quad[3]);
  const bot    = dist(quad[6], quad[7], quad[4], quad[5]);
  const right  = dist(quad[2], quad[3], quad[4], quad[5]);
  const left   = dist(quad[0], quad[1], quad[6], quad[7]);
  let width = Math.max(top, bot);
  let height = Math.max(left, right);

  if (targetAspect === "A4") {
    const aspect = Math.sqrt(2.0);
    if (height > width / aspect) height = Math.floor(width / aspect);
    else width = Math.floor(height * aspect);
  } else if (targetAspect === "US_LETTER") {
    const aspect = 11.0 / 8.5;
    if (height > width / aspect) height = Math.floor(width / aspect);
    else width = Math.floor(height * aspect);
  } else if (targetAspect === "MAX_EDGE") {
    const side = Math.floor(Math.max(width, height));
    width = side; height = side;
  }
  width = Math.max(1, Math.floor(width));
  height = Math.max(1, Math.floor(height));

  const dst = [0, 0, width - 1, 0, width - 1, height - 1, 0, height - 1];
  const H = computeHomography(quad, dst);
  const data = warpBilinear(src, srcH, srcW, C, H, height, width);
  return { data, width, height };
}

function dist(x1, y1, x2, y2) {
  const dx = x1 - x2, dy = y1 - y2;
  return Math.sqrt(dx * dx + dy * dy);
}