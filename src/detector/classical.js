// Document-quad detection.
//
// Direct port of autocapture/detector/classical.py. Since this is JS
// (no opencv by default), only the pure-numpy fallback path is
// implemented. It is intentionally simple: sobel edges -> bbox of strong
// edges -> 4-vertex quad -> refine.
//
// On the simulator's high-contrast synthetic document this works. On
// real photos it should be considered a smoke test — replace with a
// YOLOv8n-doc or vnDocumentScanner for production.

import { orderCorners, quadArea, isConvex, quadAspect } from "./quad.js";

/**
 * Sobel edge magnitude for a 2D float64 image.
 * Returns a Float64Array of length H*W in [0, ~1020].
 */
export function sobelMag(gray, H, W) {
  // Edge-padded input
  const padded = new Float64Array((H + 2) * (W + 2));
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      padded[(i + 1) * (W + 2) + (j + 1)] = gray[i * W + j];
    }
  }
  // Fill edges
  for (let i = 0; i < H; i++) {
    padded[(i + 1) * (W + 2) + 0] = gray[i * W + 0];
    padded[(i + 1) * (W + 2) + (W + 1)] = gray[i * W + (W - 1)];
  }
  for (let j = 0; j < W; j++) {
    padded[0 * (W + 2) + (j + 1)] = gray[0 * W + j];
    padded[(H + 1) * (W + 2) + (j + 1)] = gray[(H - 1) * W + j];
  }
  padded[0] = gray[0];
  padded[W + 1] = gray[W - 1];
  padded[(H + 1) * (W + 2)] = gray[(H - 1) * W];
  padded[(H + 2) * (W + 2) - 1] = gray[(H - 1) * W + (W - 1)];

  const out = new Float64Array(H * W);
  // Apply 3x3 Sobel kernels: gx on x, gy on y
  const KX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const KY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      let gx = 0, gy = 0;
      for (let ki = 0; ki < 3; ki++) {
        for (let kj = 0; kj < 3; kj++) {
          const v = padded[(i + ki) * (W + 2) + (j + kj)];
          gx += KX[ki * 3 + kj] * v;
          gy += KY[ki * 3 + kj] * v;
        }
      }
      out[i * W + j] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return out;
}

/**
 * Pure-JS quad detector (fallback path).
 * Returns ordered (TL,TR,BR,BL) flat array of length 8, or null.
 *
 * @param {Float64Array} gray 2D grayscale as flat array, length H*W
 * @param {number} H
 * @param {number} W
 * @param {object} cfg detector config
 * @returns {number[] | null}
 */
export function detectNumpy(gray, H, W, cfg) {
  // 1. Sobel edges
  const edges = sobelMag(gray, H, W);

  // 2. Threshold — top 40% of gradient magnitude
  let maxE = 0;
  for (let i = 0; i < edges.length; i++) if (edges[i] > maxE) maxE = edges[i];
  if (maxE < 1) return null;
  const thr = maxE * 0.4;

  // Collect strong-edge pixel coordinates
  let nStrong = 0;
  const ys = new Float64Array(edges.length);
  const xs = new Float64Array(edges.length);
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      if (edges[i * W + j] > thr) {
        ys[nStrong] = i;
        xs[nStrong] = j;
        nStrong++;
      }
    }
  }
  if (nStrong < 100) return null;

  // 3. Bounding box of strong edges (percentile-trimmed to ignore stray text)
  const sortedY = Array.from(ys.subarray(0, nStrong)).sort((a, b) => a - b);
  const sortedX = Array.from(xs.subarray(0, nStrong)).sort((a, b) => a - b);
  const pct = (arr, p) => arr[Math.min(arr.length - 1, Math.floor(arr.length * p))];
  const y0 = Math.round(pct(sortedY, 0.02));
  const y1 = Math.round(pct(sortedY, 0.98));
  const x0 = Math.round(pct(sortedX, 0.02));
  const x1 = Math.round(pct(sortedX, 0.98));

  if (x1 - x0 < 0.1 * W || y1 - y0 < 0.1 * H) return null;

  // 4. Refine rough bbox to actual page edges
  const refined = refineToEdge(edges, H, W, [
    x0, y0, x1, y0, x1, y1, x0, y1,
  ]);

  return orderCorners(refined);
}

/**
 * Snap a rough bbox quad to the strongest gradient on each side.
 * For each side, walk along the inward normal and find the max-gradient
 * position; shift each endpoint of that side by half the displacement.
 */
function refineToEdge(edges, h, w, rough) {
  const refined = new Float64Array(8);
  for (let i = 0; i < 4; i++) {
    const p1x = rough[i * 2],     p1y = rough[i * 2 + 1];
    const p2x = rough[((i + 1) % 4) * 2], p2y = rough[((i + 1) % 4) * 2 + 1];
    const p3x = rough[((i + 2) % 4) * 2], p3y = rough[((i + 2) % 4) * 2 + 1];
    const p4x = rough[((i + 3) % 4) * 2], p4y = rough[((i + 3) % 4) * 2 + 1];

    const midX = (p1x + p2x) / 2;
    const midY = (p1y + p2y) / 2;
    const otherMidX = (p3x + p4x) / 2;
    const otherMidY = (p3y + p4y) / 2;
    let nx = otherMidX - midX;
    let ny = otherMidY - midY;
    const nLen = Math.sqrt(nx * nx + ny * ny);
    if (nLen < 1e-6) {
      refined[i * 2] = p1x; refined[i * 2 + 1] = p1y;
      continue;
    }
    nx /= nLen; ny /= nLen;

    let bestT = 0, bestVal = -1;
    for (let t = -10; t <= 10; t += 2) {
      const px = Math.max(0, Math.min(w - 1, Math.round(midX + t * nx)));
      const py = Math.max(0, Math.min(h - 1, Math.round(midY + t * ny)));
      const v = edges[py * w + px];
      if (v > bestVal) { bestVal = v; bestT = t; }
    }
    const sx = bestT * nx * 0.5;
    const sy = bestT * ny * 0.5;
    refined[i * 2]     = p1x + sx;
    refined[i * 2 + 1] = p1y + sy;
    refined[((i + 1) % 4) * 2]     = p2x + sx;
    refined[((i + 1) % 4) * 2 + 1] = p2y + sy;
  }
  return Array.from(refined);
}

/**
 * Run detection on an RGBA/BGRA ImageData.
 * Returns ordered quad (flat 8) or null.
 *
 * @param {ImageData} imageData
 * @param {object} cfg
 */
export function detect(imageData, cfg) {
  const W = imageData.width;
  const H = imageData.height;
  const data = imageData.data;

  // Down-scale to long-edge first, to match the python pipeline's behavior
  let scale = cfg.longEdge / Math.max(H, W);
  let smallW = W, smallH = H;
  if (scale < 1.0) {
    smallW = Math.max(1, Math.round(W * scale));
    smallH = Math.max(1, Math.round(H * scale));
  }

  // Extract grayscale, optionally downscaled
  const gray = new Float64Array(smallH * smallW);
  if (scale >= 1.0) {
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const p = (i * W + j) * 4;
        gray[i * W + j] = (data[p] + data[p + 1] + data[p + 2]) / 3.0;
      }
    }
  } else {
    // Bilinear downsample — matches cv2.resize default (bilinear)
    bilinearDownsample(data, W, H, gray, smallW, smallH);
  }

  // Detect on small image
  let quad = detectNumpy(gray, smallH, smallW, cfg);
  if (quad == null) return null;

  // Scale quad back up
  if (scale < 1.0) {
    for (let i = 0; i < 8; i++) quad[i] /= scale;
  }

  // Sanity check
  const area = quadArea(quad);
  const aspect = quadAspect(quad);
  const frameArea = W * H;
  const arRatio = area / frameArea;
  if (arRatio < cfg.minQuadAreaRatio) return null;
  if (arRatio > cfg.maxQuadAreaRatio) return null;
  if (!isConvex(quad)) return null;
  if (aspect < cfg.minAspect || aspect > cfg.maxAspect) return null;
  return quad;
}

function bilinearDownsample(src, srcW, srcH, dst, dstW, dstH) {
  // Map dst pixel (x, y) -> src via (x + 0.5) * srcW/dstW - 0.5
  const sx = srcW / dstW;
  const sy = srcH / dstH;
  for (let y = 0; y < dstH; y++) {
    const srcY = (y + 0.5) * sy - 0.5;
    const y0 = Math.max(0, Math.floor(srcY));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = Math.max(0, Math.min(1, srcY - y0));
    for (let x = 0; x < dstW; x++) {
      const srcX = (x + 0.5) * sx - 0.5;
      const x0 = Math.max(0, Math.floor(srcX));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const wx = Math.max(0, Math.min(1, srcX - x0));
      const p00 = (y0 * srcW + x0) * 4;
      const p01 = (y0 * srcW + x1) * 4;
      const p10 = (y1 * srcW + x0) * 4;
      const p11 = (y1 * srcW + x1) * 4;
      const top = (1 - wx) * (src[p00] + src[p00 + 1] + src[p00 + 2]) / 3
                + wx * (src[p01] + src[p01 + 1] + src[p01 + 2]) / 3;
      const bot = (1 - wx) * (src[p10] + src[p10 + 1] + src[p10 + 2]) / 3
                + wx * (src[p11] + src[p11 + 1] + src[p11 + 2]) / 3;
      dst[y * dstW + x] = (1 - wy) * top + wy * bot;
    }
  }
}