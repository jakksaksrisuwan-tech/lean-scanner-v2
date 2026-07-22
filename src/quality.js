// Per-frame quality scoring.
//
// Direct port of autocapture/quality.py.
//
// Score is in [0, 1] — higher is better. Components:
//   area, straightness, sharpness (Laplacian variance),
//   exposure (mean luma in mask), blur (Sobel magnitude mean).

import { quadArea } from "./detector/quad.js";
import { makeBinaryMask } from "./dewarp/shading.js";

function laplacianVariance(gray, H, W) {
  // 4-neighbor Laplacian: -4*p + up + down + left + right
  let sum = 0, sumSq = 0;
  for (let i = 1; i < H - 1; i++) {
    for (let j = 1; j < W - 1; j++) {
      const c = gray[i * W + j];
      const lap = -4 * c
        + gray[(i - 1) * W + j]
        + gray[(i + 1) * W + j]
        + gray[i * W + (j - 1)]
        + gray[i * W + (j + 1)];
      sum += lap;
      sumSq += lap * lap;
    }
  }
  const n = (H - 2) * (W - 2);
  const mean = sum / n;
  return sumSq / n - mean * mean;
}

function sobelMagMean(gray, H, W) {
  // Mean Sobel magnitude using 3x3 kernels. Same approach as
  // detector/classical.js sobelMag but we only need the mean, not the
  // full image, so we don't allocate the output array.
  const KX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const KY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  let sum = 0, count = 0;
  for (let i = 1; i < H - 1; i++) {
    for (let j = 1; j < W - 1; j++) {
      let gx = 0, gy = 0;
      for (let ki = -1; ki <= 1; ki++) {
        for (let kj = -1; kj <= 1; kj++) {
          const v = gray[(i + ki) * W + (j + kj)];
          gx += KX[(ki + 1) * 3 + (kj + 1)] * v;
          gy += KY[(ki + 1) * 3 + (kj + 1)] * v;
        }
      }
      sum += Math.sqrt(gx * gx + gy * gy);
      count++;
    }
  }
  return count > 0 ? sum / count : 0;
}

/**
 * @param {number[]} quad flat 8-array, ordered TL TR BR BL
 * @param {Uint8ClampedArray|Uint8Array} frameBGR length H*W*3 or H*W*4 (RGBA)
 * @param {number} H
 * @param {number} W
 * @param {object} cfg quality config
 * @returns {number} score in [0, 1]
 */
export function score(quad, frameBGR, H, W, cfg) {
  // Convert to grayscale if needed
  const isRGBA = frameBGR.length === H * W * 4;
  const C = isRGBA ? 4 : 3;
  const gray = new Float64Array(H * W);
  for (let i = 0; i < H * W; i++) {
    let s;
    if (isRGBA) {
      s = (frameBGR[i * 4] + frameBGR[i * 4 + 1] + frameBGR[i * 4 + 2]) / 3.0;
    } else {
      s = (frameBGR[i * 3] + frameBGR[i * 3 + 1] + frameBGR[i * 3 + 2]) / 3.0;
    }
    gray[i] = s;
  }
  void C;

  // 1. area
  const frameArea = H * W;
  const area = quadArea(quad);
  const areaRatio = area / frameArea;
  if (areaRatio < cfg.minAreaRatio) return 0.0;
  const areaScore = Math.min(areaRatio / cfg.targetAreaRatio, 1.0);

  // 2. straightness
  const edges = [
    dist(quad[0], quad[1], quad[2], quad[3]),
    dist(quad[2], quad[3], quad[4], quad[5]),
    dist(quad[4], quad[5], quad[6], quad[7]),
    dist(quad[6], quad[7], quad[0], quad[1]),
  ];
  const eMin = Math.min(...edges), eMax = Math.max(...edges);
  const straightness = eMax > 0 ? eMin / eMax : 0.0;

  // 3. sharpness
  const lapVar = laplacianVariance(gray, H, W);
  const sharpness = Math.min(lapVar / cfg.sharpnessFull, 1.0);

  // 4. exposure (mean luma in the quad region)
  const mask = makeBinaryMask([H, W], quad);
  let maskSum = 0, lumaSum = 0;
  for (let i = 0; i < H * W; i++) {
    if (mask[i]) { lumaSum += gray[i]; maskSum++; }
  }
  const lumaMean = maskSum > 0 ? lumaSum / maskSum : gray.reduce((s, v) => s + v, 0) / gray.length;

  let exposure = 1.0;
  if (cfg.exposureLow < lumaMean && lumaMean < cfg.exposureHigh) {
    exposure = 1.0;
  } else if (lumaMean <= 0 || lumaMean >= 255) {
    exposure = 0.0;
  } else if (lumaMean < cfg.exposureLow) {
    exposure = Math.max(0, lumaMean / cfg.exposureLow);
  } else {
    exposure = Math.max(0, (255 - lumaMean) / (255 - cfg.exposureHigh));
  }

  // 5. blur
  const blurMean = sobelMagMean(gray, H, W);
  const blur = Math.min(blurMean / cfg.blurFull, 1.0);

  const total = cfg.weightArea * areaScore
              + cfg.weightStraightness * straightness
              + cfg.weightSharpness * sharpness
              + cfg.weightExposure * exposure
              + cfg.weightBlur * blur;
  return Math.max(0, Math.min(1, total));
}

function dist(x1, y1, x2, y2) {
  const dx = x1 - x2, dy = y1 - y2;
  return Math.sqrt(dx * dx + dy * dy);
}