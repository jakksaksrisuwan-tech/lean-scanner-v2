// Contrast enhancement and adaptive thresholding.
//
// Direct port of autocapture/dewarp/contrast.py.

import { gaussianBlur } from "./shading.js";

/** Global histogram equalization on a 2D uint8 image. Returns Uint8Array. */
export function histogramEqualization(gray) {
  if (gray instanceof Uint8ClampedArray || gray instanceof Uint8Array) {
    // OK
  } else {
    throw new Error("histogramEqualization: expected Uint8Array");
  }
  const hist = new Float64Array(256);
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++;
  const cdf = new Float64Array(256);
  cdf[0] = hist[0];
  for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];

  // First non-zero CDF value
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) if (cdf[i] > 0) { cdfMin = cdf[i]; break; }

  const lut = new Uint8Array(256);
  const denom = Math.max(cdf[255] - cdfMin, 1);
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.floor((cdf[i] - cdfMin) * 255.0 / denom);
  }
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) out[i] = lut[gray[i]];
  return out;
}

/**
 * Tile-based CLAHE on a 2D uint8 image.
 * Vectorized per-tile; bilinear interpolation between tile centers.
 *
 * @param {Uint8Array} gray length H*W
 * @param {number} H
 * @param {number} W
 * @param {number} clipLimit multiplier on average bin count
 * @param {number} grid tiles per side
 * @returns {Uint8Array} length H*W
 */
export function claheSimple(gray, H, W, clipLimit = 2.0, grid = 8) {
  const tileH = Math.floor(H / grid);
  const tileW = Math.floor(W / grid);
  const nbins = 256;
  const clipVal = Math.max(1, Math.floor(clipLimit * (tileH * tileW / nbins)));

  // Per-tile LUTs
  const lut = new Uint8Array(grid * grid * 256);
  for (let ty = 0; ty < grid; ty++) {
    for (let tx = 0; tx < grid; tx++) {
      const y0 = ty * tileH, x0 = tx * tileW;
      const hist = new Float64Array(nbins);
      for (let y = y0; y < y0 + tileH; y++) {
        const base = y * W + x0;
        for (let x = 0; x < tileW; x++) hist[gray[base + x]]++;
      }
      // Clip
      let excess = 0;
      for (let i = 0; i < nbins; i++) {
        if (hist[i] > clipVal) { excess += hist[i] - clipVal; hist[i] = clipVal; }
      }
      const inc = Math.floor(excess / nbins);
      for (let i = 0; i < nbins; i++) hist[i] += inc;
      // CDF
      const cdf = new Float64Array(nbins);
      cdf[0] = hist[0];
      for (let i = 1; i < nbins; i++) cdf[i] = cdf[i - 1] + hist[i];
      const denom = Math.max(cdf[nbins - 1] - cdf[0], 1);
      const baseLut = (ty * grid + tx) * 256;
      for (let i = 0; i < nbins; i++) {
        lut[baseLut + i] = Math.floor((cdf[i] - cdf[0]) * 255 / denom);
      }
    }
  }

  // Vectorized bilinear interp between tiles
  const out = new Uint8Array(H * W);
  for (let y = 0; y < H; y++) {
    const gy = (y + 0.5) / tileH - 0.5;
    const gy0 = Math.max(0, Math.min(grid - 1, Math.floor(gy)));
    const gy1 = Math.max(0, Math.min(grid - 1, gy0 + 1));
    const wy = Math.max(0, Math.min(1, gy - gy0));
    for (let x = 0; x < W; x++) {
      const gx = (x + 0.5) / tileW - 0.5;
      const gx0 = Math.max(0, Math.min(grid - 1, Math.floor(gx)));
      const gx1 = Math.max(0, Math.min(grid - 1, gx0 + 1));
      const wx = Math.max(0, Math.min(1, gx - gx0));
      const v = gray[y * W + x];
      const m00 = lut[(gy0 * grid + gx0) * 256 + v];
      const m01 = lut[(gy0 * grid + gx1) * 256 + v];
      const m10 = lut[(gy1 * grid + gx0) * 256 + v];
      const m11 = lut[(gy1 * grid + gx1) * 256 + v];
      const top = m00 * (1 - wx) + m01 * wx;
      const bot = m10 * (1 - wx) + m11 * wx;
      out[y * W + x] = Math.round(top * (1 - wy) + bot * wy);
    }
  }
  return out;
}

/** Gaussian adaptive threshold (binarization for B&W scan look). */
export function adaptiveThreshold(gray, H, W, block = 11, c = 2) {
  if (block < 3 || block % 2 === 0) throw new Error(`block must be odd >= 3, got ${block}`);
  const f = new Float64Array(gray.length);
  for (let i = 0; i < gray.length; i++) f[i] = gray[i];
  const localMean = gaussianBlur(f, H, W, block);
  const out = new Uint8Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    out[i] = f[i] > (localMean[i] - c) ? 255 : 0;
  }
  return out;
}