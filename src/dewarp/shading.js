// Shading / illumination correction.
//
// Direct port of autocapture/dewarp/shading.py.
//
// 1. gaussian_blur — separable, ksize must be odd >= 3, default sigma = k/6.
// 2. make_binary_mask — scanline rasterization of a convex quad.
// 3. shade_correct — flat-field correction by dividing by a heavily-blurred
//    estimate of the low-frequency illumination.

/**
 * Separable Gaussian blur on a 2D float64 image.
 * @param {Float64Array} img length H*W
 * @param {number} H
 * @param {number} W
 * @param {number} ksize must be odd >= 3
 * @returns {Float64Array} length H*W
 */
export function gaussianBlur(img, H, W, ksize) {
  if (ksize < 3 || ksize % 2 === 0) throw new Error(`ksize must be odd >= 3, got ${ksize}`);
  const sigma = ksize / 6.0;
  const k = gaussianKernel1D(ksize, sigma);
  const pad = Math.floor(ksize / 2);

  // Horizontal pass: pad columns with edge replication.
  const padded = new Float64Array(H * (W + 2 * pad));
  for (let i = 0; i < H; i++) {
    const srcRow = i * W;
    const dstRow = i * (W + 2 * pad);
    for (let j = 0; j < pad; j++) padded[dstRow + j] = img[srcRow];
    for (let j = 0; j < W; j++) padded[dstRow + pad + j] = img[srcRow + j];
    for (let j = 0; j < pad; j++) padded[dstRow + pad + W + j] = img[srcRow + W - 1];
  }

  const tmp = new Float64Array(H * W);
  for (let i = 0; i < H; i++) {
    const srcRow = i * (W + 2 * pad);
    const dstRow = i * W;
    for (let j = 0; j < W; j++) {
      let s = 0;
      for (let kk = 0; kk < ksize; kk++) s += k[kk] * padded[srcRow + j + kk];
      tmp[dstRow + j] = s;
    }
  }

  // Vertical pass
  const padded2 = new Float64Array((H + 2 * pad) * W);
  for (let i = 0; i < pad; i++) {
    const srcRow = 0;  // top edge
    const dstRow = i * W;
    for (let j = 0; j < W; j++) padded2[dstRow + j] = tmp[srcRow + j];
  }
  for (let i = 0; i < H; i++) {
    const srcRow = i * W;
    const dstRow = (pad + i) * W;
    for (let j = 0; j < W; j++) padded2[dstRow + j] = tmp[srcRow + j];
  }
  for (let i = 0; i < pad; i++) {
    const srcRow = (H - 1) * W;
    const dstRow = (pad + H + i) * W;
    for (let j = 0; j < W; j++) padded2[dstRow + j] = tmp[srcRow + j];
  }

  const out = new Float64Array(H * W);
  for (let i = 0; i < H; i++) {
    const srcBase = i * W;  // not the start row in padded2, that's the unpadded one
    // Actually the source row in padded2 for output row i is the (i)th padded row,
    // which is at index i (since pad + 0 ... pad + H - 1 contains tmp rows 0..H-1).
    // We just need to sum from padded2[(i + ki) * W + j].
    const dstRow = i * W;
    for (let j = 0; j < W; j++) {
      let s = 0;
      for (let kk = 0; kk < ksize; kk++) {
        s += k[kk] * padded2[(i + kk) * W + j];
      }
      out[dstRow + j] = s;
    }
    void srcBase;
  }
  return out;
}

/** Separable Gaussian blur on a 3-channel image (H*W*3). */
export function gaussianBlurRGB(img, H, W, ksize) {
  const out = new Float64Array(H * W * 3);
  for (let c = 0; c < 3; c++) {
    const ch = new Float64Array(H * W);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) ch[i * W + j] = img[(i * W + j) * 3 + c];
    const blurred = gaussianBlur(ch, H, W, ksize);
    for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) out[(i * W + j) * 3 + c] = blurred[i * W + j];
  }
  return out;
}

function gaussianKernel1D(size, sigma) {
  const half = (size - 1) / 2;
  const k = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - half;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  for (let i = 0; i < size; i++) k[i] /= sum;
  return k;
}

/**
 * Rasterize a (4, 2) convex quad into a binary mask.
 * Scanline fill: for each y, find x-intersections with the 4 edges.
 *
 * @param {[number, number]} shape [H, W]
 * @param {number[]} quad flat 8-array
 * @returns {Uint8Array} length H*W
 */
export function makeBinaryMask(shape, quad) {
  const [h, w] = shape;
  const mask = new Uint8Array(h * w);
  const qx = [quad[0], quad[2], quad[4], quad[6]];
  const qy = [quad[1], quad[3], quad[5], quad[7]];

  const yMin = Math.max(0, Math.min(...qy));
  const yMax = Math.min(h - 1, Math.max(...qy));
  if (yMax < yMin) return mask;

  for (let y = yMin; y <= yMax; y++) {
    const xs = [];
    for (let i = 0; i < 4; i++) {
      const p1y = qy[i];
      const p2y = qy[(i + 1) % 4];
      if ((p1y <= y && y < p2y) || (p2y <= y && y < p1y)) {
        if (p2y !== p1y) {
          const t = (y - p1y) / (p2y - p1y);
          const x = qx[i] + t * (qx[(i + 1) % 4] - qx[i]);
          xs.push(x);
        }
      }
    }
    if (xs.length >= 2) {
      xs.sort((a, b) => a - b);
      const x0 = Math.max(0, Math.floor(xs[0]));
      const x1 = Math.min(w - 1, Math.ceil(xs[xs.length - 1]));
      if (x1 >= x0) {
        for (let x = x0; x <= x1; x++) mask[y * w + x] = 1;
      }
    }
  }
  return mask;
}

/**
 * Flat-field correction inside the mask region.
 *
 * @param {Float64Array} img length H*W*3 or H*W
 * @param {Uint8Array} mask length H*W
 * @param {number} H
 * @param {number} W
 * @param {number} ksize odd >= 3
 * @returns {Uint8ClampedArray} same shape as input
 */
export function shadeCorrect(img, mask, H, W, ksize = 201) {
  const isRGB = img.length === H * W * 3;
  const C = isRGB ? 3 : 1;

  // Compute gray = mean across channels
  const gray = new Float64Array(H * W);
  if (isRGB) {
    for (let i = 0; i < H * W; i++) gray[i] = (img[i * 3] + img[i * 3 + 1] + img[i * 3 + 2]) / 3.0;
  } else {
    for (let i = 0; i < H * W; i++) gray[i] = img[i];
  }

  // gray_masked * mask, then blur to estimate illumination
  const grayMasked = new Float64Array(H * W);
  for (let i = 0; i < H * W; i++) grayMasked[i] = gray[i] * mask[i];
  let illum = gaussianBlur(grayMasked, H, W, ksize);

  // Outside the mask, illum = original gray (so output background unchanged)
  for (let i = 0; i < H * W; i++) {
    if (!mask[i]) illum[i] = gray[i];
    if (Math.abs(illum[i]) < 1e-3) illum[i] = 1e-3;
  }

  // Per-channel flat-field: (img / illum) * 128
  const out = new Uint8ClampedArray(H * W * C);
  if (isRGB) {
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < H * W; i++) {
        const v = (img[i * 3 + c] / illum[i]) * 128.0;
        out[i * 3 + c] = Math.max(0, Math.min(255, Math.round(v)));
      }
    }
  } else {
    for (let i = 0; i < H * W; i++) {
      const v = (img[i] / illum[i]) * 128.0;
      out[i] = Math.max(0, Math.min(255, Math.round(v)));
    }
  }
  return out;
}