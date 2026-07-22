// Candidate model adapters for the segment-model benchmark.
//
// Each adapter exposes a uniform interface:
//
//   const c = await load(path, opts)
//   const result = await c.detect(frame)  // { quad, score, latencyMs, ok }
//   await c.dispose()
//
// Frame shape: { rGBA: Uint8ClampedArray, width, height, gtQuad, ... }.
//
// "Score" is the model's own confidence. For mask-based candidates the
// score is derived from mean mask probability inside the inferred
// quadrilateral. For corner-based candidates the score is the mean
// peak value across the four corner heatmaps.

import { readFile } from "node:fs/promises";
import { detectRadial } from "../../detector/radial.js";

let _ort;
async function ort() {
  if (_ort) return _ort;
  _ort = await import("onnxruntime-node");
  return _ort;
}

// --- Radial: existing JS detector (no model) ---------------------------
export async function loadRadial(_path) {
  return {
    name: "radial-js",
    type: "corners",
    warmup() {},
    async detect(frame) {
      const t0 = performance.now();
      const id = { data: frame.rGBA, width: frame.width, height: frame.height };
      const r = detectRadial(id, {
        longEdge: 640,
        minQuadAreaRatio: 0.05,
        maxQuadAreaRatio: 0.98,
        minAspect: 0.2,
        maxAspect: 5.0,
      });
      const dt = performance.now() - t0;
      return r
        ? { quad: r.quad, score: r.confidence ?? 0.5, latencyMs: dt, ok: true }
        : { quad: null, score: 0, latencyMs: dt, ok: false };
    },
    async dispose() {},
  };
}

// --- Shared letterbox heatmap decoder --------------------------------
async function loadSession(path) {
  const o = await ort();
  const buf = await readFile(path);
  return await o.InferenceSession.create(buf, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });
}

function letterboxRgba(rgba, W, H, SIZE) {
  const out = new Float32Array(1 * 3 * SIZE * SIZE);
  const scale = Math.min(SIZE / W, SIZE / H);
  const nw = Math.max(1, Math.round(W * scale));
  const nh = Math.max(1, Math.round(H * scale));
  const ox = (SIZE - nw) >> 1;
  const oy = (SIZE - nh) >> 1;
  for (let y = 0; y < nh; y++) {
    for (let xx = 0; xx < nw; xx++) {
      const sx = Math.min(W - 1, Math.floor(xx / scale));
      const sy = Math.min(H - 1, Math.floor(y / scale));
      const di = (sy * W + sx) * 4;
      const oi = (oy + y) * SIZE + (ox + xx);
      out[0 * SIZE * SIZE + oi] = rgba[di]     / 255;
      out[1 * SIZE * SIZE + oi] = rgba[di + 1] / 255;
      out[2 * SIZE * SIZE + oi] = rgba[di + 2] / 255;
    }
  }
  return { tensor: out, scale, nw, nh, ox, oy };
}

/**
 * Decode the 4-corner heatmap. The model is square 256x256 -> 128x128.
 * We argmax each channel and un-letterbox the coordinates back to
 * the original image.
 */
function decodeCornerHeatmap(heat, frameW, frameH, letterbox) {
  const SIZE = 256;
  const Hh = Math.round(Math.sqrt(heat.length / 4));
  const Ww = Hh;
  const { scale, nw, nh, ox, oy } = letterbox;
  // Heatmap is 2x downsampled from the model input (256 -> 128).
  const inputStride = SIZE / Hh; // 2
  const quad = new Array(8);
  let score = 0;
  for (let k = 0; k < 4; k++) {
    const off = k * Hh * Ww;
    let mi = 0, mv = -Infinity;
    for (let i = 0; i < Hh * Ww; i++) if (heat[off + i] > mv) { mv = heat[off + i]; mi = i; }
    const py = Math.floor(mi / Ww);
    const px = mi % Ww;
    // Heatmap pixel (px, py) maps to input pixel (px*stride + 0.5).
    // Input pixel (i, j) maps back to original ((i - ox) / scale,
    // (j - oy) / scale). Use the sub-pixel center.
    const u = (px * inputStride + 0.5 - ox) / scale;
    const v = (py * inputStride + 0.5 - oy) / scale;
    const x0 = u < 0 ? 0 : Math.min(frameW - 1, u);
    const y0 = v < 0 ? 0 : Math.min(frameH - 1, v);
    quad[k * 2]     = x0;
    quad[k * 2 + 1] = y0;
    score += mv;
  }
  return { quad, score: score / 4 };
}

// --- DocAligner LCNet100 (4.8 MB) --------------------------------------
export async function loadDocAlignerLc(path) {
  const session = await loadSession(path);
  const SIZE = 256;
  return {
    name: "docaligner-lc",
    type: "corners",
    warmup() {},
    async detect(frame) {
      const t0 = performance.now();
      const o = await ort();
      const lb = letterboxRgba(frame.rGBA, frame.width, frame.height, SIZE);
      const y = await session.run({ img: new o.Tensor("float32", lb.tensor, [1, 3, SIZE, SIZE]) });
      const heat = y.heatmap.data;
      const { quad, score } = decodeCornerHeatmap(heat, frame.width, frame.height, lb);
      const dt = performance.now() - t0;
      return { quad, score, latencyMs: dt, ok: score > 0.3 };
    },
    async dispose() { await session.release?.(); },
  };
}

// --- DocAligner FastViT-SA24 (83 MB) ----------------------------------
export async function loadDocAlignerFastVit(path) {
  const session = await loadSession(path);
  const SIZE = 256;
  return {
    name: "docaligner-fastvit",
    type: "corners",
    warmup() {},
    async detect(frame) {
      const t0 = performance.now();
      const o = await ort();
      const lb = letterboxRgba(frame.rGBA, frame.width, frame.height, SIZE);
      const y = await session.run({ img: new o.Tensor("float32", lb.tensor, [1, 3, SIZE, SIZE]) });
      const heat = y.heatmap.data;
      const { quad, score } = decodeCornerHeatmap(heat, frame.width, frame.height, lb);
      const dt = performance.now() - t0;
      return { quad, score, latencyMs: dt, ok: score > 0.3 };
    },
    async dispose() { await session.release?.(); },
  };
}

// --- DeepLabV3-MobileNetV3 document mask (44 MB) -----------------------
// Two outputs ("out" and "aux"). We argmax the main "out" channel 0
// vs 1 to get a foreground mask, fit a quadrilateral, and use the
// mean foreground probability inside the inferred quad as the
// confidence score.
export async function loadPageScanDeeplab(path) {
  const session = await loadSession(path);
  const SIZE = 384;
  return {
    name: "pagescan-deeplab",
    type: "mask",
    warmup() {},
    async detect(frame) {
      const t0 = performance.now();
      const o = await ort();
      const W = frame.width, H = frame.height;
      const mean = [0.4611, 0.4359, 0.3905];
      const std  = [0.2193, 0.2150, 0.2109];
      const x = new Float32Array(1 * 3 * SIZE * SIZE);
      const scale = Math.min(SIZE / W, SIZE / H);
      const nw = Math.max(1, Math.round(W * scale));
      const nh = Math.max(1, Math.round(H * scale));
      const ox = (SIZE - nw) >> 1, oy = (SIZE - nh) >> 1;
      for (let y = 0; y < nh; y++) {
        for (let xx = 0; xx < nw; xx++) {
          const sx = Math.min(W - 1, Math.floor(xx / scale));
          const sy = Math.min(H - 1, Math.floor(y / scale));
          const di = (sy * W + sx) * 4;
          const oi = (oy + y) * SIZE + (ox + xx);
          x[0 * SIZE * SIZE + oi] = (frame.rGBA[di]     / 255 - mean[0]) / std[0];
          x[1 * SIZE * SIZE + oi] = (frame.rGBA[di + 1] / 255 - mean[1]) / std[1];
          x[2 * SIZE * SIZE + oi] = (frame.rGBA[di + 2] / 255 - mean[2]) / std[2];
        }
      }
      const y = await session.run({ input: new o.Tensor("float32", x, [1, 3, SIZE, SIZE]) });
      const logits = y.out.data; // (1, 2, Hh, Ww)
      const Hh = Math.round(Math.sqrt(logits.length / 2));
      const Ww = Hh;
      // Resample to full image resolution. Channel 1 is the
      // foreground/document class. The 128x128 heatmap covers the
      // full 384x384 letterboxed input, so we just resample each
      // output pixel onto the original image.
      const mask = new Uint8Array(W * H);
      let sum = 0, count = 0;
      const cellW = Ww / W, cellH = Hh / H;
      for (let yi = 0; yi < H; yi++) {
        for (let xi = 0; xi < W; xi++) {
          const sx = Math.min(Ww - 1, Math.floor(xi * cellW));
          const sy = Math.min(Hh - 1, Math.floor(yi * cellH));
          const l0 = logits[0 * Hh * Ww + sy * Ww + sx];
          const l1 = logits[1 * Hh * Ww + sy * Ww + sx];
          if (l1 > l0) {
            mask[yi * W + xi] = 1;
            const m = Math.max(l0, l1);
            const ex0 = Math.exp(l0 - m);
            const ex1 = Math.exp(l1 - m);
            sum += ex1 / (ex0 + ex1);
            count++;
          }
        }
      }
      const quad = maskToQuad(mask, W, H);
      const score = count > 0 ? (sum / count) : 0;
      const dt = performance.now() - t0;
      return { quad, score, latencyMs: dt, ok: !!quad && score > 0.3 };
    },
    async dispose() { await session.release?.(); },
  };
}

// --- Mask -> 4-corner quad ---------------------------------------------
// Naive axis-aligned bounding box of the mask. The benchmark cares
// about whether the mask is correct; the corner extraction here is
// the worst case and lets us measure the mask accuracy directly via
// the warp / OCR pipeline.
function maskToQuad(mask, W, H) {
  let minX = W, minY = H, maxX = -1, maxY = -1;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const x = i % W, y = (i - x) / W;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  if (maxX < 0) return null;
  return [minX, minY, maxX, minY, maxX, maxY, minX, maxY];
}

// --- Registry ------------------------------------------------------------
export const CANDIDATES = {
  "radial": { load: loadRadial, path: null, type: "corners" },
  "docaligner-lc": { load: loadDocAlignerLc, path: "/tmp/docaligner-lc.onnx", type: "corners" },
  "docaligner-fastvit": { load: loadDocAlignerFastVit, path: "/tmp/docaligner-fastvit.onnx", type: "corners" },
  "pagescan-deeplab": { load: loadPageScanDeeplab, path: "/tmp/pagescan-deeplab.onnx", type: "mask" },
};
