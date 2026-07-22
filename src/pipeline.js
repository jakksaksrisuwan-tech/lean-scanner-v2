// Main auto-capture pipeline.
//
// Wires detector -> quality scorer -> lock gate -> dewarp -> save.
// Direct port of autocapture/pipeline.py.

import { detectV3 as detectHighContrast } from "./detector/v3.js";
import { score } from "./quality.js";
import {
  LockGate,
  STATE_LOCKED,
} from "./tracker/lock.js";
import { warpQuad } from "./dewarp/homography.js";
import { makeBinaryMask, shadeCorrect } from "./dewarp/shading.js";
import { claheSimple, adaptiveThreshold } from "./dewarp/contrast.js";

/**
 * Run one frame through the full pipeline.
 * @param {ImageData} imageData the camera frame
 * @param {LockGate} gate
 * @param {object} cfg full pipeline config
 * @returns {[string, number[]|null, number, {warped: Uint8ClampedArray, w: number, h: number, mask: Uint8Array, quad: number[]} | null]}
 */
export function processFrame(imageData, gate, cfg) {
  const H = imageData.height, W = imageData.width;
  const data = imageData.data;

  // 1. Detect (v3: min-channel whiteness -> Otsu -> biggest blob -> minAreaRect).
  const det = detectHighContrast(imageData, cfg.detector);
  if (det === null) {
    const [state, smoothed, q] = gate.update(null, 0.0);
    return [state, smoothed, q, null];
  }
  const rawQuad = det.quad;

  // 2. Score
  const q = score(rawQuad, data, H, W, cfg.quality);

  // 3. Update gate (smoothed quad comes back here)
  const [state, smoothed, qOut] = gate.update(rawQuad, q);

  // 4. If locked, dewarp + enhance
  if (state === STATE_LOCKED) {
    const result = dewarpAndEnhance(data, H, W, smoothed, cfg);
    return [state, smoothed, qOut, {
      warped: result.warped,
      w: result.w,
      h: result.h,
      mask: makeBinaryMask([H, W], smoothed),
      quad: smoothed,
    }];
  }

  return [state, smoothed, qOut, null];
}

function dewarpAndEnhance(frameData, H, W, quad, cfg) {
  // 1. Perspective warp. The python pipeline works on BGR; we have RGBA.
  // The math is the same, we just feed RGB triplets.
  const rgb = new Uint8ClampedArray(H * W * 3);
  for (let i = 0; i < H * W; i++) {
    rgb[i * 3]     = frameData[i * 4];
    rgb[i * 3 + 1] = frameData[i * 4 + 1];
    rgb[i * 3 + 2] = frameData[i * 4 + 2];
  }
  const warped = warpQuad(rgb, H, W, 3, quad, cfg.dewarp.fixedAspect);

  // 2. Flat-field shading (gated the same way as the python pipeline)
  let workData = warped.data;
  let workH = warped.height, workW = warped.width;
  if (cfg.dewarp.shadingKernel > 0 && cfg.dewarp.shadingKernel < 200) {
    // Build a mask covering the whole output (we just warped the doc to fill it)
    const mask = new Uint8Array(workH * workW).fill(1);
    workData = shadeCorrect(workData, mask, workH, workW, cfg.dewarp.shadingKernel);
  }

  // 3. Contrast
  // Compute grayscale
  const gray = new Uint8Array(workH * workW);
  for (let i = 0; i < workH * workW; i++) {
    gray[i] = Math.round((workData[i * 3] + workData[i * 3 + 1] + workData[i * 3 + 2]) / 3);
  }

  let outRGB;
  if (cfg.dewarp.binarize) {
    const bw = adaptiveThreshold(gray, workH, workW, 11, 2);
    outRGB = new Uint8ClampedArray(workH * workW * 3);
    for (let i = 0; i < workH * workW; i++) {
      outRGB[i * 3] = bw[i]; outRGB[i * 3 + 1] = bw[i]; outRGB[i * 3 + 2] = bw[i];
    }
  } else {
    const eq = claheSimple(gray, workH, workW, cfg.dewarp.claheClip, cfg.dewarp.claheGrid);
    outRGB = new Uint8ClampedArray(workH * workW * 3);
    for (let i = 0; i < workH * workW; i++) {
      outRGB[i * 3] = eq[i]; outRGB[i * 3 + 1] = eq[i]; outRGB[i * 3 + 2] = eq[i];
    }
  }
  return { warped: outRGB, w: warped.width, h: warped.height };
}

/**
 * Stateful wrapper that holds the gate + counters. Mirrors autocapture.pipeline.Pipeline.
 */
export class Pipeline {
  constructor(cfg) {
    this.cfg = cfg;
    this.gate = new LockGate(cfg.lock);
    this.frameIdx = 0;
    this.captures = [];   // array of {warped, w, h}
    this.stateTrace = [];
  }

  step(imageData) {
    const [state, smoothed, q, capture] = processFrame(imageData, this.gate, this.cfg);
    this.frameIdx += 1;
    this.stateTrace.push(state);
    if (capture !== null) {
      this.captures.push({ warped: capture.warped, w: capture.w, h: capture.h });
      this.gate.fire();
    }
    // Re-run detector once for UI overlay (segmentation, centroid).
    const det = detectHighContrast(imageData, this.cfg.detector);
    return {
      gateState: state,
      smoothedQuad: smoothed,
      quality: q,
      capture,
      // UI-only fields (may be null if no detection):
      rawQuad: det ? det.quad : null,
      segmentation: det ? det.segmentation : null,
      centroid: det ? { cx: det.cx, cy: det.cy } : null,
    };
  }
}