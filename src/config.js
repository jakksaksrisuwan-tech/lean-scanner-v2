// Pipeline config. Mirrors autocapture/config.py.
//
// One config object with nested sub-configs. The defaults here match
// config.yaml. To override, pass a partial config to loadConfig().

export const DEFAULT_CONFIG = {
  cameraIndex: 0,
  fps: 30,
  outputDir: "./captures",
  detector: {
    longEdge: 640,
    cannyLow: 75,
    cannyHigh: 200,
    morphKernel: 9,
    morphIterations: 3,
    minQuadAreaRatio: 0.05,
    maxQuadAreaRatio: 0.98,
    minAspect: 0.2,
    maxAspect: 5.0,
  },
  quality: {
    captureThreshold: 0.70,
    weightArea: 0.30,
    weightStraightness: 0.20,
    weightSharpness: 0.25,
    weightExposure: 0.10,
    weightBlur: 0.15,
    minAreaRatio: 0.10,
    targetAreaRatio: 0.30,
    sharpnessFull: 200.0,
    blurFull: 80.0,
    exposureLow: 60.0,
    exposureHigh: 220.0,
    glarePixelThreshold: 240,
    glareMaxRatio: 0.15,
  },
  lock: {
    nStableFrames: 8,
    cornerDistPx: 20.0,
    qualityReleaseRatio: 0.8,
    cooldownMs: 800,
  },
  dewarp: {
    fixedAspect: "AUTO",  // natural aspect from quad edges — receipts are not A4; forcing a paper ratio squashes slips
    shadingKernel: 201,
    claheClip: 2.0,
    claheGrid: 8,
    binarize: false,
  },
};

/**
 * Deep-merge overrides into defaults.
 * @param {object} base
 * @param {object} overrides
 */
export function loadConfig(overrides) {
  const out = JSON.parse(JSON.stringify(DEFAULT_CONFIG));
  if (overrides) {
    deepMerge(out, overrides);
  }
  return out;
}

function deepMerge(target, source) {
  for (const key of Object.keys(source)) {
    const v = source[key];
    if (v && typeof v === "object" && !Array.isArray(v) &&
        target[key] && typeof target[key] === "object" && !Array.isArray(target[key])) {
      deepMerge(target[key], v);
    } else {
      target[key] = v;
    }
  }
}