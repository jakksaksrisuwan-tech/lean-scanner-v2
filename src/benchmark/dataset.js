// Synthetic labelled-dataset generator for benchmarking the radial detector.
//
// Each generated frame has:
//   - RGBA pixel data (Uint8ClampedArray of length W*H*4)
//   - width, height
//   - ground-truth quad (flat 8-array ordered TL TR BR BL)
//
// Background is one of: dark-wood, light-table, textured-cloth.
//
// Why pure-js (not browser-cavas)? So we can run from node without a
// canvas. The bg + page content are rasterised directly into the output
// RGBA buffer. The rotated page is warped onto the frame using the same
// homography + warpBilinear as the production pipeline.

import { computeHomography } from "../dewarp/homography.js";
import { warpBilinear } from "../dewarp/homography.js";
import { orderCorners } from "../detector/quad.js";

/** Deterministic PRNG — mulberry32. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Render a "page" content as a 3-channel RGB buffer (length = srcW*srcH*3).
 *  Bright paper with sparse text lines. Returned in RGB order (not BGR). */
function makePageContent(srcW, srcH, rng) {
  const data = new Uint8ClampedArray(srcW * srcH * 3).fill(240);  // paper white
  const margin = Math.floor(Math.min(srcW, srcH) * 0.08);
  // Title bar
  for (let y = margin; y < margin + Math.floor(srcH * 0.06); y++) {
    for (let x = margin; x < srcW - margin; x++) {
      const idx = (y * srcW + x) * 3;
      data[idx] = 30; data[idx + 1] = 30; data[idx + 2] = 30;
    }
  }
  // Body lines (horizontal darker strips with varying widths)
  const lineCount = 12;
  const lineTop = margin + Math.floor(srcH * 0.12);
  const lineH = Math.floor((srcH - 2 * margin - Math.floor(srcH * 0.10)) / lineCount);
  for (let i = 0; i < lineCount; i++) {
    const lineW = Math.floor(srcW * (0.5 + 0.4 * (i % 3) / 2));
    const yLine = lineTop + i * lineH;
    const th = Math.max(2, Math.floor(lineH * 0.25));
    for (let y = yLine; y < yLine + th && y < srcH - margin; y++) {
      for (let x = margin; x < margin + lineW && x < srcW - margin; x++) {
        const idx = (y * srcW + x) * 3;
        data[idx] = 80; data[idx + 1] = 80; data[idx + 2] = 80;
      }
    }
  }
  // Sparse noise on paper for realism (deterministic)
  for (let i = 0; i < 200; i++) {
    const x = Math.floor(rng() * srcW);
    const y = Math.floor(rng() * srcH);
    const idx = (y * srcW + x) * 3;
    const n = Math.floor(rng() * 30) - 15;
    data[idx]     = Math.max(0, Math.min(255, data[idx] + n));
    data[idx + 1] = Math.max(0, Math.min(255, data[idx + 1] + n));
    data[idx + 2] = Math.max(0, Math.min(255, data[idx + 2] + n));
  }
  return data;
}

/** Build the background to a chosen "scene" type. Returns RGBA Uint8ClampedArray.
 *
 *   'dark-wood'    : dark brown / black
 *   'light-table'  : beige / off-white, slightly textured
 *   'textured-cloth': mid-grey with strong edge noise
 */
function makeBackground(W, H, type, rng) {
  const data = new Uint8ClampedArray(W * H * 3);
  // Base color
  let baseR, baseG, baseB;
  if (type === "dark-wood") {
    baseR = 40; baseG = 30; baseB = 25;
  } else if (type === "light-table") {
    baseR = 230; baseG = 225; baseB = 215;
  } else {
    baseR = 130; baseG = 130; baseB = 130;
  }
  for (let i = 0; i < W * H; i++) {
    data[i * 3] = baseR;
    data[i * 3 + 1] = baseG;
    data[i * 3 + 2] = baseB;
  }
  // Add texture: noise + larger blobs
  // (1) Per-pixel noise
  for (let i = 0; i < W * H; i++) {
    const n = Math.floor((rng() - 0.5) * 40);
    data[i * 3]     = Math.max(0, Math.min(255, data[i * 3] + n));
    data[i * 3 + 1] = Math.max(0, Math.min(255, data[i * 3 + 1] + n));
    data[i * 3 + 2] = Math.max(0, Math.min(255, data[i * 3 + 2] + n));
  }
  // (2) Large blobs (deterministic gradient patches) - simulate wood grain,
  //     table scratches, cloth weave
  const numBlobs = Math.floor(W * H / 4000);
  for (let i = 0; i < numBlobs; i++) {
    const cx = rng() * W;
    const cy = rng() * H;
    const r = 8 + rng() * 30;
    const dark = (type === "dark-wood") ? Math.floor(rng() * 30) + 10
              : (type === "light-table") ? Math.floor(rng() * 30) - 15
              : Math.floor(rng() * 60) - 30;
    for (let y = Math.max(0, Math.floor(cy - r)); y < Math.min(H, Math.ceil(cy + r)); y++) {
      for (let x = Math.max(0, Math.floor(cx - r)); x < Math.min(W, Math.ceil(cx + r)); x++) {
        const dx = x - cx, dy = y - cy;
        const falloff = Math.max(0, 1 - Math.sqrt(dx * dx + dy * dy) / r);
        const idx = (y * W + x) * 3;
        const adjust = dark * falloff;
        data[idx]     = Math.max(0, Math.min(255, data[idx] - adjust));
        data[idx + 1] = Math.max(0, Math.min(255, data[idx + 1] - adjust));
        data[idx + 2] = Math.max(0, Math.min(255, data[idx + 2] - adjust));
      }
    }
  }
  return data;
}

/** Compute the GT quad: a rectangle of size docW × docH centered at frame
 *  center, rotated by `theta` radians. Returns flat 8-array TL TR BR BL. */
function gtQuad(W, H, theta, docW, docH) {
  const cx = W / 2, cy = H / 2;
  const hw = docW / 2, hh = docH / 2;
  const cosT = Math.cos(theta), sinT = Math.sin(theta);
  // Corners in local coords (CCW from top-left when theta=0)
  const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  const world = local.map(([x, y]) => {
    const wx = cx + x * cosT - y * sinT;
    const wy = cy + x * sinT + y * cosT;
    return [wx, wy];
  });
  // Re-order to TL TR BR BL using orderCorners
  const flat = [].concat(...world);
  return orderCorners(flat);
}

/** Composite a rotated page onto a background frame.
 *
 *   bg:      Uint8ClampedArray length W*H*3 (RGB)
 *   page:    Uint8ClampedArray length srcW*srcH*3 (RGB)
 *   dstQuad: flat 8 of 4 (x,y) corners in frame coords
 *
 *   Pixels where the page is "brighter than the bg by a margin" overwrite
 *   the bg. This is a simple compositing — good enough for benchmarks.
 */
function composite(bg, W, H, page, srcW, srcH, dstQuad) {
  // Warp page onto dstQuad into a temp buffer using warpBilinear.
  const warped = warpBilinear(page, srcW, srcH, 3, computeHomography(
    [0, 0, srcW - 1, 0, srcW - 1, srcH - 1, 0, srcH - 1],
    dstQuad
  ), H, W);

  // Composite: page where noticeably brighter than bg
  const out = new Uint8ClampedArray(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    const pm = (warped[i * 3] + warped[i * 3 + 1] + warped[i * 3 + 2]) / 3;
    const bm = (bg[i * 3] + bg[i * 3 + 1] + bg[i * 3 + 2]) / 3;
    if (pm > bm + 5 && pm > 50) {
      // page pixel
      out[i * 3]     = warped[i * 3];
      out[i * 3 + 1] = warped[i * 3 + 1];
      out[i * 3 + 2] = warped[i * 3 + 2];
    } else {
      // bg
      out[i * 3]     = bg[i * 3];
      out[i * 3 + 1] = bg[i * 3 + 1];
      out[i * 3 + 2] = bg[i * 3 + 2];
    }
  }
  return out;
}

/** Add RGBA alpha channel (all 255) to an RGB buffer. */
function rgbToRgba(rgb) {
  const n = rgb.length / 3;
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    rgba[i * 4]     = rgb[i * 3];
    rgba[i * 4 + 1] = rgb[i * 3 + 1];
    rgba[i * 4 + 2] = rgb[i * 3 + 2];
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

/**
 * Generate one labelled frame.
 * @param {object} opts
 * @param {number} opts.width
 * @param {number} opts.height
 * @param {number} opts.theta rotation in radians (0 = upright)
 * @param {string} opts.bg "dark-wood" | "light-table" | "textured-cloth"
 * @param {number} opts.seed
 * @returns {{ rGBA: Uint8ClampedArray, width: number, height: number, gtQuad: number[], bg: string, theta: number }}
 */
export function generateFrame(opts) {
  const W = opts.width, H = opts.height;
  const theta = opts.theta;
  const bgType = opts.bg;
  const rng = mulberry32(opts.seed);

  const bg = makeBackground(W, H, bgType, rng);

  // Doc is 60% wide × 50% tall
  const docW = W * 0.6, docH = H * 0.5;
  const pageSrcW = Math.floor(W * 0.4), pageSrcH = Math.floor(H * 0.35);
  const page = makePageContent(pageSrcW, pageSrcH, rng);

  // Compute GT quad (same as quad returned to user — TL TR BR BL ordered)
  const quad = gtQuad(W, H, theta, docW, docH);

  // Composite page onto bg using dstQuad = quad
  const composite_ = composite(bg, W, H, page, pageSrcW, pageSrcH, quad);

  return {
    rGBA: rgbToRgba(composite_),
    width: W,
    height: H,
    gtQuad: quad,
    bg: bgType,
    theta,
  };
}

/** Generate a dataset of N frames at varying rotations and bg types.
 *  Returns an array of the same frame objects as generateFrame().
 *
 *  Defaults: 250 frames, 3 bg types, ~14 angles × 6 doc-position variants.
 */
export function generateDataset({ count = 250, width = 320, height = 240, seed = 1 } = {}) {
  const bgTypes = ["dark-wood", "light-table", "textured-cloth"];
  const anglesDeg = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85];
  const framesPerCombo = Math.ceil(count / (bgTypes.length * anglesDeg.length));

  const rng = mulberry32(seed);
  const frames = [];
  let i = 0;
  for (const bg of bgTypes) {
    for (const angDeg of anglesDeg) {
      const theta = (angDeg * Math.PI) / 180;
      // Position variants: jitter the doc away from frame center
      for (let v = 0; v < framesPerCombo && frames.length < count; v++) {
        // Skip frames where the doc would land mostly off-frame at large theta.
        // Center the doc exactly only for now; jittering is for later.
        frames.push(generateFrame({
          width, height, theta, bg,
          seed: seed + i,
        }));
        i++;
      }
    }
  }
  return frames;
}
