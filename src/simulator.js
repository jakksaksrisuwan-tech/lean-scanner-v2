// Synthetic document video source.
//
// Direct port of simulator.py. Renders a sequence of frames showing a
// document at various positions and orientations, so the auto-capture
// pipeline can be exercised end-to-end without a real camera.
//
// Used by:
//   - the browser app's "simulator" button (offline smoke test)
//   - the test suite for end-to-end verification

import { computeHomography, warpBilinear } from "./dewarp/homography.js";

// Real receipt pages baked from the SRD bills dataset (assets/sim/*.jpg,
// dewarped crops). Loaded async; the synthetic B&W page is the immediate
// fallback so the sim works before the fetch lands (and offline).
async function loadReceiptPage(rng) {
  const idx = await (await fetch("assets/sim/index.json")).json();
  const name = idx[Math.floor(rng() * idx.length)];
  const img = new Image();
  img.src = "assets/sim/" + name;
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, c.width, c.height);
}

/**
 * Render a clean B&W document page to an ImageData on an offscreen canvas.
 * Returns ImageData (RGBA).
 */
function drawDocumentPage(w, h, lines = 12) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, h);

  const margin = Math.floor(Math.min(w, h) * 0.08);
  ctx.fillStyle = "black";
  ctx.fillRect(margin, margin, w - 2 * margin, Math.floor(h * 0.06));

  const lineH = Math.floor((h - 2 * margin - Math.floor(h * 0.10)) / lines);
  let y = margin + Math.floor(h * 0.12);
  ctx.fillStyle = "rgb(60,60,60)";
  for (let i = 0; i < lines; i++) {
    const lineW = Math.floor(w * (0.5 + 0.4 * (i % 3) / 2));
    ctx.fillRect(margin, y, lineW, Math.floor(lineH * 0.3));
    y += lineH;
  }
  return ctx.getImageData(0, 0, w, h);
}

/**
 * Warp the page image onto a 4-corner quad in the destination canvas.
 * Pure-JS implementation via bilinear inverse-warp. Works on every
 * browser (no DOMMatrix / setTransform(matrix) dependency, which is the
 * reason the previous version failed on iOS Chrome and older Safari).
 *
 * @param {ImageData} pageImg  source page (RGBA)
 * @param {number[]} dstQuad   flat 8-array of 4 corners
 * @param {number} outW
 * @param {number} outH
 * @returns {ImageData} the warped composite
 */
function warpPerspective(pageImg, dstQuad, outW, outH) {
  const srcH = pageImg.height, srcW = pageImg.width;

  // Use the homography module's warpBilinear — bilinear inverse-warp in
  // pure JS. iOS Chrome / Safari have intermittent bugs with creating an
  // intermediate 2D context for imageData manipulation, so we go through
  // exactly one canvas (the OUT canvas) at the end.
  const H = computeHomography(
    [0, 0, srcW - 1, 0, srcW - 1, srcH - 1, 0, srcH - 1],
    dstQuad
  );

  // Convert pageImg (RGBA Uint8ClampedArray) into RGB triples.
  // The warped RGB is built directly into our out pixels; we don't
  // stage a tmp canvas because iOS Chrome occasionally returns null
  // from getContext('2d') for tmp canvases created mid-frame.
  const srcRGB = new Uint8ClampedArray(srcW * srcH * 3);
  for (let i = 0; i < srcW * srcH; i++) {
    srcRGB[i * 3]     = pageImg.data[i * 4];
    srcRGB[i * 3 + 1] = pageImg.data[i * 4 + 1];
    srcRGB[i * 3 + 2] = pageImg.data[i * 4 + 2];
  }
  const warpedRGB = warpBilinear(srcRGB, srcH, srcW, 3, H, outH, outW);

  // Build a fresh ImageData via the OUT canvas (one stage only).
  // We pass our RGBA directly into createImageData: route one.
  const outCanvas = document.createElement("canvas");
  outCanvas.width = outW;
  outCanvas.height = outH;
  const outCtx = outCanvas.getContext("2d");
  if (!outCtx) {
    throw new Error("canvas.getContext('2d') returned null on " + outW + "x" + outH);
  }
  const imgData = outCtx.createImageData(outW, outH);
  for (let i = 0; i < outW * outH; i++) {
    imgData.data[i * 4]     = warpedRGB[i * 3];
    imgData.data[i * 4 + 1] = warpedRGB[i * 3 + 1];
    imgData.data[i * 4 + 2] = warpedRGB[i * 3 + 2];
    imgData.data[i * 4 + 3] = 255;
  }
  return imgData;
}

/**
 * SyntheticDocumentSource: emits a stream of frames simulating a
 * document in front of a camera.
 *
 * Pattern (matches the python simulator):
 *   0..30   : doc enters from the right
 *   30..90  : holds steady near the center, gentle wobble
 *   90..120 : doc moves away / out of frame
 *   120..150: blank window (cooldown)
 *   150..180: doc re-enters
 *   180..240: second hold
 *   else    : blank
 */
export class SyntheticDocumentSource {
  constructor({ width = 640, height = 480, seed = 42 } = {}) {
    this.W = width; this.H = height;
    this.frameIdx = 0;
    this.rng = mulberry32(seed);
    this.page = drawDocumentPage(Math.floor(width * 0.6), Math.floor(height * 0.75));
    this.bgCanvas = buildBackground(width, height, this.rng);
    // Swap in a real receipt from the bills dataset when it loads.
    loadReceiptPage(this.rng).then((p) => { this.page = p; }).catch(() => {});
  }

  _quadForFrame() {
    const f = this.frameIdx;
    if (120 <= f && f < 150) return null;
    let cx, cy, scale;
    if (f < 30) {
      const t = f / 30.0;
      cx = this.W * (0.95 - 0.35 * t);
      cy = this.H * 0.55;
      scale = 0.5 + 0.4 * t;
    } else if (f < 90) {
      const t = (f - 30) / 60.0;
      cx = this.W * 0.55 + 4 * Math.sin(2 * Math.PI * t * 1.5);
      cy = this.H * 0.50 + 3 * Math.cos(2 * Math.PI * t * 2.0);
      scale = 0.9;
    } else if (f < 120) {
      const t = (f - 90) / 30.0;
      cx = this.W * (0.55 - 0.4 * t);
      cy = this.H * (0.50 - 0.2 * t);
      scale = 0.9 * (1.0 - 0.2 * t);
    } else if (f < 180) {
      const t = (f - 150) / 30.0;
      cx = this.W * (0.95 - 0.40 * t);
      cy = this.H * 0.55;
      scale = 0.5 + 0.4 * t;
    } else if (f < 240) {
      const t = (f - 180) / 60.0;
      cx = this.W * 0.55 + 5 * Math.sin(2 * Math.PI * t * 1.2);
      cy = this.H * 0.50;
      scale = 0.9;
    } else {
      return null;
    }

    const skewX = 0.05 * Math.sin(f * 0.1);
    const skewY = 0.03 * Math.cos(f * 0.13);
    const docW = Math.floor(this.W * 0.55 * scale);
    const docH = Math.floor(this.H * 0.7 * scale);
    const halfW = docW / 2, halfH = docH / 2;
    return [
      cx - halfW * (1 + skewX), cy - halfH * (1 - skewY),
      cx + halfW * (1 - skewX), cy - halfH * (1 + skewY),
      cx + halfW * (1 + skewX), cy + halfH * (1 - skewY),
      cx - halfW * (1 - skewX), cy + halfH * (1 + skewY),
    ];
  }

  /**
   * Static mode for demo/testing: a single doc-in-frame image, no
   * motion. Use this so the lock-gate can satisfy its 8-stable-frame
   * debounce on a doc that isn't moving in and out of view.
   */
  staticFrame() {
    // Render once
    const frame = this.read(60);  // frame 60 = peak "hold" position
    this.frameIdx = 60;            // stay there forever
    return frame;
  }

  /**
   * Read one frame. The optional `atFrame` parameter lets the loop
   * request a specific frame (e.g. for the demo's hold phase). Otherwise
   * advances by 1.
   */
  read(atFrame) {
    if (atFrame !== undefined) this.frameIdx = atFrame;
    const out = document.createElement("canvas");
    out.width = this.W; out.height = this.H;
    const outCtx = out.getContext("2d");
    if (!outCtx) {
      // iOS Chrome occasionally returns null here when the canvas is
      // created off-screen. Surface the error so the loop can recover.
      throw new Error("simulator: canvas 2d context unavailable");
    }
    outCtx.drawImage(this.bgCanvas, 0, 0);

    const quad = this._quadForFrame();
    if (quad !== null) {
      const warped = warpPerspective(this.page, quad, this.W, this.H);
      // Composite only INSIDE the quad — warpBilinear fills the exterior
      // with black, and blitting that wholesale erased the background
      // (old sim dumps show a black bg for exactly this reason).
      const bgData = outCtx.getImageData(0, 0, this.W, this.H);
      const q = quad;
      const inQuad = (x, y) => {
        let inside = false;
        for (let i = 0, j = 3; i < 4; j = i++) {
          const xi = q[i * 2], yi = q[i * 2 + 1], xj = q[j * 2], yj = q[j * 2 + 1];
          if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
        }
        return inside;
      };
      for (let y = 0; y < this.H; y++) {
        for (let x = 0; x < this.W; x++) {
          if (!inQuad(x, y)) continue;
          const i = (y * this.W + x) * 4;
          bgData.data[i] = warped.data[i];
          bgData.data[i + 1] = warped.data[i + 1];
          bgData.data[i + 2] = warped.data[i + 2];
        }
      }
      outCtx.putImageData(bgData, 0, 0);
    }
    this.frameIdx += 1;
    return { ok: true, imageData: outCtx.getImageData(0, 0, this.W, this.H) };
  }

  release() {
    this.bgCanvas = null;
  }
}

function buildBackground(W, H, rng) {
  // Wood-grain-ish table: warm base, horizontal streaks (low-freq per-row
  // luma), plank seams, per-pixel noise. Matches the real-corpus failure
  // modes far better than the old flat dark fill.
  const c = document.createElement("canvas");
  c.width = W; c.height = H;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(W, H);
  const rowTone = new Float32Array(H);
  let t = 0;
  for (let y = 0; y < H; y++) {
    t = 0.92 * t + (rng() - 0.5) * 6;                    // wandering grain
    rowTone[y] = 18 * Math.sin(y * 0.05) + t;
  }
  const seam = Math.floor(H * (0.3 + 0.4 * rng()));      // one plank seam
  for (let y = 0; y < H; y++) {
    const grain = rowTone[y] + (Math.abs(y - seam) < 2 ? -35 : 0);
    for (let x = 0; x < W; x++) {
      const n = (rng() - 0.5) * 14;
      const streak = 10 * Math.sin(x * 0.01 + y * 0.2);
      const i = (y * W + x) * 4;
      img.data[i]     = clamp(120 + grain + streak + n);   // R
      img.data[i + 1] = clamp(78 + grain * 0.8 + streak * 0.7 + n);  // G
      img.data[i + 2] = clamp(48 + grain * 0.6 + streak * 0.5 + n);  // B
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function clamp(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// Deterministic RNG matching numpy's default_rng behavior for the
// small uses here (just integer draws via floor(rng()*N)). mulberry32
// is a 32-bit PRNG that gives stable sequences across runs.
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