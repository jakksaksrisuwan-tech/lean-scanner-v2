// OCR-labeled bill dataset for the segment-model benchmark.
//
// Each frame is a synthesized "receipt" containing a small block of
// real, OCR-readable text rendered with a system font, then composited
// onto a tinted page, then pasted onto one of three background types.
// The GT quad, the per-frame text, and the rotated-page image are
// captured together so the benchmark can pass the inferred warped
// output to Apple Vision and compare the recognized text against the
// original.

import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { generateFrame } from "./dataset.js";
import { computeHomography, warpBilinear } from "../dewarp/homography.js";

// System font usable in 1+ on every Mac.
const FONT = "/System/Library/Fonts/Supplemental/Arial.ttf";

let _py = null;
function py() {
  if (_py) return _py;
  const r = spawnSync("python3", ["-c", "import sys; print(sys.version)"], { encoding: "utf8" });
  if (r.status !== 0) throw new Error("python3 unavailable: " + r.stderr);
  return (_py = "python3");
}

/**
 * Render a small text page as an RGB Uint8Array. We use Pillow so the
 * rendered text is real rasterised Arial (not stroked by hand) and
 * therefore matches what Apple Vision expects.
 */
function renderTextPage(pageSrcW, pageSrcH, lines) {
  const code = `
from PIL import Image, ImageDraw, ImageFont
import numpy as np
W,H = ${pageSrcW},${pageSrcH}
img = Image.new("RGB", (W,H), (245,245,245))
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype(${JSON.stringify(FONT)}, max(14, H//12))
except Exception:
    f = ImageFont.load_default()
y = max(8, H//12)
for line in ${JSON.stringify(lines)}:
    d.text((max(8, W//14), y), line, font=f, fill=(20,20,20))
    y += int(f.size*1.4)
a = np.array(img, dtype=np.uint8).tobytes()
open("/tmp/_ls_bill.bin","wb").write(a)
`;
  const r = spawnSync(py(), ["-c", code], { encoding: "utf8" });
  if (r.status !== 0) throw new Error("renderTextPage failed: " + r.stderr);
  const buf = readFileSync("/tmp/_ls_bill.bin");
  return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
}

/**
 * Generate a single OCR-labeled frame. We re-render the dataset
 * generator's page-compositing step but paint the text into the page
 * before warping it onto the background, so the warped output is
 * legible text on a tinted rectangle, with known GT text.
 */
export function generateOcrFrame(opts) {
  const W = opts.width, H = opts.height;
  const theta = opts.theta;
  const bg = opts.bg;
  const lines = opts.text || [
    "RECEIPT 001",
    "TOTAL $5.99",
    "07/20/2026",
    "lean_scanner",
  ];
  const pageSrcW = Math.floor(W * 0.4);
  const pageSrcH = Math.floor(H * 0.35);
  const pageRgb = renderTextPage(pageSrcW, pageSrcH, lines);

  // Build the background the same way dataset.js does, then warp
  // the rendered text page onto it. We use the same makeBackground
  // approach: deterministically fill the frame with a tinted color +
  // noise, then composite the warped page on top.
  const baseRgba = synthFrame(W, H, theta, bg, pageRgb, pageSrcW, pageSrcH, opts.seed);

  return {
    rGBA: baseRgba,
    width: W, height: H,
    gtQuad: quadForRotatedDoc(W, H, theta, W * 0.6, H * 0.5),
    bg, theta,
    text: lines.join("\n"),
  };
}

function synthFrame(W, H, theta, bg, pageRgb, pageSrcW, pageSrcH, seed) {
  // Deterministic background that matches dataset.js's three types.
  let baseR, baseG, baseB;
  if (bg === "dark-wood") { baseR = 40; baseG = 30; baseB = 25; }
  else if (bg === "light-table") { baseR = 230; baseG = 225; baseB = 215; }
  else { baseR = 130; baseG = 130; baseB = 130; }
  const bgRgb = new Uint8ClampedArray(W * H * 3);
  // Tiny deterministic noise (same rng as dataset.js, but we only
  // need a static frame, so seed=seed).
  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = 0; i < W * H; i++) {
    const n = Math.floor((rng() - 0.5) * 40);
    bgRgb[i * 3]     = Math.max(0, Math.min(255, baseR + n));
    bgRgb[i * 3 + 1] = Math.max(0, Math.min(255, baseG + n));
    bgRgb[i * 3 + 2] = Math.max(0, Math.min(255, baseB + n));
  }
  // Warp the rendered page onto the rotated dst quad.
  const dstQuad = quadForRotatedDoc(W, H, theta, W * 0.6, H * 0.5);
  const srcQuad = [0, 0, pageSrcW - 1, 0, pageSrcW - 1, pageSrcH - 1, 0, pageSrcH - 1];
  const Hm = computeHomography(srcQuad, dstQuad);
  const warped = warpBilinear(pageRgb, pageSrcH, pageSrcW, 3, Hm, H, W);
  // Composite where the warped page is brighter than the bg.
  const out = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const pm = (warped[i * 3] + warped[i * 3 + 1] + warped[i * 3 + 2]) / 3;
    const bm = (bgRgb[i * 3] + bgRgb[i * 3 + 1] + bgRgb[i * 3 + 2]) / 3;
    if (pm > bm + 5 && pm > 50) {
      out[i * 4]     = warped[i * 3];
      out[i * 4 + 1] = warped[i * 3 + 1];
      out[i * 4 + 2] = warped[i * 3 + 2];
    } else {
      out[i * 4]     = bgRgb[i * 3];
      out[i * 4 + 1] = bgRgb[i * 3 + 1];
      out[i * 4 + 2] = bgRgb[i * 3 + 2];
    }
    out[i * 4 + 3] = 255;
  }
  return out;
}

function quadForRotatedDoc(W, H, theta, docW, docH) {
  const cx = W / 2, cy = H / 2;
  const hw = docW / 2, hh = docH / 2;
  const c = Math.cos(theta), s = Math.sin(theta);
  const local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]];
  const world = local.map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
  // Same TL TR BR BL heuristic as quad.js (sum + diff).
  const sums = world.map(([x, y]) => x + y);
  const diffs = world.map(([x, y]) => y - x);
  const argmin = a => a.indexOf(Math.min(...a));
  const argmax = a => a.indexOf(Math.max(...a));
  const tl = world[argmin(sums)];
  const br = world[argmax(sums)];
  const tr = world[argmin(diffs)];
  const bl = world[argmax(diffs)];
  return [...tl, ...tr, ...br, ...bl];
}

/**
 * Build a dataset of N OCR-labeled frames.
 * Defaults to 60 frames (3 bgs × 20 angles) — small enough for the
 * full Vision OCR round-trip to finish in under a minute.
 */
export function generateOcrDataset({ count = 60, width = 320, height = 240, seed = 1 } = {}) {
  const bgTypes = ["dark-wood", "light-table", "textured-cloth"];
  const anglesDeg = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 180];
  const framesPerCombo = Math.ceil(count / (bgTypes.length * anglesDeg.length));
  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = [];
  let i = 0;
  for (const bg of bgTypes) {
    for (const angDeg of anglesDeg) {
      const theta = (angDeg * Math.PI) / 180;
      for (let v = 0; v < framesPerCombo && out.length < count; v++) {
        out.push(generateOcrFrame({
          width, height, theta, bg,
          seed: seed + i,
          text: [
            "RECEIPT " + (1000 + Math.floor(rng() * 9000)),
            "TOTAL $" + (1 + rng() * 99).toFixed(2),
            "07/20/2026",
            "lean_scanner",
          ],
        }));
        i++;
      }
    }
  }
  return out;
}
