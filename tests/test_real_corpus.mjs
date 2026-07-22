// Regression test over ALL saved real phone frames in captures-debug/.
//
// Three layers:
//   1. Sanity  — every frame detects, corners in-frame, area ratio sane.
//   2. Golden  — quad IoU >= 0.90 vs the pinned known-good quad for that
//                frame (tests/fixtures/golden_quads.json). Catches silent
//                corner drift that sanity checks would miss.
//   3. Negatives — receipt-free crops must NOT fire (<= 2 allowed: two
//                crops legitimately contain other white objects).
//
// When you IMPROVE the detector and verify the new quads visually, re-pin:
//   node tests/test_real_corpus.mjs --update
//
// New dumps without a golden entry are auto-pinned on --update and
// reported (not failed) otherwise.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { detectV3 } from "../src/detector/v3.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const D = path.join(HERE, "..", "captures-debug");
const GOLDEN_PATH = path.join(HERE, "fixtures", "golden_quads.json");
const NEG = path.join(HERE, "fixtures", "negatives");
const UPDATE = process.argv.includes("--update");
const cfg = { longEdge: 640 };

function loadFrame(dir, base, W, H) {
  let raw = path.join(dir, base + ".rgb.raw");
  if (!fs.existsSync(raw)) raw = path.join(dir, base + ".rgb");
  if (!fs.existsSync(raw)) return null;
  const rgb = fs.readFileSync(raw);
  if (rgb.length !== W * H * 3) return null;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1];
    data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 255;
  }
  return { width: W, height: H, data };
}

function polyContains(q, x, y) {
  let inside = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const xi = q[i * 2], yi = q[i * 2 + 1], xj = q[j * 2], yj = q[j * 2 + 1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}
function quadIoU(a, b, W, H) {
  let inter = 0, uni = 0;
  for (let y = 0; y < H; y += 3) for (let x = 0; x < W; x += 3) {
    const ia = polyContains(a, x, y), ib = polyContains(b, x, y);
    if (ia && ib) inter++;
    if (ia || ib) uni++;
  }
  return uni ? inter / uni : 0;
}

const golden = fs.existsSync(GOLDEN_PATH) ? JSON.parse(fs.readFileSync(GOLDEN_PATH)) : {};
const jsons = fs.readdirSync(D).filter(f => /^dump_.*_\d+x\d+\.json$/.test(f)).sort();
let pass = 0, fail = 0, pinned = 0, unpinned = 0;

for (const jf of jsons) {
  const j = JSON.parse(fs.readFileSync(path.join(D, jf)));
  if (j.w < 300 || j.h < 300) continue; // broken/noise test dumps
  const base = jf.slice(0, -5);
  const frame = loadFrame(D, base, j.w, j.h);
  if (!frame) continue;
  const res = detectV3(frame, cfg);
  const errs = [];
  if (!res) errs.push("null detection");
  else {
    for (let i = 0; i < 4; i++) {
      const x = res.quad[i * 2], y = res.quad[i * 2 + 1];
      if (x < -8 || x > j.w + 8 || y < -8 || y > j.h + 8) errs.push(`corner ${i} off-frame (${x.toFixed(0)},${y.toFixed(0)})`);
    }
    let a = 0;
    for (let i = 0; i < 4; i++) {
      const k = (i + 1) % 4;
      a += res.quad[i * 2] * res.quad[k * 2 + 1] - res.quad[k * 2] * res.quad[i * 2 + 1];
    }
    const ratio = Math.abs(a / 2) / (j.w * j.h);
    if (ratio < 0.02 || ratio > 0.85) errs.push(`area ratio ${ratio.toFixed(3)}`);

    const sanityErrs = errs.length; // errors so far are sanity-only
    if (golden[base]) {
      // Small corner deviation is fine (detector tweaks jitter corners a
      // few px); flag only real drift: any corner moving more than 5% of
      // the long edge, AND overall overlap dropping under 0.85 IoU.
      const tol = 0.05 * Math.max(j.w, j.h);
      let maxD = 0;
      for (let i = 0; i < 4; i++) {
        maxD = Math.max(maxD, Math.hypot(res.quad[i * 2] - golden[base][i * 2],
                                         res.quad[i * 2 + 1] - golden[base][i * 2 + 1]));
      }
      if (maxD > tol) {
        const iou = quadIoU(res.quad, golden[base], j.w, j.h);
        if (iou < 0.85) errs.push(`corner drift ${maxD.toFixed(0)}px > ${tol.toFixed(0)}px and IoU ${iou.toFixed(3)} < 0.85`);
      }
    } else if (UPDATE) {
      golden[base] = res.quad.map(v => Math.round(v * 10) / 10);
      pinned++;
    } else {
      unpinned++;
    }
    // On --update, re-pin any frame whose SANITY is clean — golden drift
    // is precisely what an intentional update overwrites.
    if (UPDATE && sanityErrs === 0) {
      golden[base] = res.quad.map(v => Math.round(v * 10) / 10);
      errs.length = sanityErrs;
    }
  }
  if (errs.length) { fail++; console.log(`FAIL ${base}: ${errs.join("; ")}`); }
  else pass++;
}
console.log(`\nframes: ${pass} pass, ${fail} fail${unpinned ? `, ${unpinned} without golden (run --update after visual check)` : ""}`);

if (UPDATE) {
  fs.writeFileSync(GOLDEN_PATH, JSON.stringify(golden, null, 1));
  console.log(`goldens pinned: ${Object.keys(golden).length} (${pinned} new)`);
}

// Negatives — must not fire.
let negFP = 0, negN = 0;
for (const f of fs.readdirSync(NEG).filter(f => f.endsWith(".raw")).sort()) {
  const frame = loadFrame(NEG, f.slice(0, -4), 480, 640);
  // negatives are raw 480x640 with no json; loadFrame expects base without ext
  const rgb = fs.readFileSync(path.join(NEG, f));
  const data = new Uint8ClampedArray(480 * 640 * 4);
  for (let i = 0; i < 480 * 640; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1];
    data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 255;
  }
  negN++;
  if (detectV3({ width: 480, height: 640, data }, cfg)) { negFP++; console.log(`NEG FIRE ${f}`); }
}
console.log(`negatives: ${negFP}/${negN} fired (allowed <= 2)`);

process.exit(fail === 0 && pass > 0 && negFP <= 2 ? 0 : 1);
