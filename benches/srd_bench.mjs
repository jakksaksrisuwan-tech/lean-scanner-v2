// SRD real-receipt benchmark: runs the PRODUCTION JS detector (v3.js)
// over the 200-image downloaded receipts dataset. This is the honest
// benchmark — the synth bench overfits (see HANDOFF.md "the synth
// benchmark misled us").
//
// No ground-truth quads exist for SRD, so the metric is detection rate
// plus sanity (quad area, solidity via confidence). Visual grading:
// re-render with tests/render grid scripts.
//
// Run:  node benches/srd_bench.mjs
//
// First run decodes jpg -> raw RGB (long edge 640) via python3/cv2 into
// <dataset>/.raw640/ (~50 MB); later runs reuse the cache.

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { detectV3 } from "../src/detector/v3.js";

const DATASET = "/Users/jsaksrisuwan/workspace/receipts-dataset";
const CACHE = path.join(DATASET, ".raw640");

if (!fs.existsSync(CACHE)) {
  console.log("decoding jpgs -> raw cache (one-time)...");
  const py = `
import cv2, glob, os, json
os.makedirs(${JSON.stringify(CACHE)}, exist_ok=True)
meta = {}
for f in sorted(glob.glob(${JSON.stringify(DATASET)} + "/*.jpg")):
    im = cv2.imread(f)
    h, w = im.shape[:2]
    s = 640 / max(h, w)
    if s < 1: im = cv2.resize(im, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    rgb = im[:, :, ::-1]
    name = os.path.basename(f)[:-4]
    rgb.tofile(os.path.join(${JSON.stringify(CACHE)}, name + ".raw"))
    meta[name] = [rgb.shape[1], rgb.shape[0]]
json.dump(meta, open(os.path.join(${JSON.stringify(CACHE)}, "meta.json"), "w"))
print("decoded", len(meta))
`;
  const r = spawnSync("python3", ["-c", py], { encoding: "utf-8" });
  if (r.status !== 0) { console.error(r.stderr); process.exit(1); }
  console.log(r.stdout.trim());
}

const meta = JSON.parse(fs.readFileSync(path.join(CACHE, "meta.json")));
const cfg = { longEdge: 640 };
let det = 0, total = 0, sumMs = 0, sumConf = 0;
const nulls = [];
for (const [name, [W, H]] of Object.entries(meta)) {
  const rgb = fs.readFileSync(path.join(CACHE, name + ".raw"));
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    data[i * 4] = rgb[i * 3]; data[i * 4 + 1] = rgb[i * 3 + 1];
    data[i * 4 + 2] = rgb[i * 3 + 2]; data[i * 4 + 3] = 255;
  }
  total++;
  const t0 = performance.now();
  const res = detectV3({ width: W, height: H, data }, cfg);
  sumMs += performance.now() - t0;
  if (res) { det++; sumConf += res.confidence; }
  else nulls.push(name);
}
const rate = (100 * det / total).toFixed(1);
console.log(`${det}/${total} detected (${rate}%), mean conf ${(sumConf / Math.max(1, det)).toFixed(2)}, mean ${(sumMs / total).toFixed(0)}ms/frame`);
if (nulls.length) console.log("nulls:", nulls.join(" "));
// Regression floor: v3 with paperness check lands ~85%. Fail the bench if it drops below 82%.
process.exit(det / total >= 0.82 ? 0 : 1);
