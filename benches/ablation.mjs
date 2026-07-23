// Detector ablation: score detector snapshots against every corpus frame,
// grouped by scene, so era-vs-era questions get numbers instead of vibes.
//
// Metrics per (snapshot, frame):
//   - detected: quad returned
//   - iouGolden: IoU vs pinned golden (when one exists)
//   - inkCover: fraction of the frame's strong-gradient pixels inside the
//     quad (content completeness — a cutting quad scores low)
//   - fill: ink pixels / quad area proxy (tightness — a loose quad scores low)
//
// Run:  node benches/ablation.mjs
// Add snapshots: put a self-contained detector in benches/snapshots/ and
// register it below.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const D = path.join(HERE, "..", "captures-debug");
const GOLDEN = JSON.parse(fs.readFileSync(path.join(HERE, "..", "tests", "fixtures", "golden_quads.json")));

const SNAPSHOTS = {
  baseline: (await import("./snapshots/baseline_v3.js")).detectV3,
  current: (await import("../src/detector/v3.js")).detectV3,
};

function loadFrame(base, W, H) {
  let raw = path.join(D, base + ".rgb.raw");
  if (!fs.existsSync(raw)) raw = path.join(D, base + ".rgb");
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

function inQ(q, x, y) {
  let inside = false;
  for (let i = 0, j = 3; i < 4; j = i++) {
    const xi = q[i * 2], yi = q[i * 2 + 1], xj = q[j * 2], yj = q[j * 2 + 1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function iou(a, b, W, H) {
  let inter = 0, uni = 0;
  for (let y = 0; y < H; y += 3) for (let x = 0; x < W; x += 3) {
    const ia = inQ(a, x, y), ib = inQ(b, x, y);
    if (ia && ib) inter++;
    if (ia || ib) uni++;
  }
  return uni ? inter / uni : 0;
}

// strong-gradient mask of min channel at 1/2 res
function inkStats(frame, quad) {
  const W = frame.width, H = frame.height, d = frame.data;
  let inkIn = 0, inkTotal = 0, areaIn = 0, cells = 0;
  for (let y = 2; y < H - 2; y += 2) {
    for (let x = 2; x < W - 2; x += 2) {
      cells++;
      const mn = (i) => Math.min(d[i], d[i + 1], d[i + 2]);
      const c = (y * W + x) * 4;
      const g = Math.abs(mn(c + 8) - mn(c - 8)) + Math.abs(mn(c + W * 8) - mn(c - W * 8));
      const isInk = g > 40;
      if (isInk) inkTotal++;
      const inside = inQ(quad, x, y);
      if (inside) {
        areaIn++;
        if (isInk) inkIn++;
      }
    }
  }
  return {
    inkCover: inkTotal ? inkIn / inkTotal : 0,
    fill: areaIn ? inkIn / areaIn : 0,
  };
}

// scene = date+hour prefix, hand-labeled
const SCENES = [
  [/^dump_20260720_17/, "wood-1740"],
  [/^dump_20260720_18[01]/, "wood-18xx"],
  [/^dump_20260720_18[35]/, "wood-1833"],
  [/^dump_20260721_12/, "wall-steel"],
  [/^dump_20260721_16/, "wood-glare"],
  [/^dump_20260722_14/, "charcoal"],
  [/^dump_20260723_09/, "tile-battery"],
  [/^dump_20260723_1[01]/, "gloss-battery"],
  [/^dump_20260723_120[01]/, "white-fold"],
  [/^dump_20260723_1209/, "bedsheet"],
];
function sceneOf(name) {
  for (const [re, s] of SCENES) if (re.test(name)) return s;
  return "other";
}

const rows = {};
const jsons = fs.readdirSync(D).filter(f => /^dump_.*_\d+x\d+\.json$/.test(f)).sort();
for (const jf of jsons) {
  const meta = JSON.parse(fs.readFileSync(path.join(D, jf)));
  if (meta.w < 300 || meta.h < 300) continue;
  const base = jf.slice(0, -5);
  const frame = loadFrame(base, meta.w, meta.h);
  if (!frame) continue;
  const scene = sceneOf(base);
  rows[scene] ??= {};
  for (const [snap, detect] of Object.entries(SNAPSHOTS)) {
    let r = null;
    try { r = detect(frame, { longEdge: 640, minAspect: 0.2, maxAspect: 5.0, minQuadAreaRatio: 0.02, maxQuadAreaRatio: 0.98 }); } catch (e) { /* snapshot crash = miss */ }
    const acc = (rows[scene][snap] ??= { n: 0, det: 0, iouSum: 0, iouN: 0, cover: 0, fill: 0 });
    acc.n++;
    if (!r) continue;
    acc.det++;
    const stats = inkStats(frame, r.quad);
    acc.cover += stats.inkCover;
    acc.fill += stats.fill;
    if (GOLDEN[base]) { acc.iouSum += iou(r.quad, GOLDEN[base], meta.w, meta.h); acc.iouN++; }
  }
}

console.log("scene            snap      det     IoU⌀   cover⌀  fill⌀");
for (const [scene, snaps] of Object.entries(rows)) {
  for (const [snap, a] of Object.entries(snaps)) {
    console.log(
      scene.padEnd(16),
      snap.padEnd(9),
      `${a.det}/${a.n}`.padEnd(7),
      (a.iouN ? (a.iouSum / a.iouN).toFixed(2) : " -  ").padEnd(6),
      (a.det ? (a.cover / a.det).toFixed(2) : " -  ").padEnd(7),
      (a.det ? (a.fill / a.det).toFixed(3) : " -  "),
    );
  }
}
