// Smoke test for src/detector/v2.js — runs without DOM (Node only).
//
// We render synthetic frames with PIL+cv2 via subprocess, decode via
// pngjs, feed the resulting ImageData shape to v2.js. Skipped if
// python deps or pngjs are missing.
//
// Run:
//   node tests/test_v2_detector.mjs

import { writeFileSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { PNG } from "pngjs";
import { detectV3 } from "../src/detector/v3.js";

const pyCheck = spawnSync("python3", ["-c", "import cv2, PIL, numpy"], { stdio: "ignore" });
if (pyCheck.status !== 0) {
  console.error("SKIP: python cv2/PIL/numpy not available.");
  process.exit(0);
}

const dir = mkdtempSync(join(tmpdir(), "v2-"));
const py = `
import sys, json, numpy as np, cv2

def synth(deg, W=640, H=480, bg=(40,30,20)):
    bg_arr = np.full((H, W, 3), bg, dtype=np.uint8)
    bg_arr = (bg_arr.astype(np.int16) + np.random.randint(-8, 8, bg_arr.shape)).clip(0,255).astype(np.uint8)
    doc_w, doc_h = int(W*0.6*0.75), int(H*0.75*0.75)
    doc = np.full((doc_h, doc_w, 3), 235, dtype=np.uint8)
    m = int(min(doc_w, doc_h)*0.08)
    cv2.rectangle(doc, (m,m), (doc_w-m, m+int(doc_h*0.06)), (0,0,0), -1)
    lines = 12
    line_h = (doc_h - 2*m - int(doc_h*0.10)) // lines
    y0 = m + int(doc_h*0.12)
    for i in range(lines):
        lw = int(doc_w*(0.5 + 0.4*(i%3)/2))
        cv2.rectangle(doc, (m, y0), (m+lw, y0+int(line_h*0.3)), (60,60,60), -1)
        y0 += line_h
    cx = W*0.55; cy = H*0.55
    hw, hh = doc_w//2, doc_h//2
    sxk = 0.03*(np.random.rand()-0.5)
    syk = 0.02*(np.random.rand()-0.5)
    src_quad = np.array([
        [cx-hw*(1+sxk), cy-hh*(1-syk)],
        [cx+hw*(1-sxk), cy-hh*(1+syk)],
        [cx+hw*(1+sxk), cy+hh*(1-syk)],
        [cx-hw*(1-sxk), cy+hh*(1+syk)],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(
        np.array([[0,0],[doc_w-1,0],[doc_w-1,doc_h-1],[0,doc_h-1]], dtype=np.float32),
        src_quad)
    warped = cv2.warpPerspective(doc, M, (W, H))
    bg_arr[warped.mean(2) > 30] = warped[warped.mean(2) > 30]
    R = cv2.getRotationMatrix2D((W/2, H/2), -deg, 1.0)
    rot = cv2.warpAffine(bg_arr, R, (W, H), borderMode=cv2.BORDER_REPLICATE)
    qh = np.concatenate([src_quad, np.ones((4, 1))], axis=1)
    gt_q = (R @ qh.T).T[:, :2]
    return rot, gt_q

out = []
for deg in [0, 10, 20, 30]:
    np.random.seed(42)
    frame, gt = synth(deg)
    path = f"${dir}/frame_{int(deg)}.png"
    cv2.imwrite(path, frame)
    out.append({"path": path, "deg": float(deg), "gt": gt.tolist()})
print(json.dumps({"dir": "${dir}", "frames": out}))
`;
const pyRes = spawnSync("python3", ["-c", py], { encoding: "utf-8" });
if (pyRes.status !== 0) {
  console.error("python synth failed:", pyRes.stderr);
  process.exit(1);
}
const pyData = JSON.parse(pyRes.stdout);
console.log(`Generated ${pyData.frames.length} frames in ${pyData.dir}\n`);

const cfg = {
  longEdge: 640,
  minQuadAreaRatio: 0.05,
  maxQuadAreaRatio: 0.98,
  minAspect: 0.2,
  maxAspect: 5.0,
};

function quadIoU(a, b) {
  function bbox(q) {
    return [Math.min(q[0], q[2], q[4], q[6]),
            Math.min(q[1], q[3], q[5], q[7]),
            Math.max(q[0], q[2], q[4], q[6]),
            Math.max(q[1], q[3], q[5], q[7])];
  }
  const [ax0, ay0, ax1, ay1] = bbox(a);
  const [bx0, by0, bx1, by1] = bbox(b);
  const ix0 = Math.max(ax0, bx0), iy0 = Math.max(ay0, by0);
  const ix1 = Math.min(ax1, bx1), iy1 = Math.min(ay1, by1);
  const iw = Math.max(0, ix1 - ix0), ih = Math.max(0, iy1 - iy0);
  const inter = iw * ih;
  const union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
  return union > 0 ? inter / union : 0;
}

function loadPngAsImageData(path) {
  const data = readFileSync(path);
  const png = PNG.sync.read(data);
  // pngjs.data is RGBA bytes
  const rgba = new Uint8ClampedArray(png.data);
  return { width: png.width, height: png.height, data: rgba };
}

let allOk = true;
for (const f of pyData.frames) {
  const id = loadPngAsImageData(f.path);
  const res = detectV3(id, cfg);
  if (!res) {
    console.log(`deg=${f.deg}: NO detection`);
    allOk = false;
    continue;
  }
  const flat = [];
  for (const p of f.gt) flat.push(...p);
  const iou = quadIoU(res.quad, flat);
  const ok = iou >= 0.7;
  if (!ok) allOk = false;
  console.log(`deg=${f.deg}: IoU=${iou.toFixed(3)}  conf=${res.confidence.toFixed(2)}  quad=[${res.quad.map(v => v.toFixed(1)).join(", ")}]  ${ok ? "PASS" : "FAIL"}`);
}

console.log(allOk ? "\nALL OK" : "\nSOME FAILED");
process.exit(allOk ? 0 : 1);
