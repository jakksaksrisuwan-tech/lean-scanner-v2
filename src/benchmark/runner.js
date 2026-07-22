// Run the radial detector over a generated dataset, emit a per-frame + aggregate report.
//
// Usage:
//   import { runBenchmark } from "./runner.js";
//   const frames = generateDataset({ count: 250 });
//   const report = runBenchmark(frames);
//   console.log(report.summary());
//
// Or via benches/run.mjs (CLI).
//
// Metrics:
//   - polygon IoU per frame
//   - mean corner error per frame (in pixels)
//   - center offset per frame (in pixels)
//   - aggregate by angle (bins of 15°) and bg type

import { detectRadial } from "../detector/radial.js";
import { polygonIoU, meanCornerError, centerError } from "./scorer.js";

// IoU thresholds for a useful reported score
const IOU_GOOD = 0.7;
const IOU_OK = 0.5;
const IOU_WEAK = 0.3;

/**
 * Run benchmark. Returns a structured report.
 *
 * @param {Array<{rGBA, width, height, gtQuad, bg, theta}>} frames
 * @param {object} [cfg]  detector config
 * @returns {object}  report
 */
export function runBenchmark(frames, cfg) {
  const detectorCfg = cfg || {
    longEdge: 640,
    minQuadAreaRatio: 0.05,
    maxQuadAreaRatio: 0.98,
    minAspect: 0.2,
    maxAspect: 5.0,
  };

  const perFrame = [];
  const binsByAngle = new Map();   // key = angleDeg rounded to nearest 15
  const byBg = new Map();           // key = bg type

  let detected = 0;
  let iouSum = 0;
  let iouCount = 0;
  let iouGood = 0, iouOk = 0, iouWeak = 0, iouMiss = 0;
  let iouGoodRot = 0, iouOkRot = 0, iouWeakRot = 0, iouMissRot = 0;

  for (const frame of frames) {
    const id = { data: frame.rGBA, width: frame.width, height: frame.height };
    const t0 = performance.now();
    const result = detectRadial(id, detectorCfg);
    const elapsed = performance.now() - t0;

    let row;
    if (!result || !result.quad) {
      row = {
        index: perFrame.length,
        theta: frame.theta,
        bg: frame.bg,
        detected: false,
        iou: 0,
        cornerErr: Infinity,
        centerOff: Infinity,
        elapsedMs: elapsed,
      };
      iouMiss++;
      if (frame.theta > 0) iouMissRot++;
    } else {
      detected++;
      const iou = polygonIoU(result.quad, frame.gtQuad);
      const ce = meanCornerError(result.quad, frame.gtQuad);
      const co = centerError(result.quad, frame.gtQuad);
      row = { index: perFrame.length, theta: frame.theta, bg: frame.bg,
              detected: true, iou, cornerErr: ce, centerOff: co,
              elapsedMs: elapsed };
      iouSum += iou; iouCount++;
      if (iou >= IOU_GOOD) { iouGood++; if (frame.theta > 0) iouGoodRot++; }
      else if (iou >= IOU_OK) { iouOk++; if (frame.theta > 0) iouOkRot++; }
      else if (iou >= IOU_WEAK) { iouWeak++; if (frame.theta > 0) iouWeakRot++; }
      else { iouMiss++; if (frame.theta > 0) iouMissRot++; }
    }
    perFrame.push(row);

    // Aggregate by angle bin (every 15°)
    const angleDeg = Math.round(frame.theta * 180 / Math.PI);
    const binKey = Math.round(angleDeg / 15) * 15;
    let bin = binsByAngle.get(binKey);
    if (!bin) {
      bin = { count: 0, detected: 0, iouSum: 0, iouCount: 0 };
      binsByAngle.set(binKey, bin);
    }
    bin.count++;
    if (row.detected) {
      bin.detected++;
      bin.iouSum += row.iou;
      bin.iouCount++;
    }

    // Aggregate by bg
    let bgRow = byBg.get(frame.bg);
    if (!bgRow) {
      bgRow = { count: 0, detected: 0, iouSum: 0, iouCount: 0 };
      byBg.set(frame.bg, bgRow);
    }
    bgRow.count++;
    if (row.detected) {
      bgRow.detected++;
      bgRow.iouSum += row.iou;
      bgRow.iouCount++;
    }
  }

  function summary() {
    const total = perFrame.length;
    const meanIoU = iouCount > 0 ? iouSum / iouCount : 0;
    const detRate = total > 0 ? detected / total : 0;

    const lines = [];
    lines.push("=".repeat(60));
    lines.push("AutoCapture Benchmark Report");
    lines.push("=".repeat(60));
    lines.push("");
    lines.push(`Total frames:          ${total}`);
    lines.push(`Frames detected:       ${detected}  (${(detRate * 100).toFixed(1)}%)`);
    lines.push(`Mean IoU (when det.):  ${meanIoU.toFixed(3)}`);
    lines.push("");
    lines.push("IoU bins (all frames):");
    lines.push(`  >= 0.7  GOOD :   ${iouGood}  (${(iouGood / total * 100).toFixed(1)}%)`);
    lines.push(`  0.5..0.7  OK :   ${iouOk}  (${(iouOk / total * 100).toFixed(1)}%)`);
    lines.push(`  0.3..0.5  WEAK:  ${iouWeak}  (${(iouWeak / total * 100).toFixed(1)}%)`);
    lines.push(`  <  0.3  MISS:   ${iouMiss}  (${(iouMiss / total * 100).toFixed(1)}%)`);
    lines.push("");
    lines.push("IoU bins (rotated frames only, theta > 0):");
    const rotTotal = iouGoodRot + iouOkRot + iouWeakRot + iouMissRot;
    if (rotTotal > 0) {
      lines.push(`  >= 0.7  GOOD :   ${iouGoodRot}  (${(iouGoodRot / rotTotal * 100).toFixed(1)}%)`);
      lines.push(`  0.5..0.7  OK :   ${iouOkRot}  (${(iouOkRot / rotTotal * 100).toFixed(1)}%)`);
      lines.push(`  0.3..0.5  WEAK:  ${iouWeakRot}  (${(iouWeakRot / rotTotal * 100).toFixed(1)}%)`);
      lines.push(`  <  0.3  MISS:   ${iouMissRot}  (${(iouMissRot / rotTotal * 100).toFixed(1)}%)`);
    } else {
      lines.push("  (no rotated frames)");
    }
    lines.push("");
    lines.push("By angle bin (every 15°):");
    const sortedBins = [...binsByAngle.keys()].sort((a, b) => a - b);
    for (const k of sortedBins) {
      const b = binsByAngle.get(k);
      const detPct = b.detected / b.count * 100;
      const meanIou = b.iouCount > 0 ? (b.iouSum / b.iouCount).toFixed(2) : "—";
      lines.push(`  ${k.toString().padStart(3)}° (n=${b.count.toString().padStart(3)}): detected=${b.detected.toString().padStart(3)} (${detPct.toFixed(0).padStart(3)}%), mean IoU=${meanIou}`);
    }
    lines.push("");
    lines.push("By background:");
    for (const [bg, b] of byBg.entries()) {
      const detPct = b.detected / b.count * 100;
      const meanIou = b.iouCount > 0 ? (b.iouSum / b.iouCount).toFixed(2) : "—";
      lines.push(`  ${bg.padEnd(18)} (n=${b.count.toString().padStart(3)}): detected=${b.detected.toString().padStart(3)} (${detPct.toFixed(0).padStart(3)}%), mean IoU=${meanIou}`);
    }
    lines.push("=".repeat(60));
    return lines.join("\n");
  }

  return {
    perFrame,
    summary,
    metrics: {
      total: perFrame.length,
      detected,
      meanIoU: iouCount > 0 ? iouSum / iouCount : 0,
      iouGood, iouOk, iouWeak, iouMiss,
    },
  };
}
