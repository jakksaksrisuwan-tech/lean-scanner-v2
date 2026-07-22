// Replay v2.js detector on saved dump frames, including the new edge-touching filter.
// Same as v2_replay.mjs but with the latest v2.js loaded.
import fs from 'node:fs';
import { PNG } from 'pngjs';
import { detectHighContrast } from '/Users/jsaksrisuwan/workspace/lean_scanner_v2/src/detector/v2.js';

const D = '/Users/jsaksrisuwan/workspace/lean_scanner_v2/captures-debug';
const files = fs.readdirSync(D)
  .filter(f => f.match(/^dump_\d{8}_.*480x640\.json$/))
  .sort();

const cfg = {
  longEdge: 640, minQuadAreaRatio: 0.05, maxQuadAreaRatio: 0.95,
  minAspect: 0.2, maxAspect: 5.0,
};

let detected = 0, total = 0;
const report = [];
for (const f of files) {
  const jsonPath = `${D}/${f}`;
  const base = jsonPath.slice(0, -5);
  const j = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  // Use the rgb.raw (or legacy .rgb) as we already know the layout
  let rgbPath = `${base}.rgb.raw`;
  if (!fs.existsSync(rgbPath)) rgbPath = `${base}.rgb`;
  const rgb = fs.readFileSync(rgbPath);
  const W = j.w, H = j.h;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    rgba[i*4]   = rgb[i*3];
    rgba[i*4+1] = rgb[i*3+1];
    rgba[i*4+2] = rgb[i*3+2];
    rgba[i*4+3] = 255;
  }
  const t0 = performance.now();
  const det = detectHighContrast({ width: W, height: H, data: rgba }, cfg);
  const dtMs = performance.now() - t0;
  total++;
  if (det) detected++;
  report.push({ f, W: +j.w, H: j.h, saved_q: j.rawQuad, new_det: det ? det.quad : null, ms: dtMs.toFixed(1) });
}

console.log(`detection: ${detected} / ${total}  (${(100*detected/total).toFixed(0)}%)`);
console.log('');
for (const r of report) {
  const savedQ = r.saved_q;
  const saved = savedQ ? savedQ.map(v => v.toFixed(0)).join(',') : 'null';
  const newQ = r.new_det ? r.new_det.map(v => v.toFixed(0)).join(',') : 'null';
  // Detect edge-touching on the new quad
  let touches = 0;
  if (r.new_det) {
    const w = r.W, h = r.H;
    let xs = [], ys = [];
    for (let i = 0; i < 4; i++) {
      xs.push(r.new_det[i*2]); ys.push(r.new_det[i*2+1]);
    }
    const mn = Math.min(...xs), mx = Math.max(...xs);
    const ymn = Math.min(...ys), ymx = Math.max(...ys);
    if (mn <= 4) touches++;
    if (ymn <= 4) touches++;
    if (mx >= (w - 4)) touches++;
    if (ymx >= (h - 4)) touches++;
  }
  const flag = (savedQ && r.new_det) ? (saved !== newQ ? '  CHANGED' : '  same') :
                 (savedQ && !r.new_det) ? '  now NULL' :
                 (!savedQ && r.new_det) ? '  now DET' : '';
  console.log(
    `  ${r.f}\n` +
    `    saved_q=${saved}\n` +
    `    new_q  =${newQ}  touches=${touches}  ${r.ms}ms${flag}`
  );
}
