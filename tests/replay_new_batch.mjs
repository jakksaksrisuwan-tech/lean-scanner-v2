import fs from 'node:fs';
import { detectHighContrast } from '/Users/jsaksrisuwan/workspace/lean_scanner_v2/src/detector/v2.js';

const D = '/Users/jsaksrisuwan/workspace/lean_scanner_v2/captures-debug';
const files = fs.readdirSync(D)
  .filter(f => f.match(/^dump_20260720_183.*480x640\.json$/))
  .sort();

const cfg = {
  longEdge: 640, minQuadAreaRatio: 0.05, maxQuadAreaRatio: 0.95,
  minAspect: 0.2, maxAspect: 5.0,
};

let det = 0;
for (const f of files) {
  const base = `${D}/${f.slice(0, -5)}`;
  const j = JSON.parse(fs.readFileSync(`${base}.json`, 'utf8'));
  let rgbPath = `${base}.rgb.raw`;
  if (!fs.existsSync(rgbPath)) rgbPath = `${base}.rgb`;
  const rgb = fs.readFileSync(rgbPath);
  const W = +j.w, H = +j.h;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    rgba[i*4]   = rgb[i*3];
    rgba[i*4+1] = rgb[i*3+1];
    rgba[i*4+2] = rgb[i*3+2];
    rgba[i*4+3] = 255;
  }
  const t0 = performance.now();
  const r = detectHighContrast({ width: W, height: H, data: rgba }, cfg);
  const ms = performance.now() - t0;
  if (r) det++;
  // Edge-touch count for new quad
  let touches = 0;
  if (r) {
    let xs = [], ys = [];
    for (let i = 0; i < 4; i++) { xs.push(r.quad[i*2]); ys.push(r.quad[i*2+1]); }
    const mn = Math.min(...xs), mx = Math.max(...xs);
    const ymn = Math.min(...ys), ymx = Math.max(...ys);
    if (mn <= 4) touches++;
    if (ymn <= 4) touches++;
    if (mx >= (W - 4)) touches++;
    if (ymx >= (H - 4)) touches++;
  }
  const saved = j.rawQuad;
  const saved_str = saved ? saved.map(v => v.toFixed(0)).join(',') : 'null';
  const new_str = r ? r.quad.map(v => v.toFixed(0)).join(',') : 'null';
  const flag = saved && r ? (saved_str !== new_str ? ' CHANGED' : ' same') :
               saved && !r ? ' now NULL ✓' :
               !saved && r ? ' now DET' : '';
  console.log(`  ${f.slice(0, 28)}  saved=${saved_str}  new=${new_str}  touches=${touches}  ${ms.toFixed(0)}ms${flag}`);
}
console.log(`\n${files.length} frames, ${det} detected`);
