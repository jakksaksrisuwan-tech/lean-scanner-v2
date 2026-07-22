// Visualize: saved-quad (yellow) vs new-quad (cyan) overlaid on the
// dump frame, side by side for the 11 frames that still detect.
import fs from 'node:fs';
import { detectHighContrast } from '/Users/jsaksrisuwan/workspace/lean_scanner_v2/src/detector/v2.js';
import { PNG } from 'pngjs';

const D = '/Users/jsaksrisuwan/workspace/lean_scanner_v2/captures-debug';
const cfg = { longEdge: 640, minQuadAreaRatio: 0.05, maxQuadAreaRatio: 0.95,
              minAspect: 0.2, maxAspect: 5.0 };
const files = fs.readdirSync(D).filter(f => /^dump_\d{8}_.*480x640\.json$/.test(f)).sort();

const keep = [];
for (const f of files) {
  const base = D + '/' + f.slice(0, -5);
  const j = JSON.parse(fs.readFileSync(base + '.json', 'utf8'));
  const rgbPath = fs.existsSync(base + '.rgb.raw') ? base + '.rgb.raw' : base + '.rgb';
  const rgb = fs.readFileSync(rgbPath);
  const W = j.w, H = j.h;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    rgba[i*4]   = rgb[i*3];
    rgba[i*4+1] = rgb[i*3+1];
    rgba[i*4+2] = rgb[i*3+2];
    rgba[i*4+3] = 255;
  }
  const det = detectHighContrast({ width: W, height: H, data: rgba }, cfg);
  if (det) keep.push({ f: f, w: W, h: H, rgb: rgb, saved: j.rawQuad, q: det.quad });
}

// Side-by-side composite: original | cyan (new) overlay
import zlib from 'node:zlib';

function pngEncode(rgba, W, H) {
  const channels = 4;
  const stride = W * channels;
  const raw = Buffer.alloc((stride + 1) * H);
  for (let y = 0; y < H; y++) {
    raw[y * (stride + 1)] = 0; // filter
    Buffer.from(rgba.buffer, rgba.byteOffset + y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }
  const idat = zlib.deflateSync(raw);
  function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const t = Buffer.from(type, 'ascii');
    const crc = Buffer.alloc(4);
    // CRC-32 over type+data
    let c = 0xffffffff;
    const tab = []; for (let n = 0; n < 256; n++) {
      let cc = n; for (let k = 0; k < 8; k++) cc = cc & 1 ? 0xedb88320 ^ (cc >>> 1) : cc >>> 1;
      tab[n] = cc;
    }
    const buf = Buffer.concat([t, data]);
    for (let i = 0; i < buf.length; i++) c = tab[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    crc.writeUInt32BE((c ^ 0xffffffff) >>> 0);
    return Buffer.concat([len, t, data, crc]);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
  ihdr.writeUInt8(8, 8); ihdr.writeUInt8(channels, 9);
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),
                        chunk('IHDR', ihdr),
                        chunk('IDAT', idat),
                        chunk('IEND', Buffer.alloc(0))]);
}

const cols = 3;
const rows = Math.ceil(keep.length / cols);
const thumb = 200;
const gap = 8;
const outW = cols * thumb + (cols - 1) * gap;
const outH = rows * thumb + (rows - 1) * gap;
const composite = new Uint8ClampedArray(outW * outH * 4);

function nearestSample(src, srcW, srcH, x, y) {
  const ix = Math.min(srcW - 1, Math.max(0, Math.floor(x)));
  const iy = Math.min(srcH - 1, Math.max(0, Math.floor(y)));
  const o = (iy * srcW + ix) * 3;
  return [src[o], src[o + 1], src[o + 2]];
}

function drawLine(canvas, W, H, x1, y1, x2, y2, r, g, b) {
  // rasterize a line into the canvas (naive bresenham)
  const dx = Math.abs(x2 - x1), dy = Math.abs(y2 - y1);
  const sx = x1 < x2 ? 1 : -1, sy = y1 < y2 ? 1 : -1;
  let err = dx - dy, x = x1, y = y1;
  while (true) {
    if (x >= 0 && x < W && y >= 0 && y < H) {
      for (let dy2 = -1; dy2 <= 1; dy2++) {
        for (let dx2 = -1; dx2 <= 1; dx2++) {
          const nx = x + dx2, ny = y + dy2;
          if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
            const idx = (ny * W + nx) * 4;
            canvas[idx]   = r;
            canvas[idx+1] = g;
            canvas[idx+2] = b;
            canvas[idx+3] = 255;
          }
        }
      }
    }
    if (x === x2 && y === y2) break;
    const e2 = 2 * err;
    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 <  dx) { err += dx; y += sy; }
  }
}

function drawQuad(canvas, W, H, q, r, g, b) {
  const pts = [[q[0], q[1]], [q[2], q[3]], [q[4], q[5]], [q[6], q[7]]];
  for (let i = 0; i < 4; i++) {
    const a = pts[i], b2 = pts[(i + 1) % 4];
    drawLine(canvas, W, H, a[0], a[1], b2[0], b2[1], r, g, b);
  }
}

for (let i = 0; i < keep.length; i++) {
  const k = keep[i];
  const r = i % cols, c = Math.floor(i / cols);
  const tx = c * (thumb + gap), ty = r * (thumb + gap);
  // Draw thumbnail
  for (let y = 0; y < thumb; y++) {
    for (let x = 0; x < thumb; x++) {
      const sx = x / thumb * k.w, sy = y / thumb * k.h;
      const [R, G, B] = nearestSample(k.rgb, k.w, k.h, sx, sy);
      const idx = ((ty + y) * outW + (tx + x)) * 4;
      composite[idx] = R; composite[idx+1] = G; composite[idx+2] = B; composite[idx+3] = 255;
    }
  }
  drawQuad(composite, outW, outH,
    [k.q[0]*thumb/k.w + tx, k.q[1]*thumb/k.h + ty,
     k.q[2]*thumb/k.w + tx, k.q[3]*thumb/k.h + ty,
     k.q[4]*thumb/k.w + tx, k.q[5]*thumb/k.h + ty,
     k.q[6]*thumb/k.w + tx, k.q[7]*thumb/k.h + ty],
    0, 255, 255);
}

fs.writeFileSync('/tmp/kept_quads.png', pngEncode(composite, outW, outH));
console.log('saved /tmp/kept_quads.png with', keep.length, 'kept detections');
