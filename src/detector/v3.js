// Document detector — recursive classical pipeline, no ML, no deps.
//
//   detectV3(frame)
//     ├─ detectClassic(full frame)        confident (>=0.8)? done
//     ├─ floodRescue: Gaussian-biased basin flood localizes the doc,
//     │    detectClassic re-runs inside the padded ROI (strict mode) —
//     │    global stats (Otsu, scene gate, bg median) become local.
//     │    Arbitration: overlapping quads -> TIGHTER wins (ink conf is
//     │    inflated); disagreement -> higher confidence wins.
//     └─ refineByCrop: re-run detectClassic in a crop around the winning
//        quad; may shrink it, never grow it.
//
//   detectClassic = two scene-gated paths (measured on the phone corpus):
//     whiteness  min(R,G,B) → Otsu → open → candidate blobs scored by
//                boundary sharpness + brightness separation + center
//                prior → max-area hull quad → least-squares side refine.
//                (paper is desaturated+bright; glare fades in, paper is
//                a step edge)
//     ink        gradient density → dilate/erode → Mahalanobis-trimmed
//                hull (for text-covered docs on desaturated bright bgs
//                where whiteness has no signal)
//
// Every stage was earned by a failure in captures-debug/ and is pinned
// by tests/test_real_corpus.mjs (golden quads + negatives). Change
// behavior => run `npm test`; improve behavior => `npm run golden:update`
// after visual review. History: HANDOFF.md + research/.

import { orderCorners, quadArea } from "./quad.js";
import {
  downscaleMinMax, otsu, morph as _morph, morphOpen,
  boxSum, components, biggestComponent, convexHull,
} from "./primitives.js";

// Max-area quadrilateral inscribed in the convex hull. A receipt seen at
// an angle is a trapezoid, not a rectangle — minAreaRect circumscribes and
// pushes corners into the bg. The max-area 4-vertex subset of the hull
// lands on the true corners instead.
// ponytail: brute force O(n^4) over decimated hull (≤40 verts → ≤92k combos)
function maxAreaQuad(hull) {
  let h = hull;
  if (h.length > 40) {
    const step = Math.ceil(h.length / 40);
    h = h.filter((_, i) => i % step === 0);
  }
  const n = h.length;
  if (n < 4) return null;
  const tri = (a, b, c) =>
    Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]));
  let best = null, bestA = -1;
  for (let i = 0; i < n - 3; i++)
    for (let j = i + 1; j < n - 2; j++)
      for (let k = j + 1; k < n - 1; k++)
        for (let l = k + 1; l < n; l++) {
          const a = tri(h[i], h[j], h[k]) + tri(h[i], h[k], h[l]);
          if (a > bestA) { bestA = a; best = [h[i], h[j], h[k], h[l]]; }
        }
  return { quad: best, area: bestA / 2 };
}

// Refine each quad side with a least-squares line over nearby boundary
// points (crumpled edges make single hull vertices noisy), then intersect
// adjacent lines for the final corners.
function refineQuad(quad, boundary) {
  const lines = [];
  for (let k = 0; k < 4; k++) {
    const p0 = quad[k], p1 = quad[(k + 1) % 4];
    const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    const L = Math.hypot(dx, dy);
    if (L < 1) return null;
    const ux = dx / L, uy = dy / L;         // side direction
    const nx = -uy, ny = ux;                // side normal
    let sx = 0, sy = 0, cnt = 0;
    const sel = [];
    for (let i = 0; i < boundary.length; i += 2) {
      const rx = boundary[i] - p0[0], ry = boundary[i + 1] - p0[1];
      const t = rx * ux + ry * uy;
      if (t < 0.1 * L || t > 0.9 * L) continue;
      if (Math.abs(rx * nx + ry * ny) > 5) continue;
      sel.push(boundary[i], boundary[i + 1]);
      sx += boundary[i]; sy += boundary[i + 1]; cnt++;
    }
    if (cnt < 8) { lines.push({ nx, ny, d: nx * p0[0] + ny * p0[1] }); continue; }
    const mx = sx / cnt, my = sy / cnt;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < sel.length; i += 2) {
      const ex = sel[i] - mx, ey = sel[i + 1] - my;
      sxx += ex * ex; sxy += ex * ey; syy += ey * ey;
    }
    // principal direction of the scatter = refined side direction
    const t = (sxx + syy) / 2;
    const disc = Math.sqrt(((sxx - syy) / 2) ** 2 + sxy * sxy);
    let vx, vy;
    if (Math.abs(sxy) < 1e-9) { [vx, vy] = sxx >= syy ? [1, 0] : [0, 1]; }
    else { vx = sxy; vy = (t + disc) - sxx; const l2 = Math.hypot(vx, vy); vx /= l2; vy /= l2; }
    const rnx = -vy, rny = vx;
    lines.push({ nx: rnx, ny: rny, d: rnx * mx + rny * my });
  }
  const out = [];
  for (let k = 0; k < 4; k++) {
    const A = lines[k], B = lines[(k + 1) % 4];
    const det = A.nx * B.ny - B.nx * A.ny;
    if (Math.abs(det) < 1e-9) return null;
    out.push((A.d * B.ny - B.d * A.ny) / det, (A.nx * B.d - B.nx * A.d) / det);
  }
  return out;
}

export function detectWhiteness(imageData, cfg, strict = false) {
  const ds = downscaleMinMax(imageData, cfg.longEdge);
  const { mn: ch, mx, W, H, scale } = ds;

  const thr = otsu(ch);
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) if (ch[i] > thr) mask[i] = 1;

  const opened = morphOpen(mask, W, H, 5);
  let totMn = 0;
  for (let i = 0; i < W * H; i++) totMn += ch[i];

  // Score every sizable candidate blob and pick the most paper-like —
  // NEVER just the biggest. A sun-glare patch on polished wood is bright,
  // desaturated, and often bigger than the receipt; what tells them apart
  // is the boundary: paper is a step edge (sharp ≈ 1.0), glare fades in
  // (sharp ≈ 0.6). Measured on the glare corpus of 2026-07-21.
  let best = null;
  for (const comp of components(opened, W, H, 5)) {
    if (comp.area < 0.02 * W * H) break;      // sorted desc — rest are smaller
    if (comp.area > 0.90 * W * H) continue;   // whole-frame blob: nothing to align

    // Paperness: desaturated + clearly brighter than bg. Thresholds from
    // 35 receipt-free negative crops (34 → 0 false fires) + 35 positives.
    let fgMn = 0, fgSat = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const p = comp.pixels[i + 1] * W + comp.pixels[i];
      fgMn += ch[p];
      fgSat += mx[p] > 0 ? (mx[p] - ch[p]) / mx[p] : 0;
    }
    fgMn /= comp.area; fgSat /= comp.area;
    const bgN = W * H - comp.area;
    const bgMn = bgN > 0 ? (totMn - fgMn * comp.area) / bgN : 0;
    const sep = fgMn - bgMn;
    if (fgSat > 0.30 || sep < 50) continue;

    // Boundary pixels (pixel with an off- or out-of-frame neighbor).
    const boundary = [];
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const x = comp.pixels[i], y = comp.pixels[i + 1];
      if (x === 0 || y === 0 || x === W - 1 || y === H - 1 ||
          !opened[(y - 1) * W + x] || !opened[(y + 1) * W + x] ||
          !opened[y * W + x - 1] || !opened[y * W + x + 1]) {
        boundary.push(x, y);
      }
    }
    // Sharpness on the OUTER silhouette only (row/col extremes). The
    // boundary list above also contains interior hole edges around text
    // glyphs, which have no gradient and would dilute the measure.
    const rowMin = new Int16Array(H).fill(-1), rowMax = new Int16Array(H).fill(-1);
    const colMin = new Int16Array(W).fill(-1), colMax = new Int16Array(W).fill(-1);
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const x = comp.pixels[i], y = comp.pixels[i + 1];
      if (rowMin[y] < 0 || x < rowMin[y]) rowMin[y] = x;
      if (x > rowMax[y]) rowMax[y] = x;
      if (colMin[x] < 0 || y < colMin[x]) colMin[x] = y;
      if (y > colMax[x]) colMax[x] = y;
    }
    let sharpN = 0, sharpT = 0;
    const step = (x, y) => {
      const xl = Math.max(0, x - 3), xr = Math.min(W - 1, x + 3);
      const yu = Math.max(0, y - 3), yd = Math.min(H - 1, y + 3);
      return Math.max(Math.abs(ch[y * W + xr] - ch[y * W + xl]),
                      Math.abs(ch[yd * W + x] - ch[yu * W + x]));
    };
    for (let y = 0; y < H; y++) {
      if (rowMin[y] < 0) continue;
      sharpT += 2;
      if (step(rowMin[y], y) > 25) sharpN++;
      if (step(rowMax[y], y) > 25) sharpN++;
    }
    for (let x = 0; x < W; x++) {
      if (colMin[x] < 0) continue;
      sharpT += 2;
      if (step(x, colMin[x]) > 25) sharpN++;
      if (step(x, colMax[x]) > 25) sharpN++;
    }
    const sharp = sharpT ? sharpN / sharpT : 0;
    // Small blobs must have a crisp boundary: every false-fire blob in the
    // negatives corpus is small AND soft (sharp <= 0.72); real receipts
    // measure >= 0.75. Big blobs are exempt — SRD white-on-white receipts
    // fill the frame with soft boundaries and are still real.
    if ((strict || comp.area < 0.15 * W * H) && sharp < 0.74) continue;
    // Strict (padded-crop) contexts: the true doc is interior to the
    // crop by construction — a blob spanning nearly the whole crop is
    // the background (bright bg merged with bright paper), not the doc.
    if (strict) {
      let wx0 = W, wx1 = 0, wy0 = H, wy1 = 0;
      for (let i = 0; i < comp.pixels.length; i += 2) {
        const x = comp.pixels[i], y = comp.pixels[i + 1];
        if (x < wx0) wx0 = x; if (x > wx1) wx1 = x;
        if (y < wy0) wy0 = y; if (y > wy1) wy1 = y;
      }
      if (wx1 - wx0 >= 0.88 * W && wy1 - wy0 >= 0.88 * H) continue;
    }
    let cmx = 0, cmy = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) { cmx += comp.pixels[i]; cmy += comp.pixels[i + 1]; }
    cmx /= comp.area; cmy /= comp.area;
    const dc = Math.hypot(cmx - W / 2, cmy - H / 2) / (0.5 * Math.hypot(W, H));
    const score = (2 * sharp + Math.min(1.5, sep / 100)) * (1 - 0.4 * dc);
    if (!best || score > best.score) best = { comp, boundary, score };
  }
  if (!best) return null;
  const { comp, boundary } = best;
  const hull = convexHull(boundary);
  if (hull.length < 4) return null;
  const maq = maxAreaQuad(hull);
  if (!maq) return null;
  const refined = refineQuad(maq.quad, boundary) ??
    maq.quad.flat();

  // shoelace area of the refined quad
  let qa = 0;
  for (let k = 0; k < 4; k++) {
    const k2 = (k + 1) % 4;
    qa += refined[k * 2] * refined[k2 * 2 + 1] - refined[k2 * 2] * refined[k * 2 + 1];
  }
  qa = Math.abs(qa / 2);

  // Solidity: blob fills its quad ⇒ it's really a paper sheet,
  // not a sprawl of merged highlights.
  const solidity = comp.area / Math.max(1, qa);
  if (solidity < 0.6) return null;

  // No edge-touch filter: a receipt running past the frame edge is a valid
  // live-scanner detection (the quality scorer penalizes framing). The
  // merged-blob failure it used to guard is covered by the whole-frame and
  // solidity checks — validated on the SRD 200-receipt corpus (26% → 93%).

  const flat = refined.slice();
  if (scale < 1.0) for (let i = 0; i < 8; i++) flat[i] /= scale;
  const quad = orderCorners(flat);

  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += quad[i * 2]; cy += quad[i * 2 + 1]; }
  cx /= 4; cy /= 4;

  // 24×24 segmentation grid for UI (fraction of ON pixels per cell)
  const G = 24;
  const seg = new Float64Array(G * G);
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const y0 = Math.floor(gy / G * H), y1 = Math.min(H, Math.floor((gy + 1) / G * H));
      const x0 = Math.floor(gx / G * W), x1 = Math.min(W, Math.floor((gx + 1) / G * W));
      let t = 0, n = 0;
      for (let yy = y0; yy < y1; yy++)
        for (let xx = x0; xx < x1; xx++) { t += opened[yy * W + xx]; n++; }
      seg[gy * G + gx] = n ? t / n : 0;
    }
  }

  return { quad, confidence: Math.min(1, solidity), segmentation: seg, cx, cy };
}

// Ink-density fallback for light backgrounds (white wall, steel table):
// there the paper isn't brighter or less saturated than the bg, so the
// whiteness path has nothing to grab — but the document is packed with
// text/grid strokes while the bg is smooth. Gradient-dense blob = doc.
// Runs at 320px (7×7 morphology is too slow at 640 in JS) and ONLY on
// globally desaturated scenes, so gradient-rich wood grain never enters.
function detectInk(imageData, cfg, strict = false) {
  const ds = downscaleMinMax(imageData, 320);
  const { mn, mx, W, H, scale } = ds;
  // gradient magnitude |dx|+|dy| on the min channel
  const mag = new Float32Array(W * H);
  let magMax = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const m = Math.abs(mn[i + 1] - mn[i - 1]) + Math.abs(mn[i + W] - mn[i - W]);
      mag[i] = m;
      if (m > magMax) magMax = m;
    }
  }
  // 92nd percentile via histogram
  const bins = new Int32Array(256);
  for (let i = 0; i < mag.length; i++) bins[Math.min(255, mag[i] | 0)]++;
  let acc = 0, p92 = 30;
  for (let b = 0; b < 256; b++) { acc += bins[b]; if (acc >= 0.92 * mag.length) { p92 = b; break; } }
  const thr = Math.max(30, p92);
  const ink = new Uint8Array(W * H);
  for (let i = 0; i < mag.length; i++) if (mag[i] > thr) ink[i] = 1;

  // Density filter: text is gradient-DENSE (packed glyphs), while tile
  // grout, cables, and table edges are isolated high-gradient LINES.
  // Drop ink pixels whose 17px neighborhood is mostly empty — otherwise
  // dilation welds grout lines into the doc blob and the hull runs away
  // along the seams (20260723 battery-pack-on-tile corpus).
  // Wide window: a grout/cable NETWORK stays sparse at 33px scale while a
  // document region (glyphs + rules mixed) stays moderately dense. Small
  // windows can't tell a grout line from a form's grid line — both are
  // thin — so the context radius is what separates them.
  const dens = boxSum(ink, W, H, 16);
  const area33 = 33 * 33;
  for (let i = 0; i < ink.length; i++) {
    if (ink[i] && dens[i] < 0.09 * area33) ink[i] = 0;
  }

  const big = _morph(_morph(ink, W, H, 7, false), W, H, 7, false);   // dilate x2
  const core = _morph(_morph(big, W, H, 7, true), W, H, 7, true);    // erode x2
  // Candidate selection with a soft center prior: the user aims the
  // subject roughly at the frame center; a same-sized blob at the edge
  // (tile seams, table edge) loses to a centered one, but a decisively
  // bigger off-center blob still wins (prior, not constraint).
  let comp = null, bestScore = -1;
  for (const c of components(core, W, H, 5)) {
    if (c.area < 0.02 * W * H) break;
    if (c.area > 0.90 * W * H) continue;
    let mx = 0, my = 0;
    for (let i = 0; i < c.pixels.length; i += 2) { mx += c.pixels[i]; my += c.pixels[i + 1]; }
    mx /= c.area; my /= c.area;
    const dc = Math.hypot(mx - W / 2, my - H / 2) / (0.5 * Math.hypot(W, H));
    const score = c.area * (1 - 0.7 * dc);
    if (score > bestScore) { bestScore = score; comp = c; }
  }
  if (!comp) return null;
  // Mahalanobis trim: a grout/cable tail welded to the doc blob is a
  // small pixel mass far from the blob's center — clip >2.5 sigma before
  // the hull so maxAreaQuad can't pick tail points as corners.
  // Iterated: a heavy tail inflates the covariance on pass 1, hiding
  // itself; re-estimating on the trimmed set converges in 2-3 passes.
  for (let pass = 0; pass < 3; pass++) {
    let mx = 0, my = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) { mx += comp.pixels[i]; my += comp.pixels[i + 1]; }
    mx /= comp.area; my /= comp.area;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const ex = comp.pixels[i] - mx, ey = comp.pixels[i + 1] - my;
      sxx += ex * ex; sxy += ex * ey; syy += ey * ey;
    }
    sxx /= comp.area; sxy /= comp.area; syy /= comp.area;
    const det = sxx * syy - sxy * sxy;
    if (det <= 1e-6) break;
    const i11 = syy / det, i12 = -sxy / det, i22 = sxx / det;
    const kept = [];
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const ex = comp.pixels[i] - mx, ey = comp.pixels[i + 1] - my;
      const d2 = ex * (i11 * ex + i12 * ey) + ey * (i12 * ex + i22 * ey);
      if (d2 <= 2.5 * 2.5) kept.push(comp.pixels[i], comp.pixels[i + 1]);
    }
    if (kept.length < 16 || kept.length === comp.pixels.length) break;
    comp = { pixels: kept, area: kept.length / 2 };
  }

  // hull over ORIGINAL ink pixels inside the core component (undilated
  // points keep steel-scratch bloat out of the corners)
  const sel = [];
  for (let i = 0; i < comp.pixels.length; i += 2) {
    const x = comp.pixels[i], y = comp.pixels[i + 1];
    if (ink[y * W + x]) sel.push(x, y);
  }
  const hull = convexHull(sel.length >= 8 ? sel : comp.pixels);
  if (hull.length < 4) return null;
  const maq = maxAreaQuad(hull);
  if (!maq) return null;
  const quad = maq.quad;
  // doc-ness: the dilated ink must fill most of the quad
  const solidity = comp.area / Math.max(1, maq.area);
  if (solidity < 0.45) return null;

  // additive margin: erosion radius (7*2/2 per side ≈ 7px core loss after
  // symmetric open, measured) + paper border beyond the outermost strokes.
  // Deliberately loose — the second-pass crop refinement in detectV3
  // tightens it against the local boundary (see refineByCrop).
  // In strict (rescue/refine crop) contexts the hull comes from clean
  // local statistics — the full-frame margin guess mostly re-adds bg
  // ring there (loose battery-pack quads). Keep it only as erosion
  // compensation.
  const MARGIN = strict ? 8 : 22;
  let cx = 0, cy = 0;
  for (const [x, y] of quad) { cx += x; cy += y; }
  cx /= 4; cy /= 4;
  const fw = imageData.width, fh = imageData.height;
  const flat = [];
  for (const [x, y] of quad) {
    const vx = x - cx, vy = y - cy;
    const L = Math.max(1, Math.hypot(vx, vy));
    flat.push(
      Math.min(fw - 1, Math.max(0, (cx + vx * (1 + MARGIN / L)) / scale)),
      Math.min(fh - 1, Math.max(0, (cy + vy * (1 + MARGIN / L)) / scale)));
  }
  const ordered = orderCorners(flat);
  let ocx = 0, ocy = 0;
  for (let i = 0; i < 4; i++) { ocx += ordered[i * 2]; ocy += ordered[i * 2 + 1]; }
  return { quad: ordered, confidence: Math.min(1, solidity), segmentation: null, cx: ocx / 4, cy: ocy / 4 };
}

// ── Flood localizer ─────────────────────────────────────────────────────
// Gaussian-biased basin flood at 160px: documentness evidence (whiteness
// OR ink density) weighted by a centered Gaussian prior forms a "dip";
// flood from the deepest point by descending energy; the waterline is the
// flood size whose puddle is most convex-rectangular (overflood grows
// tentacles, killing the score; floods hugging 2+ frame borders are
// rejected). Returns a padded full-res ROI, or null.
//
// The ROI's purpose is to turn the classic detector's GLOBAL statistics
// (Otsu, scene gate, bg median) into LOCAL ones — inside the crop the bg
// is just the document's immediate surroundings, so thresholds are
// bimodal by construction. See research/flood-detector-experiment.md.
function floodLocalize(imageData) {
  const ds = downscaleMinMax(imageData, 160);
  const { mn, mx, W, H, scale } = ds;
  const n = W * H;

  // evidence = max(whiteness, ink-density)
  const mag = new Float32Array(n);
  const bins = new Int32Array(256);
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const m = Math.abs(mn[i + 1] - mn[i - 1]) + Math.abs(mn[i + W] - mn[i - W]);
      mag[i] = m;
      bins[Math.min(255, m | 0)]++;
    }
  }
  let acc = 0, p92 = 30;
  for (let b = 0; b < 256; b++) { acc += bins[b]; if (acc >= 0.92 * n) { p92 = b; break; } }
  const thr = Math.max(30, p92);
  const ink = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (mag[i] > thr) ink[i] = 1;
  const dens = boxSum(ink, W, H, 6);
  const densCap = 0.30 * 13 * 13;
  const cx = W / 2, cy = H / 2;
  const sig2 = (0.45 * Math.min(W, H)) ** 2;
  const level = new Uint8Array(n); // quantized energy 0..255
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const satP = mx[i] > 0 ? (mx[i] - mn[i]) / mx[i] : 0;
      const wn = (mn[i] / 255) * (1 - satP);
      const di = Math.min(1, dens[i] / densCap);
      const ev = Math.max(wn, 0.9 * di);
      const g = Math.exp(-0.5 * ((x - cx) * (x - cx) + (y - cy) * (y - cy)) / sig2);
      level[i] = Math.min(255, Math.round(255 * ev * (0.35 + 0.65 * g)));
    }
  }

  // bucket-queue flood from the global max, popping highest-energy frontier
  let seed = 0;
  for (let i = 0; i < n; i++) if (level[i] > level[seed]) seed = i;
  const buckets = Array.from({ length: 256 }, () => []);
  const visited = new Uint8Array(n);
  buckets[level[seed]].push(seed);
  visited[seed] = 1;
  let top = level[seed];
  const order = new Int32Array(n);
  let m = 0;
  while (m < n) {
    while (top > 0 && buckets[top].length === 0) top--;
    if (buckets[top].length === 0) break;
    const i = buckets[top].pop();
    order[m++] = i;
    const x = i % W, y = (i / W) | 0;
    for (let k = 0; k < 4; k++) {
      const nx = x + (k === 0 ? 1 : k === 1 ? -1 : 0);
      const ny = y + (k === 2 ? 1 : k === 3 ? -1 : 0);
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
      const j = ny * W + nx;
      if (!visited[j]) {
        visited[j] = 1;
        buckets[level[j]].push(j);
        if (level[j] > top) top = level[j];
      }
    }
  }

  // waterline sweep: most convex-compact puddle wins
  const minA = (0.03 * n) | 0, maxA = Math.min(m, (0.55 * n) | 0);
  const mask = new Uint8Array(n);
  let idx = 0, bestScore = -1, bestBox = null;
  for (let c = 0; c < 50; c++) {
    const target = (minA + (maxA - minA) * (c / 49)) | 0;
    while (idx < target) mask[order[idx++]] = 1;
    const comp = biggestComponent(mask, W, H);
    if (!comp || comp.area < minA) continue;
    let x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (let i = 0; i < comp.pixels.length; i += 2) {
      const x = comp.pixels[i], y = comp.pixels[i + 1];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    const borders = (x0 <= 1 ? 1 : 0) + (y0 <= 1 ? 1 : 0) + (x1 >= W - 2 ? 1 : 0) + (y1 >= H - 2 ? 1 : 0);
    if (borders >= 2) continue;
    const hull = convexHull(comp.pixels);
    if (hull.length < 3) continue;
    let ha = 0;
    for (let i = 0; i < hull.length; i++) {
      const [ax, ay] = hull[i], [bx, by] = hull[(i + 1) % hull.length];
      ha += ax * by - bx * ay;
    }
    ha = Math.abs(ha / 2);
    const score = comp.area / Math.max(1, ha);
    if (score > bestScore) { bestScore = score; bestBox = { x0, y0, x1, y1 }; }
  }
  if (!bestBox) return null;

  // padded full-res ROI (25% pad so the doc stays well under the classic
  // path's 90%-of-frame cap and local bg stats have material)
  const bw = bestBox.x1 - bestBox.x0, bh = bestBox.y1 - bestBox.y0;
  const px = Math.max(6, 0.25 * bw), py = Math.max(6, 0.25 * bh);
  const fx0 = Math.max(0, Math.round((bestBox.x0 - px) / scale));
  const fy0 = Math.max(0, Math.round((bestBox.y0 - py) / scale));
  const fx1 = Math.min(imageData.width, Math.round((bestBox.x1 + px) / scale));
  const fy1 = Math.min(imageData.height, Math.round((bestBox.y1 + py) / scale));
  if (fx1 - fx0 < 32 || fy1 - fy0 < 32) return null;
  return { x0: fx0, y0: fy0, w: fx1 - fx0, h: fy1 - fy0 };
}

function cropImage(imageData, roi) {
  const out = new Uint8ClampedArray(roi.w * roi.h * 4);
  const src = imageData.data, sw = imageData.width;
  for (let y = 0; y < roi.h; y++) {
    const srcOff = ((y + roi.y0) * sw + roi.x0) * 4;
    out.set(src.subarray(srcOff, srcOff + roi.w * 4), y * roi.w * 4);
  }
  return { width: roi.w, height: roi.h, data: out };
}

function detectClassic(imageData, cfg, strict = false) {
  // Scene gate: on globally desaturated scenes (steel table, white wall)
  // the whiteness signal is meaningless — paper is no brighter or whiter
  // than the bg — so the ink-density path leads and whiteness is the
  // fallback. On colored scenes (wood) whiteness leads and ink NEVER runs,
  // because gradient-rich grain would false-fire it.
  const ds = downscaleMinMax(imageData, 64);
  let sat = 0;
  for (let i = 0; i < ds.mn.length; i++) sat += ds.mx[i] > 0 ? (ds.mx[i] - ds.mn[i]) / ds.mx[i] : 0;
  // Two-axis scene gate (measured on the phone corpus):
  //   saturated bg (wood, sat>0.30)              -> whiteness only
  //   desaturated + BRIGHT bg (white wall/steel,
  //     median-mn>95: paper no brighter than bg) -> ink first
  //   desaturated + DARK bg (charcoal, median<95:
  //     whiteness signal is huge and stable)     -> whiteness first
  // Ink-first on dark scenes caused 180px corner jitter (20260722_142x):
  // text gradients merged with table scratches into runaway hulls.
  const desaturated = sat / ds.mn.length <= 0.30;
  const sorted = Array.from(ds.mn).sort((a, b) => a - b);
  const bgBright = sorted[sorted.length >> 1] > 95;
  if (desaturated && bgBright) {
    return detectInk(imageData, cfg, strict) ?? detectWhiteness(imageData, cfg, strict);
  }
  return detectWhiteness(imageData, cfg, strict) ?? (desaturated ? detectInk(imageData, cfg, strict) : null);
}

export function detectV3(imageData, cfg) {
  // Classic full-frame first — every pinned golden and the 2/35 negative
  // record were earned by this path. A CONFIDENT classic result returns
  // verbatim; a weak one (runaway hulls score low solidity) must beat
  // the flood-rescue in confidence to survive.
  const classic = detectClassic(imageData, cfg);
  let out;
  if (classic && classic.confidence >= 0.8) out = classic;
  else {
    const rescued = floodRescue(imageData, cfg);
    // Same doc (quads overlap) -> the TIGHTER quad wins: the ink path's
    // confidence is structurally inflated (solidity of dilated ink ~1.0),
    // so confidence only referees when the two disagree on location.
    out = classic && rescued
      ? (quadOverlap(classic.quad, rescued.quad) >= 0.5
          ? preferTighter(imageData, classic, rescued)
          : (rescued.confidence > classic.confidence ? rescued : classic))
      : (classic ?? rescued);
  }
  if (!out) return null;
  return refineByCrop(imageData, cfg, out) ?? out;
}

// Tighter-wins with an ink-preservation guard: a smaller quad is only
// "tighter" if it still contains the document's ink — the ink path's
// text-block hull is smaller than the paper but CLIPS logos and margins
// (20260723_11xx receipt corpus: quads cut the header logo). Dropping
// >3% of the looser quad's ink content disqualifies the tighter one.
function preferTighter(imageData, a, b) {
  const [tight, loose] = quadArea(a.quad) <= quadArea(b.quad) ? [a, b] : [b, a];
  if (inkInQuad(imageData, tight.quad) >= 0.97 * inkInQuad(imageData, loose.quad)) return tight;
  return loose;
}

// Count of strong-gradient (ink/text) pixels inside a quad, at 160px.
function inkInQuad(imageData, quad) {
  const ds = downscaleMinMax(imageData, 160);
  const { mn, W, H, scale } = ds;
  const q = quad.map((v) => v * scale);
  let count = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      const g = Math.abs(mn[i + 1] - mn[i - 1]) + Math.abs(mn[i + W] - mn[i - W]);
      if (g <= 40) continue;
      let inside = false;
      for (let a2 = 0, b2 = 3; a2 < 4; b2 = a2++) {
        const xi = q[a2 * 2], yi = q[a2 * 2 + 1], xj = q[b2 * 2], yj = q[b2 * 2 + 1];
        if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (inside) count++;
    }
  }
  return count;
}

// Sampled IoU of two quads over their joint bbox.
function quadOverlap(a, b) {
  const inQ = (q, x, y) => {
    let inside = false;
    for (let i = 0, j = 3; i < 4; j = i++) {
      const xi = q[i * 2], yi = q[i * 2 + 1], xj = q[j * 2], yj = q[j * 2 + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  };
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const q of [a, b]) {
    for (let i = 0; i < 4; i++) {
      x0 = Math.min(x0, q[i * 2]); x1 = Math.max(x1, q[i * 2]);
      y0 = Math.min(y0, q[i * 2 + 1]); y1 = Math.max(y1, q[i * 2 + 1]);
    }
  }
  const step = Math.max(2, ((x1 - x0) + (y1 - y0)) / 80 | 0);
  let inter = 0, uni = 0;
  for (let y = y0; y <= y1; y += step) {
    for (let x = x0; x <= x1; x += step) {
      const ia = inQ(a, x, y), ib = inQ(b, x, y);
      if (ia && ib) inter++;
      if (ia || ib) uni++;
    }
  }
  return uni ? inter / uni : 0;
}

// Second-pass crop refinement: re-run the detector on a crop around the
// first quad. Refine-by-recursion, not by constraint — a loose first
// quad (e.g. the ink path's +22px margin) is a fine localizer; inside
// the crop the boundary is locally unambiguous and the same proven code
// tightens it. Accepted only when it clearly matches the first result
// (IoU >= 0.6) with confidence at least as good.
function refineByCrop(imageData, cfg, first) {
  const q = first.quad;
  let bx0 = Infinity, by0 = Infinity, bx1 = -Infinity, by1 = -Infinity;
  for (let i = 0; i < 4; i++) {
    bx0 = Math.min(bx0, q[i * 2]); bx1 = Math.max(bx1, q[i * 2]);
    by0 = Math.min(by0, q[i * 2 + 1]); by1 = Math.max(by1, q[i * 2 + 1]);
  }
  const pad = 0.15 * Math.max(bx1 - bx0, by1 - by0);
  const x0 = Math.max(0, Math.round(bx0 - pad)), y0 = Math.max(0, Math.round(by0 - pad));
  const x1 = Math.min(imageData.width, Math.round(bx1 + pad));
  const y1 = Math.min(imageData.height, Math.round(by1 + pad));
  const w = x1 - x0, h = y1 - y0;
  if (w < 48 || h < 48) return null;
  if (w >= imageData.width * 0.95 && h >= imageData.height * 0.95) return null;
  const r = detectClassic(cropImage(imageData, { x0, y0, w, h }), cfg, true);
  // Refinement may tighten (shrink) even at slightly lower confidence,
  // but must never grow the quad — growth is the loose-ring failure mode.
  if (!r || r.confidence < 0.9 * first.confidence) return null;
  const rGlobal = { quad: r.quad.map((v, i) => v + (i % 2 === 0 ? x0 : y0)) };
  if (inkInQuad(imageData, rGlobal.quad) < 0.97 * inkInQuad(imageData, first.quad)) return null;
  for (let i = 0; i < 8; i += 2) { r.quad[i] += x0; r.quad[i + 1] += y0; }
  // same object? sampled quad IoU against the first pass
  let inter = 0, uni = 0;
  const inQ = (qq, x, y) => {
    let inside = false;
    for (let i = 0, j = 3; i < 4; j = i++) {
      const xi = qq[i * 2], yi = qq[i * 2 + 1], xj = qq[j * 2], yj = qq[j * 2 + 1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  };
  for (let yy = y0; yy < y1; yy += 4) {
    for (let xx = x0; xx < x1; xx += 4) {
      const a = inQ(q, xx, yy), b = inQ(r.quad, xx, yy);
      if (a && b) inter++;
      if (a || b) uni++;
    }
  }
  if (!uni || inter / uni < 0.6) return null;
  if (quadArea(r.quad) > 1.02 * quadArea(q)) return null;
  r.cx += 0; r.cy += 0;
  let cx = 0, cy = 0;
  for (let i = 0; i < 4; i++) { cx += r.quad[i * 2]; cy += r.quad[i * 2 + 1]; }
  r.cx = cx / 4; r.cy = cy / 4;
  r.segmentation = null;
  return r;
}

function floodRescue(imageData, cfg) {
  // Flood RESCUE: when the classic pass sees nothing, localize the
  // document basin (Gaussian-biased flood) and rerun the classic
  // detector inside the padded ROI, where its global statistics become
  // local (bimodal by construction). A confidence bar keeps the local
  // gates — more permissive on garbage — from re-admitting empty scenes.
  const roi = floodLocalize(imageData);
  if (!roi || (roi.w >= imageData.width && roi.h >= imageData.height)) return null;
  const crop = cropImage(imageData, roi);
  // strict: the sharpness floor applies to ALL blob sizes in a rescue
  // crop — zooming makes any object big enough to dodge the small-blob
  // floor, and soft-boundary gloss highlights (leather bag, measured
  // sharp~0.5 at conf 0.88) are exactly what the floor exists to reject.
  const r = detectClassic(crop, cfg, true);
  if (!r || r.confidence < 0.75) return null;
  // The ROI is padded 25% so a true document sits interior to the crop;
  // a quad touching the crop edge is a clipped partial structure (zipper
  // band, table edge) that leaked through the local gates.
  for (let i = 0; i < 4; i++) {
    if (r.quad[i * 2] < 3 || r.quad[i * 2] > crop.width - 3 ||
        r.quad[i * 2 + 1] < 3 || r.quad[i * 2 + 1] > crop.height - 3) return null;
  }
  for (let i = 0; i < 8; i += 2) { r.quad[i] += roi.x0; r.quad[i + 1] += roi.y0; }
  r.cx += roi.x0; r.cy += roi.y0;
  r.segmentation = null; // grid was crop-relative; UI treats null as "skip"
  return r;
}
