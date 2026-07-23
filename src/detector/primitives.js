// Shared pixel-level primitives for the detector and superscan.
// Everything here is pure, allocation-per-call, and framework-free.

/**
 * Downscale an RGBA ImageData-like to `longEdge` (nearest-neighbour) and
 * emit min/max channel planes in one pass. min(R,G,B) is the "whiteness"
 * plane (paper is desaturated+bright); max-min gives saturation.
 */
export function downscaleMinMax(imageData, longEdge) {
  const W = imageData.width, H = imageData.height, d = imageData.data;
  const scale = Math.min(1.0, longEdge / Math.max(W, H));
  const nW = Math.max(1, Math.round(W * scale)), nH = Math.max(1, Math.round(H * scale));
  const mn = new Uint8Array(nW * nH), mx = new Uint8Array(nW * nH);
  for (let y = 0; y < nH; y++) {
    const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < nW; x++) {
      const si = (sy * W + Math.min(W - 1, Math.floor(x / scale))) * 4;
      const r = d[si], g = d[si + 1], b = d[si + 2];
      const i = y * nW + x;
      mn[i] = r < g ? (r < b ? r : b) : (g < b ? g : b);
      mx[i] = r > g ? (r > b ? r : b) : (g > b ? g : b);
    }
  }
  return { mn, mx, W: nW, H: nH, scale };
}

/** Otsu threshold of a Uint8Array plane. */
export function otsu(ch) {
  const hist = new Float64Array(256);
  for (let i = 0; i < ch.length; i++) hist[ch[i]]++;
  const total = ch.length;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, best = 0, thr = 127;
  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    const wF = total - wB;
    if (wF === 0) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) { best = between; thr = t; }
  }
  return thr;
}

// Separable box morphology via prefix sums — O(N) regardless of kernel
// size; identical to the naive window scan.
export function morph(mask, W, H, k, isErode) {
  const r = (k - 1) >> 1;
  const tmp = new Uint8Array(W * H);
  const preR = new Int32Array(W + 1);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) preR[x + 1] = preR[x] + mask[row + x];
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r);
      const s = preR[x1 + 1] - preR[x0];
      tmp[row + x] = isErode ? (s === x1 - x0 + 1 ? 1 : 0) : (s > 0 ? 1 : 0);
    }
  }
  const out = new Uint8Array(W * H);
  const preC = new Int32Array(H + 1);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) preC[y + 1] = preC[y] + tmp[y * W + x];
    for (let y = 0; y < H; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r);
      const s = preC[y1 + 1] - preC[y0];
      out[y * W + x] = isErode ? (s === y1 - y0 + 1 ? 1 : 0) : (s > 0 ? 1 : 0);
    }
  }
  return out;
}
export function morphOpen(mask, W, H, k) {
  return morph(morph(mask, W, H, k, true), W, H, k, false);
}

/** Windowed sum of a 0/1 mask via prefix sums (radius r box). */
export function boxSum(mask, W, H, r) {
  const pre = new Int32Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let rowAcc = 0;
    for (let x = 0; x < W; x++) {
      rowAcc += mask[y * W + x];
      pre[(y + 1) * (W + 1) + x + 1] = pre[y * (W + 1) + x + 1] + rowAcc;
    }
  }
  const out = new Int32Array(W * H);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(H - 1, y + r) + 1;
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(W - 1, x + r) + 1;
      out[y * W + x] = pre[y1 * (W + 1) + x1] - pre[y0 * (W + 1) + x1]
                     - pre[y1 * (W + 1) + x0] + pre[y0 * (W + 1) + x0];
    }
  }
  return out;
}

/** 8-connected components of a 0/1 mask, largest first (top `max`). */
export function components(mask, W, H, max = 5) {
  const label = new Int32Array(W * H);
  const stack = new Int32Array(W * H);
  const out = [];
  let next = 1;
  for (let start = 0; start < W * H; start++) {
    if (!mask[start] || label[start]) continue;
    let top = 0, area = 0;
    const pixels = [];
    stack[top++] = start;
    label[start] = next;
    while (top > 0) {
      const i = stack[--top];
      const x = i % W, y = (i / W) | 0;
      pixels.push(x, y);
      area++;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= H) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= W) continue;
          const j = ny * W + nx;
          if (mask[j] && !label[j]) { label[j] = next; stack[top++] = j; }
        }
      }
    }
    next++;
    out.push({ pixels, area });
  }
  out.sort((a, b) => b.area - a.area);
  return out.slice(0, max);
}
export function biggestComponent(mask, W, H) {
  const c = components(mask, W, H, 1);
  return c.length ? c[0] : null;
}

/** Andrew monotone chain convex hull. flat [x,y,...] → [[x,y],...] CCW. */
export function convexHull(flat) {
  const pts = [];
  for (let i = 0; i < flat.length; i += 2) pts.push([flat[i], flat[i + 1]]);
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [], upper = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}
