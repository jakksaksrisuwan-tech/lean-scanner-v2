// Tiny ndarray helper backed by typed arrays.
//
// The autocapture pipeline only needs:
//   - 2D float64 arrays for math (homography, warp, blur)
//   - element-wise ops: add, sub, mul, div, sqrt, abs
//   - reductions: sum, max, mean, var
//   - reshaping: (H,W) <-> flat
//   - meshgrid + linspace for the warp
//   - padding, clipping, slicing
//
// Pulling in a 200kb math lib would defeat the point. This file is ~250
// lines and covers everything the algorithms touch. If we ever need
// general linear algebra beyond SVD, add it to math/.
//
// Convention: row-major, shape [rows, cols], .data is a Float64Array.

export class NDArray {
  /**
   * @param {Float64Array} data
   * @param {number[]} shape  [rows, cols] for 2D
   */
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
  }

  get rows() { return this.shape[0]; }
  get cols() { return this.shape[1]; }
  get size() { return this.data.length; }

  static zeros(rows, cols) {
    return new NDArray(new Float64Array(rows * cols), [rows, cols]);
  }
  static empty(rows, cols) {
    return NDArray.zeros(rows, cols);
  }
  static fromArray(rows, cols, values) {
    const a = new Float64Array(rows * cols);
    for (let i = 0; i < values.length; i++) a[i] = values[i];
    return new NDArray(a, [rows, cols]);
  }
  static fromTyped(data2d) {
    // data2d: Float64Array of length rows*cols
    const rows = data2d.length ? Math.sqrt(data2d.length) : 0;
    return new NDArray(data2d, [rows, rows | 0]);
  }

  get(i, j) { return this.data[i * this.cols + j]; }
  set(i, j, v) { this.data[i * this.cols + j] = v; }

  /** Element-wise: this + other (scalar or same-shape). Returns new. */
  add(other) {
    const out = new Float64Array(this.size);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] + other;
    } else {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] + other.data[i];
    }
    return new NDArray(out, this.shape.slice());
  }
  sub(other) {
    const out = new Float64Array(this.size);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] - other;
    } else {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] - other.data[i];
    }
    return new NDArray(out, this.shape.slice());
  }
  /** In-place add */
  iadd(other) {
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) this.data[i] += other;
    } else {
      for (let i = 0; i < this.size; i++) this.data[i] += other.data[i];
    }
    return this;
  }

  mul(other) {
    const out = new Float64Array(this.size);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] * other;
    } else {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] * other.data[i];
    }
    return new NDArray(out, this.shape.slice());
  }
  div(other) {
    const out = new Float64Array(this.size);
    if (typeof other === "number") {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] / other;
    } else {
      for (let i = 0; i < this.size; i++) out[i] = this.data[i] / other.data[i];
    }
    return new NDArray(out, this.shape.slice());
  }
  /** Scalar assign from a 2D iterator */
  apply(fn) {
    const out = new Float64Array(this.size);
    for (let i = 0; i < this.size; i++) out[i] = fn(this.data[i], i);
    return new NDArray(out, this.shape.slice());
  }

  /** Reshape (zero-copy if sizes match) */
  reshape(newShape) {
    if (newShape[0] * newShape[1] !== this.size) {
      throw new Error(`reshape: ${this.shape} -> ${newShape} size mismatch`);
    }
    return new NDArray(this.data, newShape.slice());
  }

  transpose() {
    const r = this.rows, c = this.cols;
    const out = new Float64Array(this.size);
    for (let i = 0; i < r; i++)
      for (let j = 0; j < c; j++)
        out[j * r + i] = this.data[i * c + j];
    return new NDArray(out, [c, r]);
  }

  /** Sum of all elements */
  sum() {
    let s = 0;
    for (let i = 0; i < this.size; i++) s += this.data[i];
    return s;
  }
  max() {
    let m = -Infinity;
    for (let i = 0; i < this.size; i++) if (this.data[i] > m) m = this.data[i];
    return m;
  }
  min() {
    let m = Infinity;
    for (let i = 0; i < this.size; i++) if (this.data[i] < m) m = this.data[i];
    return m;
  }
  mean() { return this.sum() / this.size; }
  variance() {
    const m = this.mean();
    let s = 0;
    for (let i = 0; i < this.size; i++) { const d = this.data[i] - m; s += d * d; }
    return s / this.size;
  }

  /** Copy */
  clone() {
    return new NDArray(new Float64Array(this.data), this.shape.slice());
  }

  /** Pad with edge replication (mode='edge' in numpy). Returns new. */
  padEdges(padTop, padBot, padLeft, padRight) {
    const H = this.rows, W = this.cols;
    const NH = H + padTop + padBot, NW = W + padLeft + padRight;
    const out = new Float64Array(NH * NW);
    for (let i = 0; i < NH; i++) {
      const si = Math.max(0, Math.min(H - 1, i - padTop));
      for (let j = 0; j < NW; j++) {
        const sj = Math.max(0, Math.min(W - 1, j - padLeft));
        out[i * NW + j] = this.data[si * W + sj];
      }
    }
    return new NDArray(out, [NH, NW]);
  }

  /** Index a row */
  row(i) {
    const out = new Float64Array(this.cols);
    const base = i * this.cols;
    for (let j = 0; j < this.cols; j++) out[j] = this.data[base + j];
    return out;
  }

  /** Where cond > 0, take from this, else from other. cond can be a number threshold or NDArray. */
  static where(cond, a, b) {
    const out = new Float64Array(a.size);
    if (typeof cond === "number") {
      for (let i = 0; i < a.size; i++) out[i] = a.data[i] > cond ? a.data[i] : b.data[i];
    } else {
      for (let i = 0; i < a.size; i++) out[i] = cond.data[i] ? a.data[i] : b.data[i];
    }
    return out;  // raw Float64Array; caller wraps if needed
  }

  toString() {
    return `NDArray(${this.shape.join("x")})`;
  }
}

/** Build a meshgrid matching numpy.meshgrid(arange(W), arange(H)).
 * Returns {u, v} as Float64Arrays of length H*W.
 * u varies along the column axis (x), v varies along the row axis (y).
 */
export function meshgrid(W, H) {
  const u = new Float64Array(H * W);
  const v = new Float64Array(H * W);
  for (let i = 0; i < H; i++) {
    const base = i * W;
    for (let j = 0; j < W; j++) {
      u[base + j] = j;
      v[base + j] = i;
    }
  }
  return { u, v };
}

/** Like np.linspace(0, n-1, k).astype(int) — pick k evenly spaced indices in [0, n). */
export function linspaceIdx(n, k) {
  if (n === 0) return new Int32Array(0);
  if (k >= n) {
    const out = new Int32Array(n);
    for (let i = 0; i < n; i++) out[i] = i;
    return out;
  }
  const out = new Int32Array(k);
  for (let i = 0; i < k; i++) {
    out[i] = Math.round((i * (n - 1)) / (k - 1));
  }
  return out;
}

/** Linear-algebra helpers used in the algorithms. */

/** Invert a 3x3 matrix. Returns Float64Array of length 9, row-major. */
export function inv3x3(M) {
  const a = M[0], b = M[1], c = M[2];
  const d = M[3], e = M[4], f = M[5];
  const g = M[6], h = M[7], i = M[8];
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) {
    throw new Error("inv3x3: singular matrix");
  }
  const invDet = 1.0 / det;
  return new Float64Array([
    A * invDet, D * invDet, G * invDet,
    B * invDet, E * invDet, H * invDet,
    C * invDet, F * invDet, I * invDet,
  ]);
}

/** Invert an 8x8 matrix using Gauss-Jordan with partial pivoting.
 * Only used by the Kalman filter. */
export function inv8x8(M) {
  // Build augmented [M | I]
  const n = 8;
  const A = new Float64Array(n * 2 * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) A[i * 2 * n + j] = M[i * n + j];
    A[i * 2 * n + n + i] = 1;
  }
  // Forward elimination with partial pivot
  for (let k = 0; k < n; k++) {
    // Find pivot
    let maxVal = Math.abs(A[k * 2 * n + k]);
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i * 2 * n + k]);
      if (v > maxVal) { maxVal = v; maxRow = i; }
    }
    if (maxVal < 1e-12) throw new Error("inv8x8: singular");
    if (maxRow !== k) {
      for (let j = 0; j < 2 * n; j++) {
        const t = A[k * 2 * n + j];
        A[k * 2 * n + j] = A[maxRow * 2 * n + j];
        A[maxRow * 2 * n + j] = t;
      }
    }
    // Normalize pivot row
    const pv = A[k * 2 * n + k];
    for (let j = 0; j < 2 * n; j++) A[k * 2 * n + j] /= pv;
    // Eliminate column
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const f = A[i * 2 * n + k];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[i * 2 * n + j] -= f * A[k * 2 * n + j];
    }
  }
  // Extract inverse
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) out[i * n + j] = A[i * 2 * n + n + j];
  }
  return out;
}

/** Determinant of a 2x2 (used in homography disambiguation) */
export function det2x2(M) {
  return M[0] * M[3] - M[1] * M[2];
}