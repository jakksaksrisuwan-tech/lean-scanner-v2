// Kalman filter for the 4-corner quad.
//
// Direct port of autocapture/tracker/kalman.py.
//
// State vector: [x1, y1, x2, y2, x3, y3, x4, y4]  (8-dim)
// Transition:   identity + process noise (random walk per coordinate)
// Measurement:  8 coordinates emitted by the detector
//
// Textbook linear Kalman. Pure-JS, no deps.

import { inv8x8 } from "../math/ndarray.js";

export class KalmanQuad {
  /**
   * @param {number} processNoise per-step process variance (Q)
   * @param {number} measurementNoise per-coord measurement variance (R)
   */
  constructor(processNoise = 0.02, measurementNoise = 4.0) {
    this.Q = processNoise;
    this.R = measurementNoise;
    this.x = null;     // state, Float64Array length 8
    this.P = null;     // covariance, Float64Array length 64 (8x8 row-major)
    this.lockedFrames = 0;
    this.age = 0;
  }

  /**
   * @param {number[]|null} rawQuad flat 8-array of (4,2) ordered TL TR BR BL
   * @returns {[number[]|null, number, boolean]} (smoothed, innovation, isLocked)
   */
  update(rawQuad) {
    if (rawQuad === null || rawQuad === undefined) {
      this.lockedFrames = 0;
      this.x = null;
      this.P = null;
      this.age = 0;
      return [null, 0.0, false];
    }

    const z = new Float64Array(8);
    for (let i = 0; i < 8; i++) z[i] = rawQuad[i];

    if (this.x === null) {
      // Initialize from first detection
      this.x = new Float64Array(z);
      this.P = new Float64Array(64);
      for (let i = 0; i < 8; i++) this.P[i * 8 + i] = this.R * 4;
      this.lockedFrames = 0;
      this.age = 1;
      return [rawQuad, 0.0, false];
    }

    // Snap on large innovation: when the quad genuinely moved (user
    // approaching/retreating — mean corner shift > 8% of the diagonal),
    // converging at filter speed just paints a stale quad floating off
    // the document. Re-seed from the measurement instead.
    let inno = 0;
    for (let i = 0; i < 4; i++) {
      inno += Math.hypot(z[i * 2] - this.x[i * 2], z[i * 2 + 1] - this.x[i * 2 + 1]);
    }
    inno /= 4;
    const diag = Math.hypot(z[4] - z[0], z[5] - z[1]);
    if (inno > Math.max(30, 0.08 * diag)) {
      this.x = new Float64Array(z);
      this.P = new Float64Array(64);
      for (let i = 0; i < 8; i++) this.P[i * 8 + i] = this.R * 4;
      this.lockedFrames = 0;
      this.age = 1;
      return [rawQuad, 0.0, false];
    }

    // Predict: P += I*Q (random walk on each coord)
    for (let i = 0; i < 8; i++) this.P[i * 8 + i] += this.Q;

    // Innovation y = z - x
    const y = new Float64Array(8);
    for (let i = 0; i < 8; i++) y[i] = z[i] - this.x[i];

    // S = P + I*R
    const S = new Float64Array(this.P);
    for (let i = 0; i < 8; i++) S[i * 8 + i] += this.R;

    // K = P @ inv(S)
    const SInv = inv8x8(S);
    const K = matMul(this.P, SInv, 8, 8, 8);

    // x = x + K @ y
    const Ky = matVec(K, 8, 8, y);
    for (let i = 0; i < 8; i++) this.x[i] += Ky[i];

    // P = (I - K) @ P
    const ImK = identitySub(K, 8);
    this.P = matMul(ImK, this.P, 8, 8, 8);

    this.age += 1;

    // Smoothed = x reshaped to (4, 2)
    const smoothed = new Array(8);
    for (let i = 0; i < 8; i++) smoothed[i] = this.x[i];

    // Innovation: mean L2 over the 4 corner pairs
    let inSum = 0;
    for (let i = 0; i < 4; i++) {
      const dx = y[i * 2], dy = y[i * 2 + 1];
      inSum += Math.sqrt(dx * dx + dy * dy);
    }
    return [smoothed, inSum / 4, false];  // is_locked is set by LockGate
  }

  reset() {
    this.x = null;
    this.P = null;
    this.lockedFrames = 0;
    this.age = 0;
  }
}

function matMul(A, B, m, k, n) {
  // A (m×k) @ B (k×n) -> C (m×n)
  const C = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i * k + p] * B[p * n + j];
      C[i * n + j] = s;
    }
  }
  return C;
}

function matVec(A, m, n, v) {
  // A (m×n) @ v (n) -> out (m)
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i * n + j] * v[j];
    out[i] = s;
  }
  return out;
}

function identitySub(K, n) {
  // I - K (n×n)
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out[i * n + j] = (i === j ? 1 : 0) - K[i * n + j];
    }
  }
  return out;
}