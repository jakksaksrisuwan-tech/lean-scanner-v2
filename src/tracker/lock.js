// Lock state machine — the QR-style "fire after N stable frames" gate.
//
// Direct port of autocapture/tracker/lock.py.
//
// States:
//   SEARCHING  — no lock, looking for a quad
//   UNSTABLE   — quad detected but failing a stability check
//   ALMOST     — stable and good quality but not yet for N frames
//   LOCKED     — ready to fire
//   COOLDOWN   — just fired; suppress until cooldown expires

import { KalmanQuad } from "./kalman.js";
import { meanCornerDrift, quadIou } from "../detector/quad.js";

export const STATE_SEARCHING = "SEARCHING";
export const STATE_UNSTABLE = "UNSTABLE";
export const STATE_ALMOST = "ALMOST";
export const STATE_LOCKED = "LOCKED";
export const STATE_COOLDOWN = "COOLDOWN";

export class LockGate {
  constructor(lockCfg) {
    this.cfg = lockCfg;
    this.kf = new KalmanQuad();
    this.state = STATE_SEARCHING;
    this._cooldownUntilFrame = 0;
    this._frameIdx = 0;
    this._prevSmoothed = null;
  }

  /**
   * Feed a new detection result.
   * @param {number[]|null} rawQuad
   * @param {number} quality
   * @returns {[string, number[]|null, number]} (state, smoothed, quality)
   */
  update(rawQuad, quality) {
    this._frameIdx += 1;
    const [smoothed] = this.kf.update(rawQuad);

    // Cooldown: suppress until it expires
    if (this.state === STATE_COOLDOWN) {
      if (this._frameIdx < this._cooldownUntilFrame) {
        return [STATE_COOLDOWN, smoothed, quality];
      }
      // cooldown ended — start fresh
      this.state = STATE_SEARCHING;
      this.kf.reset();
    }

    if (smoothed === null) {
      this.state = STATE_SEARCHING;
      this._prevSmoothed = null;
      return [STATE_SEARCHING, null, quality];
    }

    // Quality release rule (matches the python implementation's behavior)
    if (this.state === STATE_LOCKED && quality < 0.3) {
      this.kf.lockedFrames = 0;
      this.state = STATE_UNSTABLE;
      return [STATE_UNSTABLE, smoothed, quality];
    }

    // Drift gate — tolerance is RELATIVE to quad size. At close range the
    // quad spans most of the frame and a millimetre of hand tremor moves
    // corners >20px; a fixed threshold resets the gate every few frames
    // (smoothed quad vanishes, lock never engages). 4% of the diagonal
    // ≈ the same hand steadiness at any distance; floor at cornerDistPx.
    let drift = 0;
    if (this._prevSmoothed !== null) {
      drift = meanCornerDrift(smoothed, this._prevSmoothed);
    }
    this._prevSmoothed = Array.from(smoothed);

    const diag = Math.hypot(smoothed[4] - smoothed[0], smoothed[5] - smoothed[1]);
    const driftTol = Math.max(this.cfg.cornerDistPx, 0.04 * diag);
    if (drift > driftTol) {
      this.kf.lockedFrames = 0;
      this.state = STATE_UNSTABLE;
      return [STATE_UNSTABLE, smoothed, quality];
    }

    if (quality < 0.5) {
      this.kf.lockedFrames = 0;
      this.state = STATE_UNSTABLE;
      return [STATE_UNSTABLE, smoothed, quality];
    }

    // OK — accumulate stable frames
    this.kf.lockedFrames += 1;
    if (this.kf.lockedFrames >= this.cfg.nStableFrames) {
      this.state = STATE_LOCKED;
      return [STATE_LOCKED, smoothed, quality];
    }

    this.state = STATE_ALMOST;
    return [STATE_ALMOST, smoothed, quality];
  }

  /** Record that a capture just happened. Enters COOLDOWN. */
  fire() {
    this.state = STATE_COOLDOWN;
    // Convert cooldown_ms to frames (assume 30 fps; matches python default)
    const cooldownFrames = Math.floor(this.cfg.cooldownMs / 1000.0 * 30);
    this._cooldownUntilFrame = this._frameIdx + cooldownFrames;
    this.kf.reset();
    this._prevSmoothed = null;
  }
}