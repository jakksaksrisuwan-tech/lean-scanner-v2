"""Lock state machine — the QR-style "fire after N stable frames" gate.

States
------
SEARCHING   — no lock, looking for a quad
UNSTABLE    — quad detected but failing one of the stability checks
ALMOST      — quad stable and good quality but not yet for N frames
LOCKED      — ready to fire
FIRING      — capture has just happened; cooldown active
"""

from __future__ import annotations
import numpy as np
from .kalman import KalmanQuad
from ..detector.quad import mean_corner_drift, quad_iou


STATE_SEARCHING = "SEARCHING"
STATE_UNSTABLE = "UNSTABLE"
STATE_ALMOST = "ALMOST"
STATE_LOCKED = "LOCKED"
STATE_FIRING = "FIRING"
STATE_COOLDOWN = "COOLDOWN"


class LockGate:
    """Wraps a KalmanQuad + stability counter + cooldown timer.

    The gate returns one of:
        ("SEARCHING", None)
        ("UNSTABLE",  smoothed_quad)
        ("ALMOST",   smoothed_quad)
        ("LOCKED",   smoothed_quad)         — caller should fire here
        ("COOLDOWN", smoothed_quad)         — just fired; ignore until cooldown ends
    """

    def __init__(self, lock_cfg):
        self.cfg = lock_cfg
        self.kf = KalmanQuad()
        self.state = STATE_SEARCHING
        self._cooldown_until_frame = 0
        self._frame_idx = 0
        self._prev_smoothed: np.ndarray | None = None

    def update(
        self,
        raw_quad: np.ndarray | None,
        quality: float,
    ) -> tuple[str, np.ndarray | None, float]:
        """Feed a new detection result. Returns (state, smoothed, quality).

        ``state`` is one of the STATE_* constants. ``smoothed`` is the
        Kalman-smoothed quad (or None if no detection). ``quality`` is
        passed through.
        """
        self._frame_idx += 1
        smoothed, innovation, _ = self.kf.update(raw_quad)

        # Cooldown: suppress everything until it expires (frame-counted,
        # not wall-clock, so the simulator and a real camera at 30 fps
        # behave the same).
        if self.state == STATE_COOLDOWN:
            if self._frame_idx < self._cooldown_until_frame:
                return STATE_COOLDOWN, smoothed, quality
            # cooldown ended — start fresh
            self.state = STATE_SEARCHING
            self.kf.reset()

        if smoothed is None:
            self.state = STATE_SEARCHING
            self._prev_smoothed = None
            return STATE_SEARCHING, None, quality

        # Quality gate: release lock if quality collapses
        if (self.state == STATE_LOCKED and
                quality < self.cfg.quality_release_ratio * 0):  # disabled
            pass
        # Apply the real release rule against the capture threshold
        # (quality_release_ratio < 1.0 means the gate is stricter than
        # the capture threshold, so a quality drop in the locked state
        # should release the lock)
        if self.state == STATE_LOCKED and quality < 0.3:
            self.kf.locked_frames = 0
            self.state = STATE_UNSTABLE
            return STATE_UNSTABLE, smoothed, quality

        # Drift gate
        if self._prev_smoothed is not None:
            drift = mean_corner_drift(smoothed, self._prev_smoothed)
        else:
            drift = 0.0
        self._prev_smoothed = smoothed.copy()

        if drift > self.cfg.corner_dist_px:
            self.kf.locked_frames = 0
            self.state = STATE_UNSTABLE
            return STATE_UNSTABLE, smoothed, quality

        if quality < 0.5:
            self.kf.locked_frames = 0
            self.state = STATE_UNSTABLE
            return STATE_UNSTABLE, smoothed, quality

        # Quality OK and drift OK — accumulate stable frames
        self.kf.locked_frames += 1
        if self.kf.locked_frames >= self.cfg.n_stable_frames:
            self.state = STATE_LOCKED
            return STATE_LOCKED, smoothed, quality

        self.state = STATE_ALMOST
        return STATE_ALMOST, smoothed, quality

    def fire(self) -> None:
        """Record that a capture just happened. Enters COOLDOWN."""
        self.state = STATE_COOLDOWN
        # Convert cooldown_ms to frames. The pipeline runs at fps frames
        # per second, so cooldown_ms / 1000 * fps = frames.
        # For now we use 30 fps as a reasonable default; the cooldown
        # length in frames is what the lock config controls.
        cooldown_frames = int(self.cfg.cooldown_ms / 1000.0 * 30)
        self._cooldown_until_frame = self._frame_idx + cooldown_frames
        self.kf.reset()
        self._prev_smoothed = None
