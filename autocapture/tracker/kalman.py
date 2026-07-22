"""Kalman filter for the 4-corner quad.

State vector: [x1, y1, x2, y2, x3, y3, x4, y4]  (8 dim)
Transition:   identity + process noise  (random walk on each coordinate)
Measurement:  the 8 coordinates emitted by the detector

This is a textbook linear Kalman filter. Pure numpy, no filterpy needed.
"""

from __future__ import annotations
import numpy as np


class KalmanQuad:
    """8-state Kalman filter for a 4-corner document quad."""

    def __init__(self, process_noise: float = 0.02, measurement_noise: float = 4.0):
        self.Q = process_noise           # process variance per step
        self.R = measurement_noise       # measurement variance per coord
        self.x: np.ndarray | None = None # state
        self.P: np.ndarray | None = None # covariance
        self.locked_frames = 0
        self.age = 0

    def update(self, raw_quad: np.ndarray | None) -> tuple[np.ndarray | None, float, bool]:
        """Update the filter with a new detection.

        Returns
        -------
        smoothed : (4, 2) ndarray or None
        innovation : mean L2 distance between measurement and prediction
        is_locked : whether the filter has been stable for the lock window
        """
        if raw_quad is None:
            self.locked_frames = 0
            self.x = None
            self.P = None
            self.age = 0
            return None, 0.0, False

        z = raw_quad.astype(np.float64).reshape(8)

        if self.x is None:
            # Initialize from the first detection
            self.x = z.copy()
            self.P = np.eye(8) * self.R * 4
            self.locked_frames = 0
            self.age = 1
            return raw_quad, 0.0, False

        # Predict (random walk)
        self.P += np.eye(8) * self.Q

        # Update
        y = z - self.x                                   # innovation
        S = self.P + np.eye(8) * self.R
        K = self.P @ np.linalg.inv(S)                    # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K) @ self.P
        self.age += 1

        smoothed = self.x.reshape(4, 2)
        innovation = float(np.linalg.norm(y.reshape(4, 2), axis=1).mean())

        return smoothed, innovation, False  # is_locked is set by LockGate

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.locked_frames = 0
        self.age = 0
