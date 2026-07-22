"""Quad utilities — ordering, validation, area, IoU."""

from __future__ import annotations
import numpy as np


def quad_area(quad: np.ndarray) -> float:
    """Signed area of a 4-corner polygon (absolute value)."""
    q = quad.astype(np.float64).reshape(-1, 2)
    x = q[:, 0]; y = q[:, 1]
    return 0.5 * abs(
        x[0] * y[1] - x[1] * y[0] +
        x[1] * y[2] - x[2] * y[1] +
        x[2] * y[3] - x[3] * y[2] +
        x[3] * y[0] - x[0] * y[3]
    )


def order_corners(quad: np.ndarray) -> np.ndarray:
    """Reorder an arbitrary 4-corner quad to TL, TR, BR, BL (clockwise).

    Uses the sum/diff trick: TL has min (x+y), BR has max (x+y),
    TR has min (x-y), BL has max (x-y).
    """
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    s = q.sum(axis=1)
    d = np.diff(q, axis=1).ravel()
    out = np.zeros_like(q)
    out[0] = q[np.argmin(s)]
    out[2] = q[np.argmax(s)]
    out[1] = q[np.argmin(d)]
    out[3] = q[np.argmax(d)]
    return out


def is_convex(quad: np.ndarray) -> bool:
    """Quick convexity test (cross products all same sign)."""
    q = quad.astype(np.float64).reshape(4, 2)
    signs = []
    for i in range(4):
        a = q[i]
        b = q[(i + 1) % 4]
        c = q[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = np.asarray(signs)
    return (signs > 0).all() or (signs < 0).all()


def quad_aspect(quad: np.ndarray) -> float:
    """Bounding-box aspect ratio (w / h)."""
    q = quad.astype(np.float64)
    x_min, y_min = q.min(axis=0)
    x_max, y_max = q.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    return w / h if h > 0 else 0.0


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate IoU of two quads using their bounding boxes.

    True IoU on rotated quads requires Sutherland-Hodgman. For the
    tracker, axis-aligned IoU is good enough and 50x faster.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ax0, ay0 = a.min(axis=0); ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0); bx1, by1 = b.max(axis=0)
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def mean_corner_drift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Euclidean distance between corresponding corners."""
    a = a.astype(np.float64).reshape(4, 2)
    b = b.astype(np.float64).reshape(4, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())
