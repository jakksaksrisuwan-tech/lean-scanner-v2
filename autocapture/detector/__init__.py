"""Quad detection: classical OpenCV pipeline + pure-numpy fallback."""
from .classical import detect, detect_opencv, detect_numpy
from .quad import order_corners, quad_area, quad_aspect, quad_iou, is_convex, mean_corner_drift

__all__ = [
    "detect", "detect_opencv", "detect_numpy",
    "order_corners", "quad_area", "quad_aspect", "quad_iou",
    "is_convex", "mean_corner_drift",
]
