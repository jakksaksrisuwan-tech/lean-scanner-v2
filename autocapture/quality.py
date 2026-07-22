"""Per-frame quality scoring (pure numpy).

Score is in [0, 1] — higher is better. The capture gate fires when
score >= ``cfg.capture_threshold`` AND the tracker has been stable for
N consecutive frames.
"""

from __future__ import annotations
import numpy as np
from .detector.quad import quad_area, quad_aspect
from .dewarp.shading import gaussian_blur


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the discrete Laplacian — a classic sharpness measure."""
    g = gray.astype(np.float64)
    p = np.pad(g, 1, mode="edge")
    lap = (
        -4 * p[1:-1, 1:-1]
        + p[:-2, 1:-1] + p[2:, 1:-1]
        + p[1:-1, :-2] + p[1:-1, 2:]
    )
    return float(lap.var())


def _sobel_mag_mean(gray: np.ndarray) -> float:
    """Mean Sobel gradient magnitude on a 2D uint8 image."""
    g = gray.astype(np.float64)
    p = np.pad(g, 1, mode="edge")
    # gx: derivative in x
    gx = (
        -p[:-2, 1:-1] + p[2:, 1:-1]
        - 2 * p[1:-1, :-2] + 2 * p[1:-1, 2:]
        - p[:-2, 2:] + p[2:, 2:]
    )
    # gy: derivative in y (same kernel, applied along rows)
    gy = (
        -p[1:-1, :-2] + p[1:-1, 2:]
        - 2 * p[:-2, 1:-1] + 2 * p[2:, 1:-1]
        - p[2:, :-2] + p[2:, 2:]
    )
    return float(np.sqrt(gx * gx + gy * gy).mean())


def _glare_ratio(gray: np.ndarray, quad: np.ndarray) -> float:
    """Fraction of pixels inside the quad that are clipped to white."""
    from .dewarp.shading import make_binary_mask
    mask = make_binary_mask(gray.shape, quad)
    if mask.sum() == 0:
        return 0.0
    over = (gray > 240) & (mask > 0)
    return float(over.sum()) / float(mask.sum())


def score(quad: np.ndarray, frame_bgr: np.ndarray, cfg) -> float:
    """Compute the per-frame quality score in [0, 1]."""
    h, w = frame_bgr.shape[:2]
    if frame_bgr.ndim == 3:
        gray = frame_bgr.mean(axis=2)
    else:
        gray = frame_bgr

    # 1. area
    frame_area = h * w
    area = quad_area(quad)
    area_ratio = area / frame_area
    if area_ratio < cfg.min_area_ratio:
        return 0.0
    area_score = min(area_ratio / cfg.target_area_ratio, 1.0)

    # 2. straightness — ratio of min to max bounding edge
    q = quad.astype(np.float64)
    edges = []
    for i in range(4):
        edges.append(np.linalg.norm(q[i] - q[(i + 1) % 4]))
    e_min, e_max = min(edges), max(edges)
    straightness = (e_min / e_max) if e_max > 0 else 0.0

    # 3. sharpness (Laplacian variance)
    lap_var = _laplacian_variance(gray)
    sharpness = min(lap_var / cfg.sharpness_full, 1.0)

    # 4. exposure (mean luma in the quad region)
    from .dewarp.shading import make_binary_mask
    mask = make_binary_mask(gray.shape, quad)
    if mask.sum() > 0:
        luma_mean = float(gray[mask > 0].mean())
    else:
        luma_mean = float(gray.mean())
    if cfg.exposure_low < luma_mean < cfg.exposure_high:
        exposure = 1.0
    elif luma_mean <= 0 or luma_mean >= 255:
        exposure = 0.0
    else:
        # smooth penalty outside the band
        if luma_mean < cfg.exposure_low:
            exposure = max(0.0, luma_mean / cfg.exposure_low)
        else:
            exposure = max(0.0, (255 - luma_mean) / (255 - cfg.exposure_high))

    # 5. blur (gradient magnitude)
    blur_mean = _sobel_mag_mean(gray)
    blur = min(blur_mean / cfg.blur_full, 1.0)

    score = (
        cfg.weight_area * area_score +
        cfg.weight_straightness * straightness +
        cfg.weight_sharpness * sharpness +
        cfg.weight_exposure * exposure +
        cfg.weight_blur * blur
    )
    return float(np.clip(score, 0.0, 1.0))
