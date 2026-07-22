"""Contrast enhancement and adaptive thresholding (pure numpy).

Production note: for speed and quality, prefer OpenCV's
``cv2.createCLAHE`` and ``cv2.adaptiveThreshold``. The functions in this
module are the verified fallback for when OpenCV is not available —
they are correct but slower than the OpenCV equivalents.
"""

from __future__ import annotations
import numpy as np
from .shading import gaussian_blur


def histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Global histogram equalization on a 2D uint8 image.

    Fast pure-numpy implementation using ``np.cumsum``.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected 2D, got shape {gray.shape}")
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0] if (cdf > 0).any() else 0
    lut = np.floor((cdf - cdf_min) * 255.0 / max(cdf[-1] - cdf_min, 1)).astype(np.uint8)
    return lut[gray]


def clahe_simple(gray: np.ndarray, clip_limit: float = 2.0, grid: int = 8) -> np.ndarray:
    """Tile-based CLAHE on a 2D uint8 image.

    Vectorized per-tile, no Python loops over pixels. Tiles are
    non-overlapping; bilinear interpolation between tile centers is
    done with index arrays. This is fast enough to run on a 1080p
    image in well under a second.

    Parameters
    ----------
    gray : (H, W) uint8 ndarray
    clip_limit : multiple of average bin count to clip the histogram to
    grid : tiles per side
    """
    if gray.ndim != 2:
        raise ValueError(f"expected 2D, got shape {gray.shape}")
    h, w = gray.shape
    tile_h = h // grid
    tile_w = w // grid

    # Build per-tile LUT
    lut = np.zeros((grid, grid, 256), dtype=np.uint8)
    nbins = 256
    clip = int(clip_limit * (tile_h * tile_w / nbins))
    for ty in range(grid):
        for tx in range(grid):
            y0, x0 = ty * tile_h, tx * tile_w
            tile = gray[y0:y0 + tile_h, x0:x0 + tile_w]
            hist = np.bincount(tile.ravel(), minlength=nbins)
            excess = np.maximum(hist - clip, 0).sum()
            hist = np.minimum(hist, clip)
            inc = excess // nbins
            hist += inc
            cdf = hist.cumsum()
            denom = max(cdf[-1] - cdf[0], 1)
            lut[ty, tx] = ((cdf - cdf[0]) * 255 / denom).astype(np.uint8)

    # Vectorized bilinear interpolation between tiles
    yy, xx = np.indices((h, w), dtype=np.float64)
    # tile-center coordinates in pixel space
    gy = (yy + 0.5) / tile_h - 0.5
    gx = (xx + 0.5) / tile_w - 0.5
    gy0 = np.clip(np.floor(gy).astype(np.int64), 0, grid - 1)
    gy1 = np.clip(gy0 + 1, 0, grid - 1)
    gx0 = np.clip(np.floor(gx).astype(np.int64), 0, grid - 1)
    gx1 = np.clip(gx0 + 1, 0, grid - 1)
    wy = np.clip(gy - gy0, 0, 1)
    wx = np.clip(gx - gx0, 0, 1)

    v = gray.astype(np.int64)
    m00 = lut[gy0, gx0, v]
    m01 = lut[gy0, gx1, v]
    m10 = lut[gy1, gx0, v]
    m11 = lut[gy1, gx1, v]
    top = m00 * (1 - wx) + m01 * wx
    bot = m10 * (1 - wx) + m11 * wx
    out = (top * (1 - wy) + bot * wy).astype(np.uint8)
    return out


def adaptive_threshold(gray: np.ndarray, block: int = 11, c: int = 2) -> np.ndarray:
    """Gaussian adaptive threshold (binarization for the B&W scan look).

    Parameters
    ----------
    gray  : (H, W) uint8 ndarray
    block : odd kernel size (local mean window)
    c     : constant subtracted from the local mean
    """
    if gray.ndim != 2:
        raise ValueError(f"expected 2D, got shape {gray.shape}")
    if block < 3 or block % 2 == 0:
        raise ValueError(f"block must be odd >= 3, got {block}")
    local_mean = gaussian_blur(gray.astype(np.float64), block).astype(np.float64)
    return ((gray.astype(np.float64) > (local_mean - c)) * 255).astype(np.uint8)
