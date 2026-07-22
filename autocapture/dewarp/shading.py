"""Shading / illumination correction.

Flat-field correction: estimate the low-frequency illumination as a
heavily-blurred luma, then divide the original by it. This removes
vignette, hand-shadows, and uneven ambient lighting.

Pure-numpy convolution for the Gaussian blur (separable). No scipy.
"""

from __future__ import annotations
import numpy as np


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Separable Gaussian blur, 2D or 3D, ksize x ksize.

    Default sigma = ksize/6 is the standard 3-sigma-each-side convention.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"ksize must be odd >= 3, got {ksize}")
    sigma = ksize / 6.0
    k1d = _gaussian_kernel_1d(ksize, sigma)
    if img.ndim == 2:
        pad = ksize // 2
        # horizontal pass: for each output column j, sum k1d[i] * padded[j + i]
        padded = np.pad(img, ((0, 0), (pad, pad)), mode="edge")
        out_w = img.shape[1]
        tmp = np.zeros_like(img, dtype=np.float64)
        for i in range(ksize):
            tmp += k1d[i] * padded[:, i:i + out_w]
        # vertical pass
        padded = np.pad(tmp, ((pad, pad), (0, 0)), mode="edge")
        out_h = img.shape[0]
        out = np.zeros_like(tmp)
        for i in range(ksize):
            out += k1d[i] * padded[i:i + out_h, :]
        return out
    elif img.ndim == 3:
        out = np.empty_like(img, dtype=np.float64)
        for c in range(img.shape[2]):
            out[..., c] = gaussian_blur(img[..., c], ksize)
        return out
    else:
        raise ValueError(f"img must be 2D or 3D, got {img.ndim}D")


def make_binary_mask(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    """Rasterize a (4, 2) convex quad into a binary mask of the given HxW.

    Vectorized half-plane test: a convex quad is the intersection of the
    four half-planes of its edges, so a pixel is inside iff all four edge
    cross-products share a sign. Broadcast over the pixel grid — no
    per-row python loop. Boundary-inclusive: pixel centers exactly on
    an edge count as inside.
    """
    h, w = shape_hw
    q = quad.astype(np.float64).reshape(4, 2)[:, :, None, None]   # (4,2,1,1)
    e = np.roll(q, -1, axis=0) - q                                # edge vectors
    x = np.arange(w, dtype=np.float64)[None, None, :]              # (1,1,w)
    y = np.arange(h, dtype=np.float64)[None, :, None]              # (1,h,1)
    cross = e[:, 0] * (y - q[:, 1]) - e[:, 1] * (x - q[:, 0])      # (4,h,w)
    # boundary-inclusive (pixel centers exactly on an edge count as inside)
    tol = 1e-9 * np.hypot(e[:, 0], e[:, 1])
    inside = (cross >= -tol).all(axis=0) | (cross <= tol).all(axis=0)
    return inside.astype(np.uint8)


def shade_correct(
    img: np.ndarray,
    mask: np.ndarray,
    ksize: int = 201,
) -> np.ndarray:
    """Flat-field correction inside the mask region.

    Parameters
    ----------
    img  : (H, W) or (H, W, 3) ndarray, uint8 or float
    mask : (H, W) binary mask, 1 inside the document, 0 outside
    ksize : Gaussian kernel size for the illumination estimate (odd)
    """
    if img.ndim == 2:
        img = img[..., None]

    f = img.astype(np.float64)
    # Mask the gray version before blurring to avoid background contaminating illumination
    gray = f.mean(axis=2) if f.ndim == 3 else f
    gray_masked = gray * mask.astype(np.float64)
    illum = gaussian_blur(gray_masked, ksize)
    # fill background with original gray (so the output background is unchanged)
    illum = np.where(mask.astype(bool), illum, gray)
    # Avoid divide-by-zero
    illum = np.where(np.abs(illum) < 1e-3, 1e-3, illum)

    # Per-channel flat-field
    out = np.empty_like(f)
    for c in range(f.shape[2]):
        out[..., c] = (f[..., c] / illum) * 128.0  # rescale to mid-gray
    out = np.clip(out, 0, 255).astype(np.uint8)

    if img.shape[2] == 1:
        out = out[..., 0]
    return out
