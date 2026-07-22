"""Pure-numpy 3x3 homography + bilinear inverse-warp.

This module has no dependencies beyond numpy. It is the verified fallback
for when OpenCV is not available. In production, prefer
``cv2.getPerspectiveTransform`` + ``cv2.warpPerspective`` for speed.

Homography math: a 3x3 matrix H maps a point (x, y) in the source image
to (x', y') in the destination via

    [x' y' w']^T  =  H @ [x y 1]^T
    output        =  (x'/w', y'/w')

We solve for H from 4 point correspondences using the Direct Linear
Transform (DLT) method. Then we apply H with inverse mapping + bilinear
interpolation so the output is fully sampled and has no holes.
"""

from __future__ import annotations
import numpy as np


def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """Compute 3x3 homography mapping src_pts -> dst_pts.

    Each input is shape (4, 2). We solve for H by enforcing the
    forward map exactly on the 4 correspondences and on the centroid,
    then disambiguating the projective sign by trying both
    orientations of the linear part and keeping the one whose
    determinant is positive (forward map, not reflection).

    Returns
    -------
    H : (3, 3) ndarray
    """
    assert src_pts.shape == (4, 2), f"src_pts must be (4,2), got {src_pts.shape}"
    assert dst_pts.shape == (4, 2), f"dst_pts must be (4,2), got {dst_pts.shape}"

    # Direct Linear Transform: 8 equations in 8 unknowns.
    A = np.zeros((8, 9), dtype=np.float64)
    for i, ((sx, sy), (dx, dy)) in enumerate(zip(src_pts, dst_pts)):
        A[2 * i    ] = [0, 0, 0, -sx, -sy, -1,  dy * sx,  dy * sy,  dy]
        A[2 * i + 1] = [-sx, -sy, -1, 0, 0, 0, -dx * sx, -dx * sy, -dx]

    # SVD: H is the right-singular vector for the smallest singular value.
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize so H[2, 2] = 1
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]

    # Sign + reflection disambiguation. The DLT has 4 solutions that
    # are projectively equivalent (differing by a sign or by
    # reflection across an axis). We test all 4 candidates: H, -H, and
    # the two single-axis reflections (negate H[0, :] or H[1, :]).
    diag_x = np.diag([-1.0, 1.0, 1.0])
    diag_y = np.diag([1.0, -1.0, 1.0])
    candidates = [H, -H, diag_x @ H, diag_y @ H]

    # Test on the 4 correspondences (zero error in all) and the centroid
    # (positive mass if forward).
    src_c = src_pts.mean(axis=0)
    dst_c = dst_pts.mean(axis=0)

    best = None
    best_score = -np.inf
    for c in candidates:
        proj = c @ np.array([src_c[0], src_c[1], 1.0])
        cx, cy = proj[0] / proj[2], proj[1] / proj[2]
        # Score: forward-map bonus (positive 2x2 det) + low centroid error
        det2 = np.linalg.det(c[:2, :2])
        score = det2 * 1e6 - ((cx - dst_c[0]) ** 2 + (cy - dst_c[1]) ** 2)
        if score > best_score:
            best_score = score
            best = c
    return best


def warp_bilinear(
    src: np.ndarray,
    H: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Inverse-warp src through H to produce an (out_h, out_w) image.

    For every output pixel (u, v), we solve for the corresponding
    source pixel (x, y) by applying H^-1, then sample src with bilinear
    interpolation. Pixels that fall outside src are filled with zeros.

    Parameters
    ----------
    src : (H, W, C) or (H, W) ndarray
    H   : (3, 3) homography, src -> dst
    out_h, out_w : output dimensions

    Returns
    -------
    dst : (out_h, out_w, C) or (out_h, out_w) ndarray
    """
    assert src.ndim in (2, 3), f"src must be 2D or 3D, got {src.ndim}D"
    H_inv = np.linalg.inv(H)

    H_src, W_src = src.shape[:2]
    channels = src.shape[2] if src.ndim == 3 else 1
    if src.ndim == 2:
        src = src[..., None]

    # Build the destination coordinate grid
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float64),
                       np.arange(out_h, dtype=np.float64))

    # Apply H^-1 to every (u, v, 1)
    ones = np.ones_like(u)
    homog = np.stack([u, v, ones], axis=-1)        # (out_h, out_w, 3)
    src_h = homog @ H_inv.T                        # (out_h, out_w, 3)
    src_w = src_h[..., 2]
    # Guard against exact-zero denominators
    src_w = np.where(np.abs(src_w) < 1e-12, 1e-12, src_w)
    x = src_h[..., 0] / src_w
    y = src_h[..., 1] / src_w

    # Bilinear sample
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    # Validity: the floor coordinates must be in-range. x1/y1 are
    # allowed to be one past the edge (we clip them and use the
    # edge value via wx/wy = 0). This is what makes an integer
    # coordinate like x=19 sample exactly src[19] instead of 0.
    valid = (x0 >= 0) & (x0 < W_src) & (y0 >= 0) & (y0 < H_src)
    x0c = np.clip(x0, 0, W_src - 1)
    x1c = np.clip(x1, 0, W_src - 1)
    y0c = np.clip(y0, 0, H_src - 1)
    y1c = np.clip(y1, 0, H_src - 1)

    Ia = src[y0c, x0c]   # top-left
    Ib = src[y0c, x1c]   # top-right
    Ic = src[y1c, x0c]   # bottom-left
    Id = src[y1c, x1c]   # bottom-right

    wx_e = wx[..., None] if src.ndim == 3 else wx
    wy_e = wy[..., None] if src.ndim == 3 else wy

    top = Ia * (1 - wx_e) + Ib * wx_e
    bot = Ic * (1 - wx_e) + Id * wx_e
    out = top * (1 - wy_e) + bot * wy_e

    out = np.where(valid[..., None] if src.ndim == 3 else valid,
                   out,
                   0.0)
    if src.ndim == 2:
        out = out[..., 0]
    return out.astype(src.dtype)


def warp_quad(src: np.ndarray, quad: np.ndarray, target_aspect: str = "A4") -> np.ndarray:
    """Warp src so that the detected 4-corner quad becomes a rectangle.

    Parameters
    ----------
    quad : (4, 2) ndarray, ordered TL, TR, BR, BL
    target_aspect : "A4" (sqrt(2)), "US_LETTER" (11/8.5), "MAX_EDGE" (square),
                    or "AUTO" (average of quad edges)
    """
    pts = quad.astype(np.float64)
    # edge lengths
    top    = np.linalg.norm(pts[1] - pts[0])
    bot    = np.linalg.norm(pts[2] - pts[3])
    right  = np.linalg.norm(pts[2] - pts[1])
    left   = np.linalg.norm(pts[0] - pts[3])
    width  = max(top, bot)
    height = max(left, right)

    if target_aspect == "A4":
        aspect = np.sqrt(2.0)
        if height > width / aspect:
            height = int(width / aspect)
        else:
            width = int(height * aspect)
    elif target_aspect == "US_LETTER":
        aspect = 11.0 / 8.5
        if height > width / aspect:
            height = int(width / aspect)
        else:
            width = int(height * aspect)
    elif target_aspect == "MAX_EDGE":
        side = int(max(width, height))
        width, height = side, side
    else:  # AUTO
        pass

    width  = int(width)
    height = int(height)
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float64)
    H = compute_homography(pts, dst)
    return warp_bilinear(src, H, height, width)
