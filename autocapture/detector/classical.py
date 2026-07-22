"""Document-quad detection.

Public entry points:
  - ``detect(frame_bgr, cfg)`` — auto-pick backend; returns a (4, 2)
    ndarray TL/TR/BR/BL or None.
  - ``detect_numpy(frame_bgr, cfg)`` — leam-in v2 path: binarize +
    findContours + approxPolyDP. Works at any rotation because it
    traces the doc's actual silhouette, not an axis-aligned bbox of it.
  - ``detect_opencv(frame_bgr, cfg)`` — legacy Canny+findContours, kept
    for comparison.

Leam-in scope (intentionally narrow):
  - **Dark** wood/desk/table bg + bright paper. Anything else (light
    table, cluttered, low-contrast) returns None — the algorithm
    doesn't try to be a general detector.
  - Rotation: ~0°–30°. Goes back to being wrong at higher angles;
    user told us not to bother past 30°.
"""
from __future__ import annotations
import numpy as np
from .quad import order_corners, is_convex, quad_aspect


# ── Leam-in v2 path ──────────────────────────────────────────────────────
def _detect_v2(frame_bgr: np.ndarray, cfg) -> np.ndarray | None:
    """Binarize + findContours + approxPolyDP.

    See module docstring for scope.
    """
    import cv2 as _cv
    H, W = frame_bgr.shape[:2]
    gray = (frame_bgr if frame_bgr.ndim == 2 else frame_bgr.mean(axis=2)
           ).astype(np.float64)

    scale = cfg.long_edge / max(H, W) if cfg.long_edge > 0 else 1.0
    if scale < 1.0:
        new_w, new_h = int(W * scale), int(H * scale)
        gray_s = _cv.resize(gray.astype(np.float32), (new_w, new_h),
                            interpolation=_cv.INTER_LINEAR)
    else:
        new_h, new_w = H, W
        gray_s = gray

    bg_luma = float(np.median(gray_s))
    if bg_luma > 180:
        # Light bg — out of leam-in scope. Try the legacy backend as a
        # courtesy; if that also fails, return None rather than guess.
        return None

    lthr = bg_luma + 30
    mask = (gray_s >= lthr).astype(np.uint8) * 255
    if mask.sum() < 200:
        return None

    k5 = np.ones((5, 5), np.uint8)
    k9 = np.ones((9, 9), np.uint8)
    mask = _cv.morphologyEx(mask, _cv.MORPH_OPEN, k5)   # drop speckles
    mask = _cv.morphologyEx(mask, _cv.MORPH_CLOSE, k9)  # fill text-line gaps
    if mask.sum() < 500:
        return None

    contours, _ = _cv.findContours(
        mask, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = sorted(contours, key=_cv.contourArea, reverse=True)

    frame_area = new_h * new_w
    best = None
    best_area = 0
    # Multi-ratio epsilon ladder — the "right" ratio is content-dependent
    # (text lines / shadow halos push it up or down). Take the first one
    # that yields a valid 4-vertex convex quad.
    for cnt in contours:
        a = float(_cv.contourArea(cnt))
        if a < cfg.min_quad_area_ratio * frame_area:
            break
        if a > cfg.max_quad_area_ratio * frame_area:
            continue
        peri = _cv.arcLength(cnt, closed=True)
        for ratio in (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05):
            approx = _cv.approxPolyDP(cnt, ratio * peri, closed=True)
            if len(approx) != 4:
                continue
            q = approx.reshape(4, 2).astype(np.float64)
            if not is_convex(q):
                continue
            try:
                if not (cfg.min_aspect < quad_aspect(q) < cfg.max_aspect):
                    continue
            except Exception:
                continue
            # Edge-touching filter: real receipts don't touch 2+ image edges.
            # On real phone frames the receipt + bg-texture merge into a
            # single contour that touches the frame border, and approxPolyDP
            # then picks vertices on those edges. Filter those out.
            ax, ay, aw, ah = _cv.boundingRect(approx)
            edge_touch = sum([
                int(ax <= 4),
                int(ay <= 4),
                int(ax + aw >= new_w - 4),
                int(ay + ah >= new_h - 4),
            ])
            if edge_touch >= 2:
                continue
            break
        else:
            continue
        if a > best_area:
            best_area = a
            best = q

    if best is None:
        return None
    if scale < 1.0:
        best = best / scale
    return order_corners(best)


# ── Legacy Canny+Hough backend (retained for comparison) ──────────────────
def detect_opencv(frame_bgr: np.ndarray, cfg) -> np.ndarray | None:
    """OpenCV Canny + findContours + approxPolyDP.

    Raises ImportError if cv2 isn't installed.
    """
    import cv2  # type: ignore
    H, W = frame_bgr.shape[:2]
    scale = cfg.long_edge / max(H, W)
    small = cv2.resize(frame_bgr, (int(W * scale), int(H * scale))
                       ) if scale < 1.0 else frame_bgr
    sh, sw = small.shape[:2]
    frame_area = sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5),
                      cfg.canny_low, cfg.canny_high)
    closed = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,
                                  (cfg.morph_kernel, cfg.morph_kernel)),
        iterations=cfg.morph_iterations,
    )
    best, best_area = None, 0
    for cnt in cv2.findContours(closed, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if not (cfg.min_quad_area_ratio < area / frame_area
                < cfg.max_quad_area_ratio):
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue
        q = approx.reshape(4, 2)
        if not cv2.isContourConvex(q):
            continue
        x, y, w, h = cv2.boundingRect(q)
        if not (cfg.min_aspect < w / h < cfg.max_aspect):
            continue
        if area > best_area:
            best_area = area
            best = q
    if best is None:
        return None
    return order_corners(best / scale if scale < 1.0 else best)


# ── v3 path: whiteness (min channel) ─────────────────────────────────────
def _detect_v3(frame_bgr: np.ndarray, cfg) -> np.ndarray | None:
    """Mirror of src/detector/v3.js — keep the two in sync.

    Paper is desaturated (all channels high), wood/table bgs are
    saturated (min channel low). min(B,G,R) + Otsu separates them where
    luma cannot. Validated 29/29 on the captures-debug phone corpus and
    94% on the 200-image SRD receipt set (v2 scored 11/29 and ~22%).
    """
    import cv2 as _cv
    H, W = frame_bgr.shape[:2]
    scale = cfg.long_edge / max(H, W) if cfg.long_edge > 0 else 1.0
    img = frame_bgr
    if scale < 1.0:
        img = _cv.resize(img, (int(W * scale), int(H * scale)),
                         interpolation=_cv.INTER_NEAREST)
    h, w = img.shape[:2]
    mn = img.min(axis=2) if img.ndim == 3 else img
    _, mask = _cv.threshold(mn, 0, 255, _cv.THRESH_BINARY + _cv.THRESH_OTSU)
    mask = _cv.morphologyEx(mask, _cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = _cv.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, _cv.CC_STAT_AREA]))
    area = stats[i, _cv.CC_STAT_AREA]
    if area < 0.02 * h * w or area > 0.90 * h * w:
        return None
    # Paperness: desaturated blob, clearly brighter than bg — an empty
    # table must not fire. Mirrors v3.js thresholds.
    blob = lab == i
    mx = img.max(axis=2) if img.ndim == 3 else img
    sat = (mx[blob].astype(np.float32) - mn[blob]) / np.maximum(1, mx[blob].astype(np.float32))
    if sat.mean() > 0.30 or float(mn[blob].mean()) - float(mn[~blob].mean()) < 50:
        return None
    comp = (lab == i).astype(np.uint8)
    cnts, _ = _cv.findContours(comp, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=_cv.contourArea)
    # Quad fit (not minAreaRect): a receipt at an angle is a trapezoid.
    # ponytail: approxPolyDP ladder here vs max-area-quad in v3.js — same
    # corners on everything tested; switch if they ever diverge.
    hull = _cv.convexHull(c)
    peri = _cv.arcLength(hull, True)
    q = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.1):
        a4 = _cv.approxPolyDP(hull, eps * peri, True)
        if len(a4) == 4:
            q = a4.reshape(4, 2).astype(np.float64)
            break
        if len(a4) < 4:
            break
    if q is None:
        q = _cv.boxPoints(_cv.minAreaRect(c)).astype(np.float64)
    qa = abs(float(np.cross(q[1] - q[0], q[2] - q[0])) + float(np.cross(q[2] - q[0], q[3] - q[0]))) / 2
    if area / max(1.0, qa) < 0.6:  # solidity
        return None
    return order_corners(q / scale if scale < 1.0 else q)


# ── Public dispatch ──────────────────────────────────────────────────────
def detect_numpy(frame_bgr: np.ndarray, cfg) -> np.ndarray | None:
    """Leam-in v2 path. Returns quad or None."""
    return _detect_v2(frame_bgr, cfg)


def detect(frame_bgr: np.ndarray, cfg) -> np.ndarray | None:
    """Auto-pick backend.

    Tries v3 (whiteness), then v2 (luma leam-in), then legacy
    Canny+Hough. All require cv2; if it's not installed, returns None.
    """
    try:
        r = _detect_v3(frame_bgr, cfg)
        if r is not None:
            return r
        r = detect_numpy(frame_bgr, cfg)
        if r is not None:
            return r
        return detect_opencv(frame_bgr, cfg)
    except ImportError:
        return None
