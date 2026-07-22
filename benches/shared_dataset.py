"""Python port of the v1 dataset generator (../lean_scanner/src/benchmark/dataset.js).

Generates a labelled dataset of synthetic documents at varying rotations
and backgrounds, suitable for benchmarking the v2 detector against the
v1 radial detector baseline that produced benches/bench-results.csv.

Ported verbatim from the JS — same RNG (mulberry32), same bg profiles
("dark-wood" / "light-table" / "textured-cloth"), same rotation range
[0, 85°], same doc composition (white page with horizontal text bars).

Output is identical to v1's generateDataset() at the same (seed, count,
W, H) — pixel-for-pixel where numpy semantics allow. Use this so
benchmarks are directly comparable.
"""
from __future__ import annotations
import numpy as np

from autocapture.dewarp.homography import compute_homography, warp_bilinear
from autocapture.detector.quad import order_corners


# ── Deterministic RNG ────────────────────────────────────────────────────
def _mulberry32(seed: int):
    """Same mulberry32 used by v1's dataset.js — pixel-for-pixel."""
    a = [seed & 0xFFFFFFFF]

    def next_():
        a[0] = (a[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = a[0]
        t ^= (t >> 15)
        t = (t * ((t | 1) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t ^= t + ((t ^ (t >> 7)) & 0xFFFFFFFF)
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    return next_


# ── Page + bg render ─────────────────────────────────────────────────────
def _make_page_content(src_w: int, src_h: int, rng) -> np.ndarray:
    """Render a paper page (RGB uint8) with a title bar + 12 body lines."""
    data = np.full((src_h, src_w, 3), 240, dtype=np.uint8)
    margin = int(min(src_w, src_h) * 0.08)
    # title bar
    data[margin: margin + int(src_h * 0.06), margin: src_w - margin] = 30
    # body lines
    line_count = 12
    line_top = margin + int(src_h * 0.12)
    line_h = (src_h - 2 * margin - int(src_h * 0.10)) // line_count
    for i in range(line_count):
        line_w = int(src_w * (0.5 + 0.4 * (i % 3) / 2))
        y_line = line_top + i * line_h
        th = max(2, int(line_h * 0.25))
        for y in range(y_line, min(src_h - margin, y_line + th)):
            for x in range(margin, min(src_w - margin, margin + line_w)):
                data[y, x] = (80, 80, 80)
    # sparse noise on paper
    for _ in range(200):
        x = int(rng() * src_w); y = int(rng() * src_h)
        n = int(rng() * 30) - 15
        v = int(data[y, x, 0]) + n
        v = max(0, min(255, v))
        data[y, x] = (v, v, v)
    return data


def _make_bg(W: int, H: int, bg_type: str, rng) -> np.ndarray:
    """Generate one of the v1 backgrounds (dark-wood/light-table/cloth)."""
    if bg_type == "dark-wood":
        base = (40, 30, 25)
    elif bg_type == "light-table":
        base = (230, 225, 215)
    else:
        base = (130, 130, 130)
    data = np.zeros((H, W, 3), dtype=np.uint8)
    data[..., 0] = base[0]; data[..., 1] = base[1]; data[..., 2] = base[2]
    # per-pixel noise (match v1's per-pixel rng calls — slow but exact)
    flat = data.reshape(-1, 3).astype(np.int32)
    for i in range(flat.shape[0]):
        noise = int((rng() - 0.5) * 40)
        flat[i, 0] = max(0, min(255, flat[i, 0] + noise))
        flat[i, 1] = max(0, min(255, flat[i, 1] + noise))
        flat[i, 2] = max(0, min(255, flat[i, 2] + noise))
    data = flat.reshape(H, W, 3).astype(np.uint8)
    # large blobs
    n_blobs = (W * H) // 4000
    for _ in range(n_blobs):
        cx = rng() * W; cy = rng() * H
        r = 8 + rng() * 30
        if bg_type == "dark-wood":
            dark = int(rng() * 30) + 10
        elif bg_type == "light-table":
            dark = int(rng() * 30) - 15
        else:
            dark = int(rng() * 60) - 30
        for y in range(max(0, int(cy - r)), min(H, int(cy + r) + 1)):
            for x in range(max(0, int(cx - r)), min(W, int(cx + r) + 1)):
                dx = x - cx; dy = y - cy
                falloff = max(0.0, 1.0 - np.hypot(dx, dy) / r)
                v0 = int(data[y, x, 0]) - int(dark * falloff)
                v1 = int(data[y, x, 1]) - int(dark * falloff)
                v2 = int(data[y, x, 2]) - int(dark * falloff)
                data[y, x] = (max(0, min(255, v0)),
                              max(0, min(255, v1)),
                              max(0, min(255, v2)))
    return data


def _gt_quad(W: int, H: int, theta: float,
             doc_w: float, doc_h: float) -> np.ndarray:
    """Centered doc, rotated by theta rad, corners ordered TL/TR/BR/BL."""
    cx = W / 2; cy = H / 2
    hw = doc_w / 2; hh = doc_h / 2
    cos_t = np.cos(theta); sin_t = np.sin(theta)
    local = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]
    world = []
    for x, y in local:
        wx = cx + x * cos_t - y * sin_t
        wy = cy + x * sin_t + y * cos_t
        world.append((wx, wy))
    flat = [c for p in world for c in p]
    return order_corners(np.asarray(flat, dtype=np.float64).reshape(4, 2))


def _composite(bg: np.ndarray, W: int, H: int,
               page: np.ndarray, src_w: int, src_h: int,
               dst_quad: np.ndarray) -> np.ndarray:
    """Composite the warped page onto the bg, pixel-by-pixel."""
    src_quad = np.array([[0, 0], [src_w - 1, 0],
                         [src_w - 1, src_h - 1], [0, src_h - 1]],
                        dtype=np.float64)
    H_mat = compute_homography(src_quad, dst_quad.astype(np.float64))
    warped = warp_bilinear(page, H_mat, H, W)
    pm = warped.mean(axis=2)
    bm = bg.mean(axis=2)
    page_mask = (pm > bm + 5) & (pm > 50)
    out = bg.copy()
    out[page_mask] = warped[page_mask]
    return out


def generate_frame(width: int, height: int, theta: float, bg: str,
                   seed: int = 1):
    """One labelled frame. Returns dict with rGBA-like data + gtQuad.

    rGBA shape is (H, W, 3) (BGR order, like v1's dataset.js RGB) + the
    RGBA alpha is dropped — Python's detector expects BGR.
    """
    W, H = width, height
    rng = _mulberry32(seed)
    # match JS: make page BEFORE makeBackground so the RNG draws match
    page_src_w = int(W * 0.4); page_src_h = int(H * 0.35)
    page = _make_page_content(page_src_w, page_src_h, rng)
    bg_arr = _make_bg(W, H, bg, rng)
    doc_w = W * 0.6; doc_h = H * 0.5
    gt = _gt_quad(W, H, theta, doc_w, doc_h)
    composited = _composite(bg_arr, W, H, page, page_src_w, page_src_h, gt)
    return {
        "frame": composited,  # (H, W, 3) BGR uint8
        "gt": gt,
        "bg": bg,
        "theta": theta,
    }


def generate_dataset(count: int = 250, width: int = 320, height: int = 240,
                     seed: int = 1, max_angle_deg: float = 85.0,
                     bg_types: list[str] | None = None):
    """Default v1 protocol: 3 bgs × 18 angles [0, 85°] in 5° steps.

    For the leam-in scope we typically restrict to dark-wood (or wood-tone)
    backgrounds. Pass ``bg_types=["dark-wood"]`` to limit.

    Returns array of frame dicts (see generate_frame).
    """
    if bg_types is None:
        bg_types = ["dark-wood", "light-table", "textured-cloth"]
    angles = [i * 5 for i in range(int(max_angle_deg / 5) + 1)]
    per_combo = max(1, (count + len(bg_types) * len(angles) - 1) //
                    (len(bg_types) * len(angles)))
    frames = []
    i = 0
    for bg_t in bg_types:
        for ang in angles:
            theta = ang * np.pi / 180
            for _ in range(per_combo):
                if len(frames) >= count:
                    return frames
                frames.append(generate_frame(width, height, theta, bg_t,
                                             seed=seed + i))
                i += 1
    return frames
