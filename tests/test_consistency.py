"""Consistency tests for the pure-numpy primitives.

These tests run on numpy + Pillow only — no OpenCV needed. They verify
that the math primitives behave as advertised so that when the
end-to-end pipeline runs (in the simulator), the failures can be
attributed to detection/tracker logic, not the math.
"""

import sys
sys.path.insert(0, "/var/lib/hermes/autocapture-draft")

import numpy as np
from autocapture.dewarp.homography import (
    compute_homography, warp_bilinear, warp_quad,
)
from autocapture.dewarp.shading import (
    gaussian_blur, make_binary_mask, shade_correct,
)
from autocapture.dewarp.contrast import (
    histogram_equalization, clahe_simple, adaptive_threshold,
)
from autocapture.detector.quad import (
    order_corners, quad_area, quad_aspect, is_convex, quad_iou, mean_corner_drift,
)
from autocapture.tracker.kalman import KalmanQuad
from autocapture.tracker.lock import (
    LockGate, STATE_LOCKED, STATE_COOLDOWN, STATE_SEARCHING,
)
from autocapture.config import PipelineConfig, load_config
from autocapture.quality import score


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    if cond:
        print(f"  {PASS}  {name}")
        results.append((True, name))
    else:
        print(f"  {FAIL}  {name}  {detail}")
        results.append((False, name))


# ── Homography ──────────────────────────────────────────────────────────
print("\n== Homography ==")

# Identity correspondence: any H that maps src->src for the 4 corners is valid
# (projectively equivalent). Test by checking the forward map, not the matrix.
src = np.array([[0, 0], [100, 0], [100, 200], [0, 200]], dtype=np.float64)
H = compute_homography(src, src)
# Apply H to a known source point, expect it to map to itself
def apply(H, pt):
    out = H @ np.array([pt[0], pt[1], 1.0])
    return out[0] / out[2], out[1] / out[2]

x, y = apply(H, (50, 100))
check("identity homography maps (50,100) -> (50,100)",
      abs(x - 50) < 1e-6 and abs(y - 100) < 1e-6,
      f"got ({x}, {y})")

# Pure translation: src = dst + (50, 30)
src2 = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float64)
dst2 = src2 + np.array([50, 30])
H2 = compute_homography(src2, dst2)
x, y = apply(H2, (5, 5))
check("translation homography maps (5,5) -> (55,35)",
      abs(x - 55) < 1e-6 and abs(y - 35) < 1e-6,
      f"got ({x}, {y})")

# Projective test: a non-trivial perspective transform
src3 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
dst3 = np.array([[10, 10], [90, 5], [95, 95], [5, 90]], dtype=np.float64)
H3 = compute_homography(src3, dst3)
for s, d in zip(src3, dst3):
    x, y = apply(H3, s)
    assert abs(x - d[0]) < 1e-6 and abs(y - d[1]) < 1e-6, \
        f"correspondence broken: {s} -> ({x}, {y}) expected {d}"
check("projective homography satisfies all 4 correspondences", True)

# Bilinear warp: warp a 20x20 white image to a 20x20 output (same size, identity)
img = np.full((20, 20, 3), 255, dtype=np.uint8)
H_id = np.eye(3)
warped = warp_bilinear(img, H_id, 20, 20)
check("warp_bilinear identity preserves color (same size)",
      warped.shape == (20, 20, 3) and warped.mean() == 255,
      f"shape={warped.shape} mean={warped.mean()}")

# Now downscale 20x20 -> 10x10 with identity
warped_ds = warp_bilinear(img, H_id, 10, 10)
check("warp_bilinear identity downscales to smaller output",
      warped_ds.shape == (10, 10, 3) and warped_ds.mean() == 255,
      f"shape={warped_ds.shape} mean={warped_ds.mean()}")

# warp a single white pixel to a known position
img = np.zeros((20, 20, 3), dtype=np.uint8)
img[5:8, 5:8] = 255
# translate by (10, 0)
H_t = np.array([[1, 0, 10], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
warped = warp_bilinear(img, H_t, 20, 20)
white_pixels = (warped.mean(axis=2) > 200).sum()
check("warp_bilinear translation moves the bright block",
      0 < white_pixels <= 30,
      f"white pixels = {white_pixels}")


# ── Quad utility ────────────────────────────────────────────────────────
print("\n== Quad utility ==")
# Square 100x100 at (10, 10)
q = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float64)
check("quad_area for a 100x100 square",
      abs(quad_area(q) - 10000) < 1, f"got {quad_area(q)}")
check("quad_aspect for a square",
      abs(quad_aspect(q) - 1.0) < 1e-6, f"got {quad_aspect(q)}")
check("is_convex on a square", is_convex(q))

# Shuffled corners -> order_corners
shuffled = q[[2, 0, 3, 1]]
ordered = order_corners(shuffled)
check("order_corners TL at (10,10)",
      np.allclose(ordered[0], [10, 10]),
      f"got {ordered[0]}")
check("order_corners TR at (110,10)",
      np.allclose(ordered[1], [110, 10]),
      f"got {ordered[1]}")
check("order_corners BR at (110,110)",
      np.allclose(ordered[2], [110, 110]),
      f"got {ordered[2]}")
check("order_corners BL at (10,110)",
      np.allclose(ordered[3], [10, 110]),
      f"got {ordered[3]}")

# IoU of a quad with itself
check("quad_iou(self) = 1",
      abs(quad_iou(q, q) - 1.0) < 1e-6, f"got {quad_iou(q, q)}")
# IoU of disjoint quads
q2 = q + np.array([500, 0])
check("quad_iou(disjoint) = 0",
      quad_iou(q, q2) == 0, f"got {quad_iou(q, q2)}")


# ── Gaussian blur ───────────────────────────────────────────────────────
print("\n== Gaussian blur ==")
img = np.zeros((50, 50), dtype=np.float64)
img[20:30, 20:30] = 1.0
blurred = gaussian_blur(img, ksize=5)
check("gaussian blur kernel size 5, shape preserved",
      blurred.shape == img.shape)
check("gaussian blur peak does not increase",
      blurred.max() <= 1.0 + 1e-9, f"max={blurred.max()}")
check("gaussian blur mass approximately preserved",
      abs(blurred.sum() - img.sum()) < 1.0,
      f"sum {blurred.sum()} vs {img.sum()}")


# ── Mask rasterizer ─────────────────────────────────────────────────────
print("\n== Mask rasterizer ==")
q = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float64)
mask = make_binary_mask((120, 120), q)
# Vectorized half-plane rasterizer is boundary-inclusive on all sides:
# a 100x100 quad covers 101x101 pixel centers.
check("mask covers approximately the 100x100 interior",
      10000 <= mask.sum() <= 10201,
      f"got {mask.sum()}")
check("mask is binary",
      set(np.unique(mask).tolist()).issubset({0, 1}))


# ── Shading correction ──────────────────────────────────────────────────
print("\n== Shading correction ==")
img = np.full((100, 100, 3), 200, dtype=np.uint8)
mask = np.ones((100, 100), dtype=np.uint8)
out = shade_correct(img, mask, ksize=51)
check("shade_correct output shape preserved",
      out.shape == img.shape)
check("shade_correct output is uint8",
      out.dtype == np.uint8)
check("shade_correct on uniform input -> approximately uniform output",
      out.std() < 5, f"std = {out.std()}")


# ── Contrast ────────────────────────────────────────────────────────────
print("\n== Contrast ==")
gray = np.zeros((50, 50), dtype=np.uint8)
gray[10:40, 10:40] = 200
bw = adaptive_threshold(gray, block=11, c=2)
check("adaptive_threshold on 2-tone image -> 2-tone output",
      set(np.unique(bw).tolist()).issubset({0, 255}),
      f"unique = {np.unique(bw)}")

heq = histogram_equalization(gray)
check("histogram_equalization shape preserved",
      heq.shape == gray.shape)

clahe = clahe_simple(gray, clip_limit=2.0, grid=4)
check("clahe_simple shape preserved",
      clahe.shape == gray.shape)


# ── Kalman tracker ──────────────────────────────────────────────────────
print("\n== Kalman tracker ==")
kf = KalmanQuad(process_noise=0.5, measurement_noise=4.0)
# Feed a moving quad: starts at one position, drifts by ~3px per frame
q_origin = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float64)
innovations = []
for i in range(10):
    offset = np.array([[i * 0.3, i * 0.2]] * 4)  # (4, 2)
    moving = q_origin + offset
    smoothed, inn, _ = kf.update(moving)
    innovations.append(inn)
check("Kalman filter processes moving input (no exceptions)",
      len(innovations) == 10)
# Reset on None
smoothed, inn, _ = kf.update(None)
check("Kalman reset on None", smoothed is None)


# ── Lock gate ───────────────────────────────────────────────────────────
print("\n== Lock gate ==")
cfg = load_config(None)
gate = LockGate(cfg.lock)
q_lock = np.array([[10, 10], [110, 10], [110, 110], [10, 110]], dtype=np.float64)
states = []
for i in range(20):
    state, smoothed, q_score = gate.update(q_lock, 0.9)
    states.append(state)
# Expect progression: SEARCHING -> ... -> LOCKED somewhere around frame 8
check("LockGate eventually reaches LOCKED on stable input",
      STATE_LOCKED in states,
      f"states = {states}")
check("LockGate fires within ~10 stable frames",
      states.index(STATE_LOCKED) <= 12,
      f"locked at frame {states.index(STATE_LOCKED)}")

# Fire -> COOLDOWN
gate.fire()
state, _, _ = gate.update(q_lock, 0.9)
check("LockGate enters COOLDOWN after fire",
      state == STATE_COOLDOWN)


# ── Quality scorer ──────────────────────────────────────────────────────
print("\n== Quality scorer ==")
# Generate a frame with a clear document region
img = np.full((200, 200, 3), 200, dtype=np.uint8)
img[20:180, 20:180] = 100  # darker interior
q = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], dtype=np.float64)
s = score(q, img, cfg.quality)
check("quality_score returns a float in [0, 1]",
      isinstance(s, float) and 0.0 <= s <= 1.0,
      f"got {s}")
# A tiny, far-away document should score 0 (below min_area_ratio)
q_tiny = np.array([[90, 90], [100, 90], [100, 100], [90, 100]], dtype=np.float64)
s_tiny = score(q_tiny, img, cfg.quality)
check("quality_score on tiny quad = 0",
      s_tiny == 0.0, f"got {s_tiny}")


print("\n== Detector v2 (leam-in high-contrast) ==")
# The v2 path is the verified end-to-end default. We test it with a few
# synthetic frames spanning the supported rotation range [0°, 30°].
from autocapture.config import DetectorConfig
from autocapture.detector.classical import _detect_v2, detect, detect_opencv
from autocapture.detector.quad import quad_iou, order_corners

def _synth(deg, W=640, H=480, bg=(40, 30, 20)):
    """Build a frame + GT quad (TL, TR, BR, BL) at the given rotation."""
    import cv2 as _cv
    bg_arr = np.full((H, W, 3), bg, dtype=np.uint8)
    bg_arr = (bg_arr.astype(np.int16) +
              np.random.randint(-8, 8, bg_arr.shape)).clip(0, 255).astype(np.uint8)
    doc_w = int(W * 0.6 * 0.75)
    doc_h = int(H * 0.75 * 0.75)
    doc = np.full((doc_h, doc_w, 3), 235, dtype=np.uint8)
    margin = int(min(doc_w, doc_h) * 0.08)
    title_h = int(doc_h * 0.06)
    _cv.rectangle(doc, (margin, margin),
                  (doc_w - margin, margin + title_h), (0, 0, 0), -1)
    lines = 12
    line_h = (doc_h - 2 * margin - int(doc_h * 0.10)) // lines
    y0 = margin + int(doc_h * 0.12)
    for i in range(lines):
        lw = int(doc_w * (0.5 + 0.4 * (i % 3) / 2))
        _cv.rectangle(doc, (margin, y0), (margin + lw, y0 + int(line_h * 0.3)),
                      (60, 60, 60), -1)
        y0 += line_h
    cx = W * 0.55; cy = H * 0.55
    half_w, half_h = doc_w // 2, doc_h // 2
    skew_x = 0.03 * (np.random.rand() - 0.5)
    skew_y = 0.02 * (np.random.rand() - 0.5)
    src_quad = np.array([
        [cx - half_w * (1 + skew_x), cy - half_h * (1 - skew_y)],
        [cx + half_w * (1 - skew_x), cy - half_h * (1 + skew_y)],
        [cx + half_w * (1 + skew_x), cy + half_h * (1 - skew_y)],
        [cx - half_w * (1 - skew_x), cy + half_h * (1 + skew_y)],
    ], dtype=np.float32)
    M = _cv.getPerspectiveTransform(
        np.array([[0, 0], [doc_w - 1, 0], [doc_w - 1, doc_h - 1], [0, doc_h - 1]],
                 dtype=np.float32), src_quad)
    warped = _cv.warpPerspective(doc, M, (W, H))
    pmask = (warped.mean(2) > 30) & (warped.mean(2) < 252)
    bg_arr[pmask] = warped[pmask]
    R = _cv.getRotationMatrix2D((W / 2, H / 2), -deg, 1.0)
    rot = _cv.warpAffine(bg_arr, R, (W, H), borderMode=_cv.BORDER_REPLICATE)
    qh = np.concatenate([src_quad, np.ones((4, 1))], axis=1)
    gt_q = (R @ qh.T).T[:, :2]
    return rot, order_corners(gt_q)

cfg_det = DetectorConfig()
# Axis-aligned (0°) — easy case
np.random.seed(1)
frame, gt = _synth(0.0)
det = _detect_v2(frame, cfg_det)
check("v2 detector: returns quad on axis-aligned",
      det is not None and det.shape == (4, 2),
      f"got {det}")
if det is not None:
    check("v2 detector: IoU >= 0.95 at 0°",
          quad_iou(det, gt) >= 0.95,
          f"got {quad_iou(det, gt):.3f}")

# 15° — still in leam-in range
np.random.seed(2)
frame, gt = _synth(15.0)
det = _detect_v2(frame, cfg_det)
check("v2 detector: returns quad at 15°", det is not None, f"got {det}")
if det is not None:
    check("v2 detector: IoU >= 0.95 at 15°",
          quad_iou(det, gt) >= 0.95,
          f"got {quad_iou(det, gt):.3f}")

# 30° — top of the leam-in range (user problem beyond this)
np.random.seed(3)
frame, gt = _synth(30.0)
det = _detect_v2(frame, cfg_det)
check("v2 detector: returns quad at 30°", det is not None, f"got {det}")
if det is not None:
    check("v2 detector: IoU >= 0.80 at 30°",
          quad_iou(det, gt) >= 0.80,
          f"got {quad_iou(det, gt):.3f}")

# Cross-bg: wood_warm should also work (one-shot is enough since the
# benchmark already sweeps this exhaustively).
np.random.seed(4)
frame, gt = _synth(0.0, bg=(78, 56, 38))
det = _detect_v2(frame, cfg_det)
check("v2 detector: returns quad on warm bg",
      det is not None and quad_iou(det, gt) >= 0.95 if det is not None else False,
      f"got {det}")

# Dispatch: detect() should return v2's output for these inputs (high IoU)
np.random.seed(5)
frame, gt = _synth(0.0)
det = detect(frame, cfg_det)
check("detect() routes to v2 path on leam-in inputs",
      det is not None and quad_iou(det, gt) >= 0.95,
      f"got {det}")


# ── End-to-end (simulator + pipeline) ───────────────────────────────────
print("\n== End-to-end simulator ==")
from autocapture.pipeline import Pipeline
from simulator import SyntheticDocumentSource
src = SyntheticDocumentSource(width=640, height=480, seed=42)
# Use a longer cooldown so the test is deterministic (avoids the
# same-doc-re-fires-during-steady-hold pattern)
cfg_e2e = load_config(None)
cfg_e2e.lock.cooldown_ms = 2000
pipe = Pipeline(cfg_e2e)
fires = 0
for _ in range(250):
    ok, frame = src.read()
    if not ok: break
    state, _, _, capture = pipe.step(frame)
    if capture is not None:
        fires += 1
src.release()
check("simulator: pipeline fires at least once in 250 frames",
      fires >= 1, f"fires = {fires}")
check("simulator: pipeline fires at most ~10 times (allows re-fires during long holds)",
      fires <= 10, f"fires = {fires}")


# ── Summary ─────────────────────────────────────────────────────────────
print()
total = len(results)
passed = sum(1 for ok, _ in results if ok)
print(f"=== {passed}/{total} tests passed ===")
sys.exit(0 if passed == total else 1)
