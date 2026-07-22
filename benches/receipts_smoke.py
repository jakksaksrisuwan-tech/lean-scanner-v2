"""Quick smoke on real receipt images: run python v2 detector.

Reports per-frame: detected? IoU vs image bbox (since SRD frames are
near-tight crops of the receipt itself)."""
import sys, time
from pathlib import Path
import numpy as np, cv2 as cv

ROOT = Path("/Users/jsaksrisuwan/workspace/lean_scanner_v2")
sys.path.insert(0, str(ROOT))

from autocapture.detector.classical import _detect_v2, detect_opencv
from autocapture.config import DetectorConfig
from autocapture.detector.quad import quad_iou, order_corners

DATA = Path("/Users/jsaksrisuwan/workspace/receipts-dataset")
files = sorted(DATA.glob("*.jpg"))
cfg = DetectorConfig()

# Detector sweep
n_total = 0
n_det_v2 = 0
n_det_opencv = 0
iou_sum_v2 = 0.0
iou_sum_opencv = 0.0
t_sum_v2 = 0.0
for f in files:
    img = cv.imread(str(f))
    if img is None: continue
    n_total += 1
    H, W = img.shape[:2]
    t0 = time.time()
    det_v2 = _detect_v2(img, cfg)
    t_v2 = (time.time() - t0) * 1000
    t_sum_v2 += t_v2
    det_ocv = detect_opencv(img, cfg)
    gt = order_corners(np.array([0, 0, W-1, 0, W-1, H-1, 0, H-1], dtype=np.float64))
    if det_v2 is not None:
        n_det_v2 += 1
        iou_v2 = quad_iou(det_v2, gt)
        iou_sum_v2 += iou_v2
    if det_ocv is not None:
        n_det_opencv += 1
        iou_ocv = quad_iou(det_ocv, gt)
        iou_sum_opencv += iou_ocv

print(f"\n[{n_total} SRD receipt frames]")
print(f"  v2 detector:    detected {n_det_v2}/{n_total} ({100*n_det_v2/n_total:.0f}%)  "
      f"mean IoU={iou_sum_v2/max(n_det_v2,1):.3f}  "
      f"mean time={t_sum_v2/max(n_total,1):.1f}ms")
print(f"  opencv legacy:   detected {n_det_opencv}/{n_total} ({100*n_det_opencv/n_total:.0f}%)  "
      f"mean IoU={iou_sum_opencv/max(n_det_opencv,1):.3f}")
