"""Polynomial text-line flattening (vertical strip-correlation) — prototype.

Estimates a smooth dy(x) displacement by cross-correlating ink profiles
of adjacent vertical strips, integrating the shifts, and fitting a
low-order polynomial; remaps columns so text baselines go horizontal.

MEASURED on 174 SRD dewarped crops (Apple Vision OCR):
  aggregate conf 0.81 -> 0.79, wins 34 / loses 78  => NET NEGATIVE blind.
  Genuinely curled tail gains big (0.65->0.78, 0.70->0.77);
  flat receipts with sparse/columnar text get damaged (0.80->0.57) —
  strip correlation mistakes price columns for curl.

To ship this it needs an is-actually-curled gate (correlation peak
quality + low residual of shifts vs the polynomial) so it only fires on
the curled tail. Vision is already curl-tolerant; weaker OCR engines
(tesseract-class) would benefit more — revisit if the OCR engine changes.

Usage: python3 research/polyflatten_prototype.py in.png out.png
"""
import sys

import cv2
import numpy as np


def polyflatten(bgr, nstrips=24, deg=3):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    maxshift = max(6, H // 40)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 12)
    sw = max(8, W // nstrips)
    ns = W // sw
    profs = []
    for i in range(ns):
        p = thr[:, i * sw:(i + 1) * sw].sum(axis=1).astype(np.float32)
        profs.append(p - p.mean())
    shifts = [0.0]
    for i in range(1, ns):
        a, b = profs[i - 1], profs[i]
        best, bs = -1e18, 0
        for s in range(-maxshift, maxshift + 1):
            v = (a[s:] * b[:H - s]).sum() if s >= 0 else (a[:H + s] * b[-s:]).sum()
            if v > best:
                best, bs = v, s
        shifts.append(shifts[-1] + bs)
    shifts = np.array(shifts, np.float32)
    shifts -= shifts.mean()
    amp = float(np.abs(shifts).max())
    if amp < 3 or amp > H * 0.25:   # flat, or rotation-dominated: refuse
        return bgr, amp
    xs = (np.arange(ns) + 0.5) * sw
    dy = np.polyval(np.polyfit(xs, shifts, deg), np.arange(W)).astype(np.float32)
    mapx, mapy = np.meshgrid(np.arange(W, dtype=np.float32),
                             np.arange(H, dtype=np.float32))
    return cv2.remap(bgr, mapx, mapy + dy[None, :], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE), amp


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    out, amp = polyflatten(img)
    cv2.imwrite(sys.argv[2], out)
    print(f"curl amplitude {amp:.0f}px")
