"""Gaussian-biased flood-fill document detector — offline prototype.

See research/flood-detector-experiment.md for results and verdict.
Run:  python3 research/flood_prototype.py <dump-name-without-ext> ...
"""
import heapq
import json
import os
import sys

import cv2
import numpy as np


def evidence(small):
    """Per-pixel documentness in [0,1]: whiteness OR local ink density."""
    f = small.astype(np.float32)
    mn = f.min(axis=2)
    mx = f.max(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
    wn = (mn / 255.0) * (1 - sat)
    gx = cv2.Sobel(mn, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(mn, cv2.CV_32F, 0, 1, 3)
    mag = np.abs(gx) + np.abs(gy)
    ink = (mag > max(30, np.percentile(mag, 92))).astype(np.float32)
    dens = np.clip(cv2.boxFilter(ink, -1, (13, 13)) / 0.30, 0, 1)
    return np.maximum(wn, dens * 0.9)


def flood_quad(rgb, sigma_frac=0.45):
    H0, W0 = rgb.shape[:2]
    s = 160 / max(H0, W0)
    small = cv2.resize(rgb, (int(W0 * s), int(H0 * s)), interpolation=cv2.INTER_AREA)
    H, W = small.shape[:2]
    ev = evidence(small)
    yy, xx = np.mgrid[0:H, 0:W]
    gauss = np.exp(-0.5 * ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((sigma_frac * min(W, H)) ** 2))
    E = ev * (0.35 + 0.65 * gauss)

    seed = np.unravel_index(np.argmax(E), E.shape)
    visited = np.zeros((H, W), bool)
    visited[seed] = True
    pq = [(-E[seed], seed[0], seed[1])]
    order = []
    while pq:
        negl, y, x = heapq.heappop(pq)
        order.append((y, x))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                visited[ny, nx] = True
                heapq.heappush(pq, (-E[ny, nx], ny, nx))

    # sweep flood sizes; waterline = most rectangular puddle not hugging borders
    minA, maxA = int(0.03 * H * W), int(0.55 * H * W)
    mask = np.zeros((H, W), np.uint8)
    idx = 0
    best, best_score = None, -1
    for target in np.unique(np.linspace(minA, min(maxA, len(order)), 50).astype(int)):
        while idx < target:
            y, x = order[idx]
            mask[y, x] = 1
            idx += 1
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < minA:
            continue
        x0, y0, w0, h0 = cv2.boundingRect(c)
        if (x0 <= 1) + (y0 <= 1) + (x0 + w0 >= W - 1) + (y0 + h0 >= H - 1) >= 2:
            continue
        rect = cv2.minAreaRect(c)
        ra = rect[1][0] * rect[1][1]
        if ra < 1:
            continue
        if a / ra > best_score:
            best_score, best = a / ra, mask.copy()
    if best is None:
        return None

    n, lab, st, _ = cv2.connectedComponentsWithStats(best)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    q = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.1):
        a4 = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a4) == 4:
            q = a4.reshape(4, 2).astype(np.float64)
            break
        if len(a4) < 4:
            break
    if q is None:
        q = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
    return q / s


if __name__ == "__main__":
    for name in sys.argv[1:]:
        meta = json.load(open(f"captures-debug/{name}.json"))
        raw = f"captures-debug/{name}.rgb.raw"
        rgb = np.frombuffer(open(raw, "rb").read(), np.uint8).reshape(meta["h"], meta["w"], 3)
        q = flood_quad(rgb)
        print(name, "->", None if q is None else q.round(1).tolist())
