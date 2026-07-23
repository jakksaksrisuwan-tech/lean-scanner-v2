"""Baseline-driven deskew + curl flatten — v2 estimator.

Order matters (physical, not mathematical):
  1. DESKEW: rotate by the angle maximizing horizontal-projection
     sharpness (text rows become razor peaks when horizontal). Immune to
     price columns; removes the rotation that fooled strip correlation.
  2. FLATTEN: per-text-line baseline estimation on the deskewed image —
     glyph bottoms are baseline samples, descenders are rejected with an
     asymmetric robust fit — then a smooth dy(x, y) field interpolated
     between baselines, applied in one remap.

Both steps no-op below their thresholds. See research/
polyflatten_prototype.py for the failed v1 (strip correlation) and its
measured 34-win/78-loss record — the baseline this must beat.

Usage: python3 research/baseline_flatten.py in.png out.png
"""
import sys

import cv2
import numpy as np


# ── 1. Deskew ────────────────────────────────────────────────────────────
def estimate_skew(gray, max_deg=8.0, coarse=0.5, fine=0.1):
    """Angle (deg) maximizing variance of row ink-sums. Two-stage search."""
    small = gray
    if max(gray.shape) > 800:
        s = 800 / max(gray.shape)
        small = cv2.resize(gray, (int(gray.shape[1] * s), int(gray.shape[0] * s)),
                           interpolation=cv2.INTER_AREA)
    thr = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 12)
    H, W = thr.shape

    def sharpness(deg):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
        rot = cv2.warpAffine(thr, M, (W, H), flags=cv2.INTER_NEAREST)
        prof = rot.sum(axis=1).astype(np.float64)
        return prof.var()

    best = max(np.arange(-max_deg, max_deg + 1e-6, coarse), key=sharpness)
    best = max(np.arange(best - coarse, best + coarse + 1e-6, fine), key=sharpness)
    return float(best)


# ── 2. Baseline flatten ──────────────────────────────────────────────────
def _glyphs(gray):
    """Connected ink components → (cx, bottom_y, h) arrays."""
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 12)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(thr)
    out = []
    Himg = gray.shape[0]
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 6 or h < 3 or h > Himg * 0.1 or w > gray.shape[1] * 0.5:
            continue  # noise, rules, or giant blobs
        out.append((x + w / 2.0, float(y + h), float(h)))
    return np.array(out) if out else np.zeros((0, 3))


def _cluster_lines(glyphs):
    """Greedy y-clustering of glyphs into text lines (post-deskew)."""
    if len(glyphs) == 0:
        return []
    med_h = np.median(glyphs[:, 2])
    order = np.argsort(glyphs[:, 1])
    lines, cur, cur_y = [], [order[0]], glyphs[order[0], 1]
    for idx in order[1:]:
        y = glyphs[idx, 1]
        if y - cur_y <= 0.7 * med_h:
            cur.append(idx)
            cur_y = 0.7 * cur_y + 0.3 * y
        else:
            lines.append(cur)
            cur, cur_y = [idx], y
    lines.append(cur)
    return [glyphs[l] for l in lines if len(l) >= 6]


def _fit_baseline(line, deg=2, drop_below=2.0, iters=3):
    """Asymmetric robust polyfit of bottom-y vs x: descenders (points
    BELOW the fit) are discarded; short glyphs above are kept."""
    pts = line[np.argsort(line[:, 0])]
    x, y = pts[:, 0], pts[:, 1]
    d = min(deg, len(x) - 1)
    keep = np.ones(len(x), bool)
    coef = np.polyfit(x, y, d)
    for _ in range(iters):
        fit = np.polyval(coef, x)
        new = y - fit < drop_below          # below-fit = descender, drop
        new &= y - fit > -3 * np.median(line[:, 2])  # sanity above
        if new.sum() < max(4, d + 2) or np.array_equal(new, keep):
            break
        keep = new
        coef = np.polyfit(x[keep], y[keep], d)
    resid = float(np.median(np.abs(np.polyval(coef, x[keep]) - y[keep])))
    return coef, (float(x.min()), float(x.max())), resid


def flatten(bgr, min_amp=4.0, max_amp_frac=0.15):
    """Deskew + baseline flatten. Returns (out_bgr, info dict)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    ang = estimate_skew(gray)
    out = bgr
    if abs(ang) >= 0.3:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -ang, 1.0)
        out = cv2.warpAffine(bgr, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    lines = _cluster_lines(_glyphs(gray))
    if len(lines) < 3:
        return out, {"angle": ang, "amp": 0.0, "lines": len(lines)}

    # per-line baselines → displacement samples dy = y_fit(x) - y_fit(mid)
    xs_grid = np.linspace(0, W - 1, 64)
    rows = []      # (baseline_y_at_mid, dy_over_grid, span_mask)
    for ln in lines:
        coef, (x0, x1), resid = _fit_baseline(ln)
        if resid > 4 or (x1 - x0) < W * 0.25:
            continue  # noisy or too-short line: no vote
        ymid = np.polyval(coef, W / 2)
        dy = np.polyval(coef, xs_grid) - ymid
        span = (xs_grid >= x0 - 20) & (xs_grid <= x1 + 20)
        rows.append((float(ymid), dy, span))
    if len(rows) < 5:   # curl needs real evidence; deskew already applied
        return out, {"angle": ang, "amp": 0.0, "lines": len(rows)}
    rows.sort(key=lambda r: r[0])

    amp = max(float(np.max(np.abs(dy[span]))) if span.any() else 0.0
              for _, dy, span in rows)
    if amp < min_amp or amp > H * max_amp_frac:
        return out, {"angle": ang, "amp": amp, "lines": len(rows), "applied": False}

    # dy(x, y): vertical interpolation between line baselines; where a line
    # didn't span an x, inherit the nearest spanning line's value.
    ys = np.array([r[0] for r in rows], np.float32)
    dys = np.stack([np.where(r[2], r[1], np.nan) for r in rows]).astype(np.float32)
    # fill NaNs per row from nearest valid columns
    for r in range(dys.shape[0]):
        v = dys[r]
        idx = np.where(~np.isnan(v))[0]
        if len(idx) == 0:
            dys[r] = 0
        else:
            dys[r] = np.interp(np.arange(len(v)), idx, v[idx])
    # interp along y for every image row, then along x for every column
    field = np.empty((H, len(xs_grid)), np.float32)
    for c in range(len(xs_grid)):
        field[:, c] = np.interp(np.arange(H), ys, dys[:, c])
    full = cv2.resize(field, (W, H), interpolation=cv2.INTER_LINEAR)

    mapx, mapy = np.meshgrid(np.arange(W, dtype=np.float32),
                             np.arange(H, dtype=np.float32))
    result = cv2.remap(out, mapx, mapy + full, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return result, {"angle": ang, "amp": amp, "lines": len(rows), "applied": True}


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    res, info = flatten(img)
    cv2.imwrite(sys.argv[2], res)
    print(info)
