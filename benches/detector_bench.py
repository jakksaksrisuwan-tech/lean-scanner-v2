"""Detector benchmark — generates a labelled dataset (v1 protocol) and
scores the v2 detector against v1's radial detector baseline.

Run:
    PYTHONPATH=. python3 benches/detector_bench.py [--count N] [--max-angle 85]
                                                    [--out artifacts/bench/]

Dataset:
    Same generator as v1's src/benchmark/dataset.js — 3 bg types
    ("dark-wood", "light-table", "textured-cloth") × 18 rotation angles
    [0, 5, 10, ..., 85°] × N frames per (rotation, bg) cell. The
    generator is ported to python as benches/shared_dataset.py so the
    numbers are directly comparable to v1's radial detector baseline
    in ../lean_scanner/benches/bench-results.csv.

Report:
    Mean IoU per (rotation, bg) cell, plus an overall summary. Writes:
      bench-results.csv    per-frame (matches v1's schema)
      bench-summary.json   per-cell stats + overall
      example_*.png        5 visual examples (worst/median/best)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocapture.detector.classical import detect
from autocapture.detector.quad import quad_iou, order_corners
from autocapture.config import DetectorConfig
from shared_dataset import generate_dataset


def _overlay(frame: np.ndarray, gt: np.ndarray, det: np.ndarray | None,
             iou: float, det_rate_str: str = "") -> np.ndarray:
    """Red = GT, green = detection."""
    out = frame.copy()
    cv.polylines(out, [gt.astype(int).reshape(-1, 1, 2)], True, (0, 0, 255), 2)
    if det is not None:
        cv.polylines(out, [det.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for p in det:
            cv.circle(out, tuple(p.astype(int)), 4, (0, 255, 0), -1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-angle", type=float, default=85.0,
                    help="max rotation in degrees (default 85, matches v1)")
    ap.add_argument("--bg-types", default="dark-wood,light-table,textured-cloth",
                    help="comma-separated bg types to include "
                         "(default: v1's full set; leam-in is 'dark-wood' only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="artifacts/bench")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Generate dataset (v1 protocol)
    bg_types = [s.strip() for s in args.bg_types.split(",") if s.strip()]
    t0 = time.time()
    frames = generate_dataset(count=args.count, width=args.width,
                              height=args.height, seed=args.seed,
                              max_angle_deg=args.max_angle,
                              bg_types=bg_types)
    t_gen = time.time() - t0
    print(f"Generated {len(frames)} labelled frames in {t_gen:.2f}s "
          f"({args.width}x{args.height}, seed={args.seed})")

    cfg = DetectorConfig()
    csv_rows: list[dict] = []
    per_cell: dict[tuple[int, str], dict] = {}

    t_det0 = time.time()
    for idx, f in enumerate(frames):
        frame = f["frame"]
        gt = f["gt"].astype(np.float64)
        t0_per = time.time()
        det = detect(frame, cfg)
        elapsed_ms = (time.time() - t0_per) * 1000.0

        bg = f["bg"]
        theta_deg = f["theta"] * 180 / np.pi
        if det is None:
            iou = 0.0
            detected = False
            det_to_save = None
        else:
            iou = quad_iou(det.astype(np.float64), gt)
            detected = True
            det_to_save = det

        # CSV row (matches v1's bench-results.csv schema)
        csv_rows.append({
            "index": idx,
            "theta_deg": round(theta_deg, 2),
            "bg": bg,
            "detected": int(detected),
            "iou": round(iou, 4),
            "corner_err_px": round(_corner_err(det, gt), 2) if detected else "",
            "center_off_px": round(_center_off(det, gt), 2) if detected else "",
            "elapsed_ms": round(elapsed_ms, 2),
        })

        cell_key = (int(round(theta_deg)), bg)
        if cell_key not in per_cell:
            per_cell[cell_key] = {"iou_sum": 0.0, "found": 0, "n": 0,
                                  "best_iou": -1, "worst_iou": 2,
                                  "best_idx": -1, "worst_idx": -1}
        c = per_cell[cell_key]
        c["n"] += 1
        if detected:
            c["iou_sum"] += iou
            c["found"] += 1
            if iou > c["best_iou"]:
                c["best_iou"] = iou; c["best_idx"] = idx
            if iou < c["worst_iou"]:
                c["worst_iou"] = iou; c["worst_idx"] = idx

        # Capture the per-frame outcome for visual examples at the end.
        f["_det"] = det_to_save; f["_iou"] = iou; f["_detected"] = detected

    t_det = time.time() - t_det0

    # ── Print report ─────────────────────────────────────────────────────
    bg_types = sorted({k[1] for k in per_cell})
    rotations = sorted({k[0] for k in per_cell})

    print(f"\nDetector benchmark — {len(frames)} frames, "
          f"{args.width}x{args.height}\n")
    print("=" * 78)
    header = "rot | " + " | ".join(f"{bn:>14}" for bn in bg_types) + " | row mean"
    print(header)
    print("-" * len(header))
    row_means = []
    for rot in rotations:
        cells = []
        row_str = []
        for bn in bg_types:
            c = per_cell.get((rot, bn))
            if c is None:
                cells.append(0.0); row_str.append(f"{'---':>14}")
                continue
            mean_iou = c["iou_sum"] / max(c["n"], 1)
            det_rate = c["found"] / max(c["n"], 1)
            cells.append(mean_iou)
            row_str.append(f"{mean_iou:.3f} ({det_rate:.0%})")
        rm = float(np.mean(cells)) if cells else 0.0
        row_means.append(rm)
        print(f"{rot:>3} | " + " | ".join(f"{s:>14}" for s in row_str) +
              f" | {rm:.3f}")
    print("-" * len(header))
    col_strs = []; col_det = []
    for bn in bg_types:
        c_vals = [per_cell.get((r, bn), {}).get("iou_sum", 0.0) for r in rotations]
        n_vals = [per_cell.get((r, bn), {}).get("n", 1) for r in rotations]
        s = sum(c_vals); n = sum(n_vals)
        col_strs.append(f"{s / n:.3f}" if n else "---")
        # detection rate
        fd = sum(per_cell.get((r, bn), {}).get("found", 0) for r in rotations)
        col_det.append(fd / n if n else 0.0)
    overall = float(np.mean(row_means)) if row_means else 0.0
    print(f"col | " + " | ".join(f"{s:>14}" for s in col_strs) +
          f" | {overall:.3f} overall")
    for bn, fr in zip(bg_types, col_det):
        print(f"   {bn}: detection rate {fr:.1%}")
    print(f"\ntotal detection time {t_det:.2f}s, "
          f"{(t_det * 1000 / max(len(frames), 1)):.2f}ms/frame")

    # ── Save CSV + summary JSON ──────────────────────────────────────────
    csv_path = os.path.join(args.out, "bench-results.csv")
    with open(csv_path, "w", newline="") as f_csv:
        w = csv.DictWriter(f_csv, fieldnames=list(csv_rows[0].keys()))
        w.writeheader(); w.writerows(csv_rows)
    summary = {
        "config": {
            "count": args.count, "width": args.width, "height": args.height,
            "seed": args.seed, "max_angle": args.max_angle,
        },
        "overall_mean_iou": overall,
        "cells": {
            f"{rot}|{bn}": {
                "iou_sum": c["iou_sum"], "n": c["n"], "found": c["found"],
                "mean_iou": c["iou_sum"] / c["n"] if c["n"] else 0,
                "best_iou": c["best_iou"], "worst_iou": c["worst_iou"],
            }
            for (rot, bn), c in per_cell.items()
        },
    }
    with open(os.path.join(args.out, "bench-summary.json"), "w") as f_json:
        json.dump(summary, f_json, indent=2)

    # ── 5 visual examples ────────────────────────────────────────────────
    found = [f for f in frames if f["_detected"]]
    found.sort(key=lambda f: f["_iou"])
    picks = []
    if len(found) >= 5:
        picks = [found[0], found[1], found[len(found) // 2],
                 found[-2], found[-1]]
    elif len(found) > 0:
        picks = found
    for j, f in enumerate(picks):
        ov = _overlay(f["frame"], f["gt"], f["_det"], f["_iou"])
        label = f"rot={f['theta']*180/np.pi:.0f}  bg={f['bg']}  iou={f['_iou']:.2f}"
        cv.putText(ov, label, (5, 16), cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 1)
        cv.imwrite(os.path.join(args.out,
                                f"example_{j}_iou{int(f['_iou']*100):02d}.png"),
                   ov)

    print(f"\nArtifacts written to {args.out}/")
    print(f"  bench-results.csv    (matches v1's per-frame schema)")
    print(f"  bench-summary.json")
    print(f"  example_*.png        (worst / median / best)")
    return 0


def _corner_err(det: np.ndarray | None, gt: np.ndarray) -> float:
    if det is None:
        return float("inf")
    return float(np.linalg.norm(det - gt, axis=1).mean())


def _center_off(det: np.ndarray | None, gt: np.ndarray) -> float:
    if det is None:
        return float("inf")
    return float(np.linalg.norm(det.mean(axis=0) - gt.mean(axis=0)))


if __name__ == "__main__":
    raise SystemExit(main())
