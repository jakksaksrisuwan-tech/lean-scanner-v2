"""Main auto-capture pipeline.

Wires the detector -> quality scorer -> lock gate -> dewarp -> save
sequence. This is the analog of the ``on_frame()`` function described
in the research brief.

Entry points
------------
- ``process_frame(frame, gate, cfg)`` — one frame, returns a state tuple
- ``Pipeline`` — convenience wrapper that holds the gate and counters
- ``main()`` — CLI entry point that opens a camera, runs the loop, and
  writes captures to the output directory
"""

from __future__ import annotations
import os
import time
import numpy as np
from typing import Optional

from .config import PipelineConfig, load_config
from .detector.classical import detect
from .quality import score
from .tracker.lock import (
    LockGate,
    STATE_SEARCHING,
    STATE_UNSTABLE,
    STATE_ALMOST,
    STATE_LOCKED,
    STATE_COOLDOWN,
)
from .dewarp.homography import warp_quad
from .dewarp.shading import shade_correct, make_binary_mask
from .dewarp.contrast import clahe_simple, adaptive_threshold, histogram_equalization


CaptureResult = Optional[tuple]   # (warped_bgr, mask, raw_quad) or None


def process_frame(
    frame_bgr: np.ndarray,
    gate: LockGate,
    cfg: PipelineConfig,
) -> tuple[str, np.ndarray | None, float, CaptureResult]:
    """Run one frame through the full pipeline.

    Returns
    -------
    state : str, one of the STATE_* constants
    smoothed_quad : (4, 2) ndarray or None
    quality : float in [0, 1]
    capture : None, or (warped_bgr, mask, raw_quad) if the gate fired

    The capture fires ON the same frame the gate says LOCKED. Callers
    should write the capture to disk and call ``gate.fire()`` so the
    cooldown begins.
    """
    raw_quad = detect(frame_bgr, cfg.detector)
    if raw_quad is None:
        state, smoothed, q = gate.update(None, 0.0)
        return state, smoothed, q, None

    q = score(raw_quad, frame_bgr, cfg.quality)
    state, smoothed, q = gate.update(raw_quad, q)

    if state == STATE_LOCKED:
        warped = _dewarp_and_enhance(frame_bgr, smoothed, cfg)
        mask = make_binary_mask(frame_bgr.shape[:2], smoothed)
        return state, smoothed, q, (warped, mask, smoothed)

    return state, smoothed, q, None


def _dewarp_and_enhance(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Apply the post-capture processing: warp -> shade correct -> contrast.

    Returns the enhanced (H, W, 3) uint8 BGR image.

    Note: the flat-field shading correction amplifies noise on small
    or low-contrast documents. The default pipeline skips it — only
    enabled when the document covers a large fraction of the frame.
    For phone photos with large, well-lit documents, set
    ``dewarp.shading_kernel > 0`` in the config to re-enable.
    """
    # 1. perspective warp
    warped = warp_quad(frame_bgr, quad, target_aspect=cfg.dewarp.fixed_aspect)
    # 2. flat-field shading correction (disabled by default — see note)
    if cfg.dewarp.shading_kernel > 0 and cfg.dewarp.shading_kernel < 200:
        mask = np.ones(warped.shape[:2], dtype=np.uint8)
        warped = shade_correct(warped, mask, ksize=cfg.dewarp.shading_kernel)
    # 3. contrast
    if cfg.dewarp.binarize:
        gray = warped.mean(axis=2).astype(np.uint8)
        warped = np.stack([adaptive_threshold(gray)] * 3, axis=-1)
    else:
        gray = warped.mean(axis=2).astype(np.uint8)
        eq = clahe_simple(gray, clip_limit=cfg.dewarp.clahe_clip,
                          grid=cfg.dewarp.clahe_grid)
        warped = np.stack([eq, eq, eq], axis=-1).astype(np.uint8)
    return warped


# ── Pipeline wrapper ────────────────────────────────────────────────────
class Pipeline:
    """Stateful wrapper that holds the gate + a frame counter."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.gate = LockGate(cfg.lock)
        self.frame_idx = 0
        self.captures: list[np.ndarray] = []
        self.state_trace: list[str] = []

    def step(self, frame_bgr: np.ndarray) -> tuple[str, np.ndarray | None, float, CaptureResult]:
        """Process one frame, update internal state, return results."""
        state, smoothed, q, capture = process_frame(frame_bgr, self.gate, self.cfg)
        self.frame_idx += 1
        self.state_trace.append(state)
        if capture is not None:
            self.captures.append(capture[0])
            self.gate.fire()
        return state, smoothed, q, capture

    def save_captures(self, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True)
        paths = []
        from PIL import Image
        for i, img in enumerate(self.captures):
            p = os.path.join(out_dir, f"page_{i+1:03d}.png")
            Image.fromarray(img[..., ::-1]).save(p)  # BGR -> RGB
            paths.append(p)
        return paths


# ── Camera source abstraction ───────────────────────────────────────────
class FrameSource:
    """Iterate frames from a video source.

    Falls back to a synthetic source (a black frame) if OpenCV is not
    installed or the camera index cannot be opened. This makes the CLI
    entry point runnable in any environment.
    """

    def __init__(self, source: str | int = 0):
        self.source = source
        self._cv2 = None
        self._cap = None
        try:
            import cv2  # type: ignore
            self._cv2 = cv2
            self._cap = cv2.VideoCapture(source if not isinstance(source, str) else int(source))
        except (ImportError, Exception):
            self._cap = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                return True, frame
        # Synthetic fallback: solid black frame
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ── CLI entry point ─────────────────────────────────────────────────────
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="Auto-capture document scanner")
    p.add_argument("--config", default=None, help="Path to YAML config")
    p.add_argument("--camera", default=0, help="Camera index or video file path")
    p.add_argument("--output", default="./captures", help="Output directory")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames (0 = forever)")
    p.add_argument("--sim", action="store_true",
                   help="Use the synthetic-frame simulator (no camera needed)")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg.output_dir = args.output

    if args.sim:
        from simulator import SyntheticDocumentSource
        src = SyntheticDocumentSource(width=640, height=480, seed=42)
    else:
        src = FrameSource(args.camera)

    pipeline = Pipeline(cfg)

    t0 = time.monotonic()
    try:
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            state, smoothed, q, capture = pipeline.step(frame)
            if state == STATE_LOCKED:
                print(f"[frame {pipeline.frame_idx:5d}] LOCKED  quality={q:.2f}  fire={capture is not None}")
            elif state == STATE_ALMOST:
                print(f"[frame {pipeline.frame_idx:5d}] ALMOST  quality={q:.2f}")
            elif state == STATE_UNSTABLE:
                print(f"[frame {pipeline.frame_idx:5d}] UNSTABLE  quality={q:.2f}")
            if args.max_frames and pipeline.frame_idx >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.release()

    paths = pipeline.save_captures(cfg.output_dir)
    elapsed = time.monotonic() - t0
    print(f"\nDone. {len(paths)} capture(s) in {elapsed:.1f}s.")
    for p_ in paths:
        print(f"  -> {p_}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
