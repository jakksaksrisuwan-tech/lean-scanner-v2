"""Synthetic document video source.

Generates a sequence of frames showing a document at various positions
and orientations, so the auto-capture pipeline can be exercised end to
end without a real camera.

The synthetic document is rendered onto a contrasting background. It
``approaches`` the camera, holds steady, then moves out. The pipeline
should fire during the steady hold.

This is a self-test for the detector + tracker + dewarp chain.
"""

from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _draw_document_page(w: int, h: int, lines: int = 12) -> np.ndarray:
    """Render a clean B&W-ish document page on a white background."""
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    # margin
    margin = int(min(w, h) * 0.08)
    # title bar
    draw.rectangle([margin, margin, w - margin, margin + int(h * 0.06)], fill="black")
    # body lines
    line_h = int((h - margin * 2 - int(h * 0.10)) / lines)
    y = margin + int(h * 0.12)
    for i in range(lines):
        line_w = int(w * (0.5 + 0.4 * (i % 3) / 2))
        draw.rectangle([margin, y, margin + line_w, y + int(line_h * 0.3)],
                       fill=(60, 60, 60))
        y += line_h
    arr = np.asarray(img)
    return arr[..., ::-1].copy()  # RGB -> BGR


def _warp_perspective(src: np.ndarray, dst_quad: np.ndarray,
                      out_size: tuple[int, int]) -> np.ndarray:
    """Pure-numpy perspective warp (delegates to the project module)."""
    from autocapture.dewarp.homography import compute_homography, warp_bilinear
    h, w = out_size
    src_h, src_w = src.shape[:2]
    # The src quad is the page corners in src order
    src_quad = np.array(
        [[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]],
        dtype=np.float64
    )
    H = compute_homography(src_quad, dst_quad.astype(np.float64))
    return warp_bilinear(src, H, h, w)


class SyntheticDocumentSource:
    """Emits a stream of frames simulating a document in front of a camera.

    The script is deterministic given a seed. Pattern:
      - frames 0..30:    doc enters from the right, off-center
      - frames 30..90:   doc holds steady near the center, slight wobble
      - frames 90..120:  doc moves away / out of frame
      - frames 120..150: blank frame (cooldown)
      - frames 150..210: doc re-enters and holds again
    """

    def __init__(self, width: int = 640, height: int = 480, seed: int = 42):
        self.W, self.H = width, height
        self.frame_idx = 0
        self.rng = np.random.default_rng(seed)
        self._page = _draw_document_page(int(width * 0.6), int(height * 0.75))
        self._bg_color = (40, 30, 25)  # dark wood-ish, BGR

    def _background(self) -> np.ndarray:
        bg = np.full((self.H, self.W, 3), self._bg_color, dtype=np.uint8)
        # Add a tiny bit of texture (deterministic)
        noise = self.rng.integers(-10, 10, size=bg.shape, dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return bg

    def _quad_for_frame(self) -> np.ndarray | None:
        """Return the 4 corners of the doc in this frame, or None if out."""
        f = self.frame_idx
        # blank window
        if 120 <= f < 150:
            return None
        # entry phase 0..30
        if f < 30:
            t = f / 30.0
            cx = self.W * (0.95 - 0.35 * t)
            cy = self.H * 0.55
            scale = 0.5 + 0.4 * t
        # hold phase 30..90
        elif f < 90:
            t = (f - 30) / 60.0
            # gentle wobble
            cx = self.W * 0.55 + 4 * np.sin(2 * np.pi * t * 1.5)
            cy = self.H * 0.50 + 3 * np.cos(2 * np.pi * t * 2.0)
            scale = 0.9
        # exit phase 90..120
        elif f < 120:
            t = (f - 90) / 30.0
            cx = self.W * (0.55 - 0.4 * t)
            cy = self.H * (0.50 - 0.2 * t)
            scale = 0.9 * (1.0 - 0.2 * t)
        # second entry 150..180
        elif f < 180:
            t = (f - 150) / 30.0
            cx = self.W * (0.95 - 0.40 * t)
            cy = self.H * 0.55
            scale = 0.5 + 0.4 * t
        # second hold 180..240
        elif f < 240:
            t = (f - 180) / 60.0
            cx = self.W * 0.55 + 5 * np.sin(2 * np.pi * t * 1.2)
            cy = self.H * 0.50
            scale = 0.9
        else:
            return None

        # small perspective skew
        skew_x = 0.05 * np.sin(f * 0.1)
        skew_y = 0.03 * np.cos(f * 0.13)
        doc_w = int(self.W * 0.55 * scale)
        doc_h = int(self.H * 0.7 * scale)
        half_w, half_h = doc_w / 2, doc_h / 2
        quad = np.array([
            [cx - half_w * (1 + skew_x), cy - half_h * (1 - skew_y)],
            [cx + half_w * (1 - skew_x), cy - half_h * (1 + skew_y)],
            [cx + half_w * (1 + skew_x), cy + half_h * (1 - skew_y)],
            [cx - half_w * (1 - skew_x), cy + half_h * (1 + skew_y)],
        ], dtype=np.float64)
        return quad

    def read(self) -> tuple[bool, np.ndarray]:
        frame = self._background()
        quad = self._quad_for_frame()
        if quad is not None:
            doc_h, doc_w = self._page.shape[:2]
            # Use the quad to warp the page onto the background
            # The src quad in the page is the full page rect.
            warped = _warp_perspective(self._page, quad, (self.H, self.W))
            # alpha-composite: where the warped page is brighter than the bg
            # Note: paper is white (luma 255) so the upper bound must be
            # inclusive — the v1 < 250 mask was hiding the paper body.
            mask = (warped.mean(axis=2) > 30) & (warped.mean(axis=2) <= 255)
            mask3 = np.stack([mask, mask, mask], axis=-1)
            frame = np.where(mask3, warped, frame)
        self.frame_idx += 1
        return True, frame

    def release(self) -> None:
        pass
