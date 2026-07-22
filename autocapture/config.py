"""Configuration loader for the auto-capture document scanner.

The config is intentionally tiny. Thresholds are documented inline so that
tuning them is a one-file change and the design is reviewable on a screen.
"""

from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class DetectorConfig:
    """Document-quad detection parameters."""
    long_edge: int = 640              # resize long edge to this before detection
    canny_low: int = 75
    canny_high: int = 200
    morph_kernel: int = 9
    morph_iterations: int = 3
    min_quad_area_ratio: float = 0.05  # 5% of frame
    max_quad_area_ratio: float = 0.98
    min_aspect: float = 0.2
    max_aspect: float = 5.0


@dataclass
class QualityConfig:
    """Per-frame quality scoring weights and thresholds."""
    # capture fires when score >= this AND stable_frames >= N
    capture_threshold: float = 0.70
    # component weights (must sum to 1.0)
    weight_area: float = 0.30
    weight_straightness: float = 0.20
    weight_sharpness: float = 0.25
    weight_exposure: float = 0.10
    weight_blur: float = 0.15
    # component sub-thresholds
    min_area_ratio: float = 0.10
    target_area_ratio: float = 0.30
    sharpness_full: float = 200.0     # laplacian variance for full score
    blur_full: float = 80.0           # sobel magnitude mean for full score
    exposure_low: float = 60.0
    exposure_high: float = 220.0
    glare_pixel_threshold: int = 240
    glare_max_ratio: float = 0.15


@dataclass
class LockConfig:
    """Stability gate (the QR-style N-frame debounce)."""
    n_stable_frames: int = 8          # ~267 ms at 30 fps
    corner_dist_px: float = 20.0      # max mean corner drift between frames
    quality_release_ratio: float = 0.8  # release if quality drops below capture_threshold * this
    cooldown_ms: int = 800            # suppress duplicate fires on same target


@dataclass
class DewarpConfig:
    """Post-capture processing."""
    fixed_aspect: str = "A4"          # "A4" | "US_LETTER" | "MAX_EDGE" | "AUTO"
    shading_kernel: int = 201
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    binarize: bool = False


@dataclass
class PipelineConfig:
    camera_index: int = 0
    fps: int = 30
    output_dir: str = "./captures"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    lock: LockConfig = field(default_factory=LockConfig)
    dewarp: DewarpConfig = field(default_factory=DewarpConfig)


def _merge_into(dc_obj, mapping):
    for k, v in mapping.items():
        if hasattr(dc_obj, k):
            cur = getattr(dc_obj, k)
            if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
                _merge_into(cur, v)
            else:
                setattr(dc_obj, k, v)
    return dc_obj


def load_config(path: str | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if path and os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _merge_into(cfg, data)
    return cfg


def save_config(cfg: PipelineConfig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
