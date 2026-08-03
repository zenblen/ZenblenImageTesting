"""
Central config for the smoothie blendedness pipeline (YOLO-only).

Priority (highest → lowest):
  CLI flags  >  config.yaml  >  dataclass defaults

Paths are relative to the ``active_pipeline/`` working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class Config:
    # --- pass/fail ---
    threshold: float = 0.90          # blend_score >= threshold → passed

    # --- container detection (ROI) ---
    detector_priority: list[str] = field(default_factory=lambda: ["yolo"])
    yolo_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_standard_seg.pt"))

    # --- YOLO-seg SPILL detection ---
    spill_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_spill_seg.pt"))
    spill_conf: float = 0.35
    spill_min_area_px: int = 400

    # --- YOLO-seg CHUNK detection ---
    chunk_detector_priority: list[str] = field(
        default_factory=lambda: ["yolo"])
    chunk_yolo_input: str = "full_filter"  # "full_filter" | "roi_crop"
    chunk_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_chunk_seg.pt"))
    chunk_conf: float = 0.25

    # --- output ---
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required to load config from YAML. pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> Config:
        """Load from yaml_path if given, otherwise look for config.yaml in cwd."""
        default_path = Path("config.yaml")
        if yaml_path is None and default_path.exists():
            yaml_path = default_path
        if yaml_path is not None:
            return cls.from_yaml(yaml_path)
        return cls()
