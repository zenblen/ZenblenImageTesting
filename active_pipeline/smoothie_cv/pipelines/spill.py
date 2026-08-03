"""
Spill-detection pipeline.

Spill = any smoothie material OUTSIDE the cup (drips/pooling on the holder
gasket, splatter on the machine). Detection is done directly by the fine-tuned
YOLO-seg spill model (``smoothie_cv.detection.spill.detect_spill``); this
pipeline wraps it in the analysis contract, applies the min-area floor that
turns the raw mask into a binary verdict, and packages a ``SpillResult``.

Unlike the chunk pipeline it does NOT take an ROI: spill is defined relative to
the cup exterior, and the model was trained to segment spilled material wherever
it lands, so the whole frame is in scope. (``analyze`` accepts an ``roi_mask``
argument for interface symmetry with BlendPipeline but ignores it.)
"""

from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import SpillResult


class SpillPipeline:

    name = "spill"

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> SpillResult:
        from smoothie_cv.detection.spill import detect_spill

        mask, confs, max_conf = detect_spill(image, self.config)
        area = int((mask > 0).sum())
        detected = area >= self.config.spill_min_area_px

        # If the union clears no min-area floor we still report the raw area, but
        # the verdict is "clean". Zero out a sub-threshold mask so downstream
        # overlays/scores don't show noise specks as a spill.
        if not detected:
            mask = np.zeros_like(mask)

        return SpillResult(
            spill_detected=detected,
            spill_area_px=area,
            mask=mask,
            confidence=max_conf,
            pipeline_name=self.name,
            metadata={
                "n_instances": len(confs),
                "instance_confs": [round(c, 3) for c in confs],
                "min_area_px": self.config.spill_min_area_px,
                "conf_floor": self.config.spill_conf,
            },
        )
