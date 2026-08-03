"""
Smoke tests for each pipeline.

Run:  pytest smoothie_cv/tests/test_pipeline.py -v
"""

from __future__ import annotations

import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendResult
from smoothie_cv.pipelines.blend import BlendPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_image(h: int = 100, w: int = 100, color: tuple = (60, 120, 40)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def _patchy_image(h: int = 100, w: int = 100) -> np.ndarray:
    img = _solid_image(h, w, color=(60, 120, 40))
    img[30:70, 30:70] = (20, 30, 200)  # red chunk in center
    return img


def _roi(h: int = 100, w: int = 100) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[5:95, 5:95] = 255
    return mask


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

class TestRoiCrop:
    def test_crop_zeros_pixels_outside_contour(self):
        from smoothie_cv.roi import crop_to_roi, paste_mask

        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 255

        roi = crop_to_roi(image, mask)
        assert roi.image.shape == (60, 40, 3)
        assert roi.offset == (30, 20)
        assert (roi.mask > 0).sum() == (mask > 0).sum()
        assert (roi.image[roi.mask == 0] == 0).all()

        crop_result = np.zeros(roi.image.shape[:2], dtype=np.uint8)
        crop_result[roi.mask > 0] = 255
        full = paste_mask(crop_result, roi)
        assert full.shape == (100, 100)
        assert (full == mask).all()

    def test_paste_round_trip(self):
        from smoothie_cv.roi import crop_to_roi, paste_mask

        image = _solid_image()
        mask = _roi()
        roi = crop_to_roi(image, mask)
        crop_mask = np.zeros(roi.image.shape[:2], dtype=np.uint8)
        crop_mask[10:20, 10:20] = 255
        full = paste_mask(crop_mask, roi)
        x, y = roi.offset
        assert full[y + 10 : y + 20, x + 10 : x + 20].sum() == 255 * 100


# ---------------------------------------------------------------------------
# Active YOLO blend pipeline
# ---------------------------------------------------------------------------

class TestBlendPipeline:
    def setup_method(self):
        self.pipeline = BlendPipeline(Config())

    def test_returns_blend_result(self):
        result = self.pipeline.analyze(_solid_image(), _roi())
        assert isinstance(result, BlendResult)

    def test_score_in_range(self):
        for img in (_solid_image(), _patchy_image()):
            result = self.pipeline.analyze(img, _roi())
            assert 0.0 <= result.blend_score <= 1.0

    def test_mask_shape_and_dtype(self):
        result = self.pipeline.analyze(_solid_image(), _roi())
        assert result.mask.shape == (100, 100)
        assert result.mask.dtype == np.uint8

    def test_pipeline_name(self):
        assert self.pipeline.analyze(_solid_image()).pipeline_name == "blend"

    def test_no_roi_mask(self):
        result = self.pipeline.analyze(_solid_image())
        assert 0.0 <= result.blend_score <= 1.0

    def test_passed_flag_always_passes_at_zero_threshold(self):
        pipeline = BlendPipeline(Config(threshold=0.0))
        assert pipeline.analyze(_solid_image()).passed is True

    def test_mask_zero_outside_roi(self):
        result = self.pipeline.analyze(_patchy_image(), _roi())
        assert (result.mask[_roi() == 0] == 0).all()
