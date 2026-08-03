"""
Shared, detector-agnostic helpers for container detection.

Colour-type classification, top-edge roughness, and the ROI overlay.
Active YOLO logic lives in ``yolo.py``.
"""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


class SmoothieType(Enum):
    RED_PINK = "red_pink"
    VIVID_YELLOW = "vivid_yellow"
    PALE_YELLOW = "pale_yellow"


def _classify_smoothie(image: np.ndarray) -> SmoothieType:
    """Classify smoothie colour type from center crop median LAB values."""
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    crop = image[cy - h // 6 : cy + h // 6, cx - w // 6 : cx + w // 6]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    med_a = float(np.median(lab[:, :, 1].astype(np.float32))) - 128.0
    med_b = float(np.median(lab[:, :, 2].astype(np.float32))) - 128.0

    if med_a > 8:
        return SmoothieType.RED_PINK
    elif med_b > 20:
        return SmoothieType.VIVID_YELLOW
    else:
        return SmoothieType.PALE_YELLOW


def _top_boundary(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (cols, top, bottom): occupied columns and their first/last fg rows.

    ``top`` and ``bottom`` are full-width arrays indexed by column. Returns None
    if fewer than two columns contain foreground.
    """
    h = mask.shape[0]
    fg = mask > 0
    cols = np.where(fg.any(axis=0))[0]
    if cols.size < 2:
        return None
    top = np.argmax(fg, axis=0)                      # first True from the top
    bottom = h - 1 - np.argmax(fg[::-1, :], axis=0)   # first True from the bottom
    return cols, top, bottom


def top_edge_roughness(mask: np.ndarray, win_frac: float = 0.06) -> float:
    """Quantify how 'squiggly' the top edge of a filled ROI mask is, in pixels.

    Measures *high-frequency* roughness: the per-column top boundary is detrended
    by subtracting a moving average (window ``win_frac`` of the occupied width),
    and the standard deviation of that residual is returned. This isolates true
    column-to-column sawtooth from benign low-frequency shape — a smoothly curved
    cup shoulder or a tilted surface scores near 0, while a jagged tan top scores
    high. (Deviation-from-a-straight-line metrics fail here because the rounded
    cup shoulders inflate them even when the edge is perfectly smooth.)

    A clean top scores ~0.5–2 px; a jagged sawtooth top scores ~3.5–6 px.
    Returns 0.0 for a degenerate mask.
    """
    boundary = _top_boundary(mask)
    if boundary is None:
        return 0.0
    cols, top, _ = boundary
    y = top[cols].astype(np.float64)
    n = len(y)
    win = max(5, int(n * win_frac) | 1)   # odd window
    pad = win // 2
    kernel = np.ones(win) / win
    smooth = np.convolve(np.pad(y, pad, mode="edge"), kernel, mode="valid")[:n]
    return float(np.std(y - smooth))


def draw_container_overlay(image: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray:
    """Return a copy of image with the detected ROI boundary drawn in green."""
    vis = image.copy()
    if roi_mask is None or not np.any(roi_mask):
        return vis
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    return vis
