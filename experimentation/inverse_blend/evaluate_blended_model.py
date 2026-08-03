"""Qualitative test of the trained 'blended' (inverse-chunk) YOLO-seg model.

The open question from day one: did the model learn something beyond the
label arithmetic (ROI - chunks), or does it just predict "the whole ROI is
blended" regardless of streaks/patches it was never trained to exclude?

Three groups, using file_ids already identified in this project:
  A. chunk-CLEAN but texture-score-LOW (streak candidates) -- if the model
     still predicts ~100% of ROI as blended here, that confirms the blind
     spot: no streak signal was ever in the labels, so none was learned.
  B. chunk-LABELED (real chunks) -- sanity check the model still correctly
     excludes the known chunk area (this is the thing it WAS trained to do).
  C. chunk-CLEAN, high texture score (genuinely good cups) -- control; should
     predict ~100% blended, correctly.

For each image: ROI area, predicted-blended area, predicted-blended as % of
ROI, and for group B, how much of the true chunk polygon got EXCLUDED by the
predicted blended mask (1.0 = perfectly punched out, matches training target).

Run (needs ultralytics/torch -> conda python):
  /opt/miniconda3/bin/python evaluate_blended_model.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texture_blendedness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_inverse_dataset as B  # noqa: E402
import common  # noqa: E402

REPO = common.REPO
WEIGHTS = REPO / "training" / "runs" / "blended-seg" / "blended-nano-v1-3" / "weights" / "best.pt"
DB = REPO / "training" / "labeling" / "labels.db"

GROUP_A = [225353, 224876, 224922, 225098, 225268, 225005, 225454, 225192,
          223491, 226363, 225216, 224939, 224076, 225479, 226830]  # streak candidates


def chunk_labeled_ids(n=8):
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_id FROM mode_status WHERE mode='chunk' AND status='labeled' "
        "ORDER BY file_id"
    ).fetchall()
    ids = [r["file_id"] for r in rows if (common.IMAGES_DIR / f"{r['file_id']}.jpg").exists()]
    return ids[:n]


def chunk_clean_good_ids(n=8, exclude=()):
    """chunk-clean ids NOT in the streak-candidate group (a rough 'good' control)."""
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_id FROM mode_status WHERE mode='chunk' AND status='clean' "
        "ORDER BY file_id"
    ).fetchall()
    ids = [r["file_id"] for r in rows
           if r["file_id"] not in exclude and (common.IMAGES_DIR / f"{r['file_id']}.jpg").exists()]
    return ids[:n]


def chunk_polys(conn, file_id):
    return [json.loads(r["polygon"]) for r in conn.execute(
        "SELECT polygon FROM annotations WHERE file_id=? AND mode='chunk' ORDER BY id",
        (file_id,))]


def predict_blended_mask(model, image, roi_shape):
    h, w = roi_shape
    r = model(image, verbose=False, device="cpu", conf=0.25)[0]
    mask = np.zeros((h, w), np.uint8)
    if r.masks is None:
        return mask
    for raw in r.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    group_b = chunk_labeled_ids(8)
    group_c = chunk_clean_good_ids(8, exclude=set(GROUP_A))

    print(f"{'group':6} {'file_id':8} {'roi_px':>9} {'pred_blend_px':>13} "
          f"{'pred_%_of_roi':>13} {'chunk_excluded':>14}")
    print("-" * 70)

    for group, ids in (("A-streak", GROUP_A), ("B-chunk", group_b), ("C-clean", group_c)):
        for fid in ids:
            path = common.IMAGES_DIR / f"{fid}.jpg"
            if not path.exists():
                print(f"{group:8} {fid:8}  (missing on disk)")
                continue
            image, roi, _logo = common.get_masks(path)
            h, w = roi.shape
            roi_px = int((roi > 0).sum())
            if roi_px < 500:
                print(f"{group:8} {fid:8}  (no usable ROI)")
                continue
            pred = predict_blended_mask(model, image, (h, w))
            pred_px = int((pred > 0).sum())
            pct = 100.0 * pred_px / roi_px

            chunk_excl = ""
            if group == "B-chunk":
                polys = chunk_polys(conn, fid)
                if polys:
                    chunk_mask = B._rasterize(
                        [[v for xy in p for v in xy] for p in polys], (h, w)) > 0
                    chunk_px = int(chunk_mask.sum())
                    excluded = int((chunk_mask & (pred == 0)).sum())
                    chunk_excl = f"{100.0 * excluded / max(1, chunk_px):.1f}%"

            print(f"{group:8} {fid:8} {roi_px:9d} {pred_px:13d} {pct:12.1f}% {chunk_excl:>14}")


if __name__ == "__main__":
    main()
