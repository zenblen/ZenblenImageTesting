"""Run a trained YOLO-seg model over the raw images and stage its predictions
for human review (model-assisted labeling — the review pipeline's first stage).

For a given mode, this finds every image that is still RAW for that mode
(downloaded, no ``mode_status`` decision, and not yet predicted) and runs the
mode's YOLO-seg model on it. Each detected instance becomes a row in
``predictions`` (pixel-space polygon + confidence); every processed image gets a
``review_status='pending'`` row — INCLUDING images where the model found nothing,
so the reviewer can still catch false-negatives. Nothing here touches the
training tables; approval happens later in ``app_review.py``.

Predictions are pixel-space so they round-trip identically when approved into
``annotations`` (which is also pixel-space; export re-normalizes).

Run under the conda python (has ultralytics + torch); MPS segfaults on YOLO-seg
so inference is forced onto CPU, matching train_multi.py.

  /opt/miniconda3/bin/python labeling/predict_batch.py --mode spill
  /opt/miniconda3/bin/python labeling/predict_batch.py --mode logo --limit 50
  # point at a run's best.pt when the checkpoint isn't deployed yet:
  /opt/miniconda3/bin/python labeling/predict_batch.py --mode spill \
      --weights runs/spill-seg/spill-nano-v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db


def _simplify(points, eps_frac: float = 0.004):
    """Reduce a dense YOLO mask contour to an editable handful of vertices.

    YOLO ``masks.xy`` polygons can carry 100+ points, which are painful to drag
    in the review UI. ``approxPolyDP`` keeps the shape within ``eps_frac`` of the
    arc length. Returns a list of [int x, int y]; degenerate results (<3 pts)
    fall back to the raw points so nothing is silently dropped.
    """
    import numpy as np

    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 3:
        return [[int(x), int(y)] for x, y in pts]
    eps = eps_frac * cv2.arcLength(pts.reshape(-1, 1, 2), True)
    approx = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
    out = approx if len(approx) >= 3 else pts
    return [[int(x), int(y)] for x, y in out]


def _targets(conn, mode: str, limit: int | None,
             include_pending: bool = False) -> list[int]:
    """Images to (re-)predict for this mode: downloaded, not flagged, and NOT
    already human-validated (``mode_status IS NULL`` excludes hand-labeled and
    approved; the rejected status is likewise left alone).

    Default: only RAW images (never predicted). With ``include_pending`` the set
    also covers ``review_status='pending'`` images — un-validated predictions
    left by a PREVIOUS model — so a freshly trained model can re-predict them
    (the per-image ``DELETE FROM predictions`` in the loop replaces the stale
    rows; the review_status upsert keeps them pending for the reviewer).
    """
    rs_clause = ("(rs.file_id IS NULL OR rs.status = 'pending')"
                 if include_pending else "rs.file_id IS NULL")
    sql = (
        "SELECT f.file_id FROM files f "
        "LEFT JOIN mode_status ms ON ms.file_id = f.file_id AND ms.mode = ? "
        "LEFT JOIN review_status rs ON rs.file_id = f.file_id AND rs.mode = ? "
        "LEFT JOIN image_flags fl ON fl.file_id = f.file_id "
        f"WHERE f.downloaded = 1 AND ms.file_id IS NULL AND {rs_clause} "
        "  AND fl.file_id IS NULL "  # skip no_smoothie / machinery shots
        "ORDER BY f.file_id ASC"
    )
    rows = conn.execute(sql, (mode, mode)).fetchall()
    ids = [r["file_id"] for r in rows]
    return ids[:limit] if limit else ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=db.MODES)
    ap.add_argument(
        "--weights",
        default=None,
        help="weights file (default: db.MODE_WEIGHTS[mode]); pass a run's "
        "best.pt when the checkpoint isn't deployed",
    )
    ap.add_argument("--conf", type=float, default=0.25,
                    help="min detection confidence (ultralytics default 0.25)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N images (for a quick trial run)")
    ap.add_argument("--repredict", action="store_true",
                    help="also re-run un-validated 'pending' images left by a "
                    "previous model (replaces their stale predictions); "
                    "approved/rejected/hand-labeled images are never touched")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else db.MODE_WEIGHTS[args.mode]
    if not weights.exists():
        raise SystemExit(
            f"weights not found: {weights}\n"
            f"Deploy the mode's best.pt to {db.MODE_WEIGHTS[args.mode]} or pass "
            f"--weights runs/{args.mode}-seg/<run>/weights/best.pt"
        )

    from ultralytics import YOLO  # deferred: only this stage needs torch

    model = YOLO(str(weights))
    conn = db.connect()
    ids = _targets(conn, args.mode, args.limit, include_pending=args.repredict)
    kind = "raw+pending" if args.repredict else "raw"
    print(f"[{args.mode}] {len(ids)} {kind} images → predicting with {weights.name}")

    model_run = weights.name
    n_dets = n_empty = 0
    for i, fid in enumerate(ids, 1):
        img_path = db.IMAGES_DIR / f"{fid}.jpg"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Chunk mode: keep only detections inside the smoothie ROI (YOLO-standard),
        # matching the live detect_chunk full_filter path. Other modes stay full-frame.
        roi_mask = None
        if args.mode == "chunk":
            from smoothie_cv.config import Config
            from smoothie_cv.detection import detect_container
            roi_mask, _ = detect_container(img, Config(), prefer="yolo")

        result = model(img, verbose=False, device="cpu", conf=args.conf)[0]

        polys: list[tuple[list, float]] = []
        if result.masks is not None and len(result.masks) > 0:
            confs = result.boxes.conf.cpu().numpy()
            for j, xy in enumerate(result.masks.xy):
                pts = _simplify(xy)
                if len(pts) < 3:
                    continue
                if roi_mask is not None and not _poly_inside_roi(pts, roi_mask):
                    continue
                polys.append((pts, float(confs[j])))

        conn.execute("DELETE FROM predictions WHERE file_id = ? AND mode = ?",
                     (fid, args.mode))
        for pts, conf in polys:
            conn.execute(
                "INSERT INTO predictions (file_id, mode, polygon, confidence, "
                "model_run, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (fid, args.mode, json.dumps(pts), conf, model_run),
            )
        conn.execute(
            "INSERT INTO review_status (file_id, mode, status, reviewed_at) "
            "VALUES (?, ?, 'pending', datetime('now')) "
            "ON CONFLICT(file_id, mode) DO UPDATE SET "
            "status='pending', reviewed_at=datetime('now')",
            (fid, args.mode),
        )
        conn.commit()

        if polys:
            n_dets += 1
        else:
            n_empty += 1
        top = max((c for _, c in polys), default=0.0)
        print(f"[{i}/{len(ids)}] {fid}  dets={len(polys)}  top_conf={top:.2f}")

    print(f"done: {n_dets} with detections, {n_empty} empty (staged as pending)")


def _poly_inside_roi(pts: list, roi_mask) -> bool:
    """True if the polygon centroid lands inside the smoothie ROI."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = int(round(sum(xs) / len(xs)))
    cy = int(round(sum(ys) / len(ys)))
    h, w = roi_mask.shape[:2]
    if not (0 <= cx < w and 0 <= cy < h):
        return False
    return bool(roi_mask[cy, cx] > 0)


if __name__ == "__main__":
    main()
