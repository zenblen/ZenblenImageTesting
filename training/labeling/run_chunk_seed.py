"""Seed candidate chunk polygons from YOLO-chunk.

For each image this writes:
  data/polygons_chunk_seed/<file_id>.json
    {"width", "height", "shapes": [[[x, y], ...], ...]}

Uses the live blend path: YOLO-standard ROI → YOLO-chunk (full_filter) →
clip to ROI. Annotators correct the model's guess in app_multi (mode 4)
instead of free-drawing from scratch.

Also writes a browsable review gallery under ``--overlays-dir`` (default
``outputs/chunk_autolabel_review/``):
  overlays/<id>.jpg   original | ROI | chunk overlay
  flagged/            symlinks for images with ≥1 seed shape
  scores.csv          file_id, n_shapes, chunk_px, detector
  README.md

Resumable: images whose seed JSON already exists are skipped unless ``--force``.

After seeding, the top N images (default 500) go to ``labeling/priority/chunk.txt``,
ranked by seed-shape count, then total area, then largest chunk.

Run:
  python labeling/run_chunk_seed.py --force
  python labeling/run_chunk_seed.py --limit 50 --force
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container
from smoothie_cv.detection.chunk import detect_chunk
from smoothie_cv.scoring.metrics import overlay_mask

from labeling import db

MAX_POLY_POINTS = 8       # cap on vertices per seed shape (fewer = less to edit)
MIN_SEED_AREA = 40        # px; drop degenerate specks so seeds aren't noise
DEFAULT_PRIORITY_TOP = 500
DEFAULT_OVERLAYS = _TRAINING / "outputs" / "chunk_autolabel_review"


def _polygon_area(points: list[list[int]]) -> float:
    """Return a polygon's area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    return abs(sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )) / 2.0


def write_chunk_priority(top: int = DEFAULT_PRIORITY_TOP) -> int:
    """Rank cached chunk seeds and write the manual-label priority queue."""
    ranked: list[tuple[int, float, float, int]] = []
    for path in db.CHUNK_SEED_DIR.glob("*.json"):
        try:
            fid = int(path.stem)
            shapes = json.loads(path.read_text()).get("shapes", [])
            areas = [_polygon_area(p) for p in shapes]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if not areas:
            continue
        ranked.append((len(areas), sum(areas), max(areas), fid))

    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    selected = ranked[:max(0, top)]
    priority_path = db.ROOT / "priority" / "chunk.txt"
    priority_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{fid}  # {count} chunks; {total:.0f}px total; {largest:.0f}px largest"
        for (count, total, largest, fid) in selected
    ]
    priority_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(selected)


def mask_to_polygons(
    mask: np.ndarray, epsilon_frac: float = 0.01,
    max_points: int = MAX_POLY_POINTS, min_area: int = MIN_SEED_AREA,
) -> list[list[list[int]]]:
    """Every external contour of a binary mask -> simplified [[x, y], ...] each."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys: list[list[list[int]]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        eps = epsilon_frac * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        while len(approx) > max_points and eps < peri:
            eps *= 1.3
            approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) >= 3:
            polys.append([[int(x), int(y)] for [[x, y]] in approx])
    return polys


def _label(img: np.ndarray, lines: list[str], color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16 + 18 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


def _roi_strip(img: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    vis = img.astype(np.float32)
    vis[roi_mask == 0] *= 0.30
    vis = vis.astype(np.uint8)
    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis


def write_overlay(
    img: np.ndarray,
    roi_mask: np.ndarray,
    chunk_mask: np.ndarray,
    fid: str,
    n_shapes: int,
    detector: str,
    out_path: Path,
) -> None:
    """[ original | ROI | chunk detection ] triptych for manual validation."""
    px = int((chunk_mask > 0).sum())
    verdict = "CHUNKS" if px else "clean"
    color = (80, 160, 255) if px else (0, 200, 0)
    orig = _label(img.copy(), [fid], (255, 255, 255))
    roi_vis = _label(_roi_strip(img, roi_mask), ["ROI (standard)"], (0, 255, 0))
    det = overlay_mask(img, chunk_mask, color=(0, 0, 255), alpha=0.55)
    det = _label(det, [f"{verdict}  {n_shapes} shapes  {px}px", f"det={detector}"], color)
    gap = np.full((img.shape[0], 6, 3), 40, np.uint8)
    panel = np.hstack([orig, gap, roi_vis, gap, det])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def process_image(
    path: Path, cfg: Config, epsilon_frac: float, overlays_dir: Path | None,
) -> dict:
    """Run YOLO-chunk (ROI-clipped) on one image; write seed JSON + optional overlay.
    Returns a scores row dict."""
    fid = path.stem
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"could not read image: {path}")

    roi_mask, _bbox = detect_container(img, cfg, prefer="yolo")
    chunk_mask, detector = detect_chunk(img, roi_mask, cfg)

    h, w = img.shape[:2]
    shapes = mask_to_polygons(chunk_mask, epsilon_frac)
    seed = {"width": int(w), "height": int(h), "shapes": shapes}
    (db.CHUNK_SEED_DIR / f"{fid}.json").write_text(json.dumps(seed))

    px = int((chunk_mask > 0).sum())
    if overlays_dir is not None:
        ov_path = overlays_dir / "overlays" / f"{fid}.jpg"
        write_overlay(img, roi_mask, chunk_mask, fid, len(shapes), detector, ov_path)
        flagged = overlays_dir / "flagged" / f"{fid}.jpg"
        clean = overlays_dir / "clean" / f"{fid}.jpg"
        flagged.parent.mkdir(parents=True, exist_ok=True)
        clean.parent.mkdir(parents=True, exist_ok=True)
        # replace existing symlink/file
        for link in (flagged, clean):
            if link.exists() or link.is_symlink():
                link.unlink()
        target = ov_path.resolve()
        if shapes:
            flagged.symlink_to(target)
        else:
            clean.symlink_to(target)

    return {
        "file_id": fid,
        "n_shapes": len(shapes),
        "chunk_px": px,
        "detector": detector,
        "verdict": "chunks" if shapes else "clean",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed chunk polygons from YOLO-chunk (+ review overlays)")
    parser.add_argument("--epsilon-frac", type=float, default=0.01,
                        help="approxPolyDP tolerance as fraction of perimeter (default 0.01)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N images (for a quick test batch)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess images even if a seed JSON already exists")
    parser.add_argument("--priority-top", type=int, default=DEFAULT_PRIORITY_TOP,
                        help=f"Put this many highest-scoring seeds in priority/chunk.txt "
                             f"(default {DEFAULT_PRIORITY_TOP})")
    parser.add_argument("--overlays-dir", type=Path, default=DEFAULT_OVERLAYS,
                        help=f"Review gallery root (default {DEFAULT_OVERLAYS})")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip writing the review gallery")
    args = parser.parse_args()

    db.ensure_dirs()
    cfg = Config()
    # Force YOLO-chunk for seeds
    cfg.chunk_detector_priority = ["yolo"]

    overlays_dir = None if args.no_overlays else args.overlays_dir
    if overlays_dir is not None:
        (overlays_dir / "overlays").mkdir(parents=True, exist_ok=True)
        (overlays_dir / "flagged").mkdir(parents=True, exist_ok=True)
        (overlays_dir / "clean").mkdir(parents=True, exist_ok=True)

    imgs = sorted(db.IMAGES_DIR.glob("*.jpg"))
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"seeding YOLO-chunk for {len(imgs)} images -> {db.CHUNK_SEED_DIR}")
    if overlays_dir:
        print(f"review gallery -> {overlays_dir}")

    rows: list[dict] = []
    done = failed = skipped = 0
    for i, p in enumerate(imgs, 1):
        seed_path = db.CHUNK_SEED_DIR / f"{p.stem}.json"
        if seed_path.exists() and not args.force:
            skipped += 1
            continue
        try:
            row = process_image(p, cfg, args.epsilon_frac, overlays_dir)
            rows.append(row)
            done += 1
            print(
                f"[{i}/{len(imgs)}] {p.stem}  {row['n_shapes']} shape(s)  "
                f"{row['chunk_px']}px  det={row['detector']}"
            )
        except Exception as e:  # noqa: BLE001 - keep the batch going
            failed += 1
            print(f"[{i}/{len(imgs)}] {p.stem}  ERROR {e}")

    print(f"done: {done} processed, {skipped} skipped (cached), {failed} failed")
    priority_count = write_chunk_priority(args.priority_top)
    print(f"wrote {priority_count} chunk-priority ids -> labeling/priority/chunk.txt")

    if overlays_dir is not None and rows:
        csv_path = overlays_dir / "scores.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        n_flag = sum(1 for r in rows if r["verdict"] == "chunks")
        n_clean = len(rows) - n_flag
        md = [
            "# Chunk auto-label review gallery",
            "",
            f"YOLO-standard ROI + YOLO-chunk (`full_filter`). n={len(rows)}",
            f"**flagged (seeds):** {n_flag}  ·  **clean:** {n_clean}",
            "",
            f"- Browse flagged: `{overlays_dir / 'flagged'}`",
            f"- All overlays: `{overlays_dir / 'overlays'}`",
            f"- Scores: `{csv_path.name}`",
            "",
            "Seeds written to `labeling/data/polygons_chunk_seed/` — open",
            "`python labeling/app_multi.py` → mode **4 · Chunk** to edit.",
            "Priority queue: `labeling/priority/chunk.txt`.",
            "",
            "For model-assisted approve/reject: after `predict_batch.py --mode chunk`,",
            "run `python labeling/app_review.py --mode chunk`.",
            "",
        ]
        (overlays_dir / "README.md").write_text("\n".join(md))
        print(f"gallery: {n_flag} flagged / {n_clean} clean → {overlays_dir}")


if __name__ == "__main__":
    main()
