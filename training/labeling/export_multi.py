"""Export per-mode YOLO-seg datasets from the multi-mode labeler.

Each mode is an INDEPENDENT single-class dataset (no mixed-class label files):
  standard -> datasets/smoothie_dataset_std/   class 0: smoothie
  spill    -> datasets/spill_dataset/          class 0: spill
  logo     -> datasets/logo_dataset/           class 0: logo
  chunk    -> datasets/chunk_dataset/          class 0: chunk
  unmixed  -> datasets/unmixed_dataset/        class 0: unmixed
  powder   -> datasets/powder_dataset/         class 0: powder

Exported filenames are mode-prefixed (``spill_4821.jpg`` / ``spill_4821.txt``)
so any file is self-identifying even out of its folder.

Status handling (mode_status):
  labeled -> one YOLO-seg line per polygon (multiple same-class shapes allowed)
  clean   -> EMPTY label file = a background/negative sample (teaches the model
             what is NOT a spill/logo; keeps false positives down)
  (no row / skipped) -> excluded entirely

Run:
  python labeling/export_multi.py                 # all three modes
  python labeling/export_multi.py --mode spill    # just one
  python labeling/export_multi.py --mode spill --val 0.15 --test 0.10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

# Default output dir per mode (under labeling/datasets/, separate from runtime checkpoints).
MODE_DIRS = {
    "standard": db.ROOT / "datasets" / "smoothie_dataset_std",
    "spill":    db.ROOT / "datasets" / "spill_dataset",
    "logo":     db.ROOT / "datasets" / "logo_dataset",
    "chunk":    db.ROOT / "datasets" / "chunk_dataset",
    "unmixed":  db.ROOT / "datasets" / "unmixed_dataset",
    "powder":   db.ROOT / "datasets" / "powder_dataset",
}


def _polygon_to_yolo(points: list[list[int]], width: int, height: int) -> str:
    """'0 x1 y1 ... xn yn' with coords normalized to [0,1]; '' if degenerate."""
    if len(points) < 3:
        return ""
    coords = []
    for x, y in points:
        coords.append(f"{x / width:.6f}")
        coords.append(f"{y / height:.6f}")
    return "0 " + " ".join(coords)


def _split_ids(ids: list[int], val_frac: float, test_frac: float):
    """Deterministic train/val/test split by sorted file_id (no randomness)."""
    n = len(ids)
    n_test = max(0, round(n * test_frac))
    n_val = max(0, round(n * val_frac))
    return ids[: n - n_val - n_test], ids[n - n_val - n_test : n - n_test], ids[n - n_test :]


def _image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) of a JPEG without a hard Pillow dependency at import time."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow required for export (pip install pillow)")
    with Image.open(path) as im:
        return im.width, im.height


def _model_approved_ids(conn, mode: str) -> set[int]:
    """file_ids promoted into training by the review pipeline (app_review.py) —
    i.e. review_status='approved' for this mode. Complete provenance for both
    labeled and clean decisions (the annotations.source column carries the same
    signal per-row for labeled shapes)."""
    return {
        r["file_id"]
        for r in conn.execute(
            "SELECT file_id FROM review_status WHERE mode = ? AND status = 'approved'",
            (mode,),
        )
    }


def export_mode(mode: str, out: Path, val_frac: float, test_frac: float,
                source: str | None = None) -> None:
    """Export one mode. ``source`` filters by provenance for ablation:
    'hand' = exclude model-approved (pseudo-label) images; 'model' = only them;
    None = all (default)."""
    class_name = db.MODE_CLASS_NAMES[mode]
    # Wipe prior images/labels first: the split is by sorted file_id + fraction,
    # so adding images shifts every boundary and a file_id can move between
    # splits. Copying over a stale export would leave the same mode_<fid> file in
    # two splits -> train/val/test LEAKAGE. Recreate the split dirs clean.
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    conn = db.connect()
    rows = conn.execute(
        """
        SELECT m.file_id, m.status
        FROM mode_status m JOIN files f ON f.file_id = m.file_id
        LEFT JOIN image_flags fl ON fl.file_id = m.file_id
        WHERE m.mode = ? AND m.status IN ('labeled', 'clean') AND f.downloaded = 1
          AND fl.file_id IS NULL   -- never export no_smoothie / machinery shots
        ORDER BY m.file_id
        """,
        (mode,),
    ).fetchall()
    if not rows:
        print(f"[{mode}] no labeled/clean images — skipping")
        return

    model_ids = _model_approved_ids(conn, mode) if source else set()
    items: list[tuple[int, str]] = []
    dropped_src = 0
    for r in rows:
        fid = r["file_id"]
        if source == "hand" and fid in model_ids:
            dropped_src += 1
            continue
        if source == "model" and fid not in model_ids:
            dropped_src += 1
            continue
        if not (db.IMAGES_DIR / f"{fid}.jpg").exists():
            print(f"[{mode}] skip {fid}: image missing on disk")
            continue
        items.append((fid, r["status"]))
    if source:
        print(f"[{mode}] provenance filter '{source}': dropped {dropped_src}, "
              f"kept {len(items)}")
    if not items:
        print(f"[{mode}] nothing exportable after validation")
        return

    ids = [fid for fid, _ in items]
    train, val, test = _split_ids(ids, val_frac, test_frac)
    split_map = {**{i: "train" for i in train}, **{i: "val" for i in val},
                 **{i: "test" for i in test}}

    counts = {"train": 0, "val": 0, "test": 0}
    empty = 0
    for fid, st in items:
        split = split_map[fid]
        src = db.IMAGES_DIR / f"{fid}.jpg"
        shutil.copyfile(src, out / "images" / split / f"{mode}_{fid}.jpg")

        lines: list[str] = []
        if st == "labeled":
            w, h = _image_size(src)
            for a in conn.execute(
                "SELECT polygon FROM annotations WHERE file_id = ? AND mode = ? ORDER BY id",
                (fid, mode),
            ):
                line = _polygon_to_yolo(json.loads(a["polygon"]), w, h)
                if line:
                    lines.append(line)
        # clean -> lines stays empty (background sample)
        label_path = out / "labels" / split / f"{mode}_{fid}.txt"
        label_path.write_text(("\n".join(lines) + "\n") if lines else "")
        if not lines:
            empty += 1
        counts[split] += 1

    # Relative path so a cloned repo trains without rewriting machine-local paths.
    (out / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    total = sum(counts.values())
    print(f"[{mode}] exported {total} images -> {out}")
    print(f"[{mode}]   train={counts['train']} val={counts['val']} test={counts['test']}")
    print(f"[{mode}]   {empty} empty label files (clean/background samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-mode YOLO-seg datasets")
    parser.add_argument("--mode", choices=db.MODES,
                        help="Export only this mode (default: all three)")
    parser.add_argument("--out", help="Override output dir (only with --mode)")
    parser.add_argument("--val", type=float, default=0.10)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument(
        "--source", choices=("hand", "model"), default=None,
        help="provenance ablation: 'hand' excludes model-approved (pseudo-label) "
        "images so you can retrain on hand labels only; 'model' keeps only "
        "review-approved images; default keeps all. Compare the two on the "
        "disjoint eval to confirm pseudo-labels help.",
    )
    args = parser.parse_args()

    modes = [args.mode] if args.mode else list(db.MODES)
    for m in modes:
        out = Path(args.out) if (args.out and args.mode) else MODE_DIRS[m]
        export_mode(m, out, args.val, args.test, source=args.source)


if __name__ == "__main__":
    main()
