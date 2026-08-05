"""Export the classification labeler's decisions into a YOLO-cls dataset.

Unlike export_multi.py (per-mode single-class YOLO-SEG datasets, one polygon
`.txt` per image), this writes a folder-per-class layout — what Ultralytics
classification (`yolo11n-cls.pt`) expects:

    datasets/cleandone_cls_dataset/
      train/dirty/*.jpg   train/clean/*.jpg
      val/dirty/*.jpg     val/clean/*.jpg
      test/dirty/*.jpg    test/clean/*.jpg

Class folders come from ``db.cls_labels(task)``, so each task uses its OWN
vocabulary — ``xytable`` exports ``{powder,no_powder}/`` instead.

No data.yaml — train_cls.py passes the dataset ROOT directory as `data=`.

Exported filenames are task-prefixed (``cleandone_4821.jpg``) so any file is
self-identifying even out of its folder.

Run:
  python labeling/export_cls.py                   # every task in db.CLS_TASKS
  python labeling/export_cls.py --task cleandone --val 0.15 --test 0.10
  python labeling/export_cls.py --task xytable
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

# Default output dir per task (under labeling/datasets/, separate from runtime checkpoints).
TASK_DIRS = {
    "cleandone": db.ROOT / "datasets" / "cleandone_cls_dataset",
    "xytable": db.ROOT / "datasets" / "xytable_cls_dataset",
}


def _split_ids(ids: list[int], val_frac: float, test_frac: float):
    """Deterministic train/val/test split by sorted file_id (no randomness).
    Same logic as export_multi._split_ids, applied per-label so both classes
    land in every split even on a small/imbalanced dataset."""
    n = len(ids)
    n_test = max(0, round(n * test_frac))
    n_val = max(0, round(n * val_frac))
    return ids[: n - n_val - n_test], ids[n - n_val - n_test : n - n_test], ids[n - n_test :]


def export_task(task: str, out: Path, val_frac: float, test_frac: float) -> None:
    category = db.TASK_CATEGORY[task]
    labels = db.cls_labels(task)
    # Wipe prior export first: the split is by sorted file_id + fraction (per
    # label), so adding images shifts boundaries — recreate clean to avoid a
    # stale file lingering in two splits (train/val/test leakage).
    for split in ("train", "val", "test"):
        if (out / split).exists():
            shutil.rmtree(out / split)
        for label in labels:
            (out / split / label).mkdir(parents=True, exist_ok=True)

    conn = db.connect()
    rows = conn.execute(
        """
        SELECT c.file_id, c.label
        FROM classifications c JOIN files f ON f.file_id = c.file_id
        WHERE c.task = ? AND f.category_name = ? AND f.downloaded = 1
        ORDER BY c.file_id
        """,
        (task, category),
    ).fetchall()
    if not rows:
        print(f"[{task}] no classified images — skipping")
        return

    by_label: dict[str, list[int]] = {label: [] for label in labels}
    missing = 0
    for r in rows:
        fid, label = r["file_id"], r["label"]
        if label not in by_label:
            continue  # defensive: ignore any stale/unknown label value
        if not (db.IMAGES_DIR / f"{fid}.jpg").exists():
            missing += 1
            continue
        by_label[label].append(fid)
    if missing:
        print(f"[{task}] skipped {missing}: image missing on disk")

    counts = {split: {label: 0 for label in labels} for split in ("train", "val", "test")}
    for label, ids in by_label.items():
        train, val, test = _split_ids(ids, val_frac, test_frac)
        for split, split_ids in (("train", train), ("val", val), ("test", test)):
            for fid in split_ids:
                src = db.IMAGES_DIR / f"{fid}.jpg"
                dst = out / split / label / f"{task}_{fid}.jpg"
                shutil.copyfile(src, dst)
                counts[split][label] += 1

    total = sum(sum(c.values()) for c in counts.values())
    print(f"[{task}] exported {total} images -> {out}")
    for split in ("train", "val", "test"):
        c = counts[split]
        print(f"[{task}]   {split}: " + " ".join(f"{label}={c[label]}" for label in labels))
    for label, ids in by_label.items():
        if len(ids) < 20:
            print(f"[{task}]   WARNING: only {len(ids)} '{label}' images — "
                  f"label more before training")
    counts_all = {label: len(ids) for label, ids in by_label.items()}
    if counts_all and max(counts_all.values()) > 3 * max(1, min(counts_all.values())):
        print(f"[{task}]   WARNING: class imbalance {counts_all} — "
              f"consider labeling more of the minority class")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export classification-labeler datasets (folder-per-class)")
    parser.add_argument("--task", choices=db.CLS_TASKS,
                        help="Export only this task (default: all)")
    parser.add_argument("--out", help="Override output dir (only with --task)")
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.10)
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(db.CLS_TASKS)
    for t in tasks:
        out = Path(args.out) if (args.out and args.task) else TASK_DIRS[t]
        export_task(t, out, args.val, args.test)


if __name__ == "__main__":
    main()
