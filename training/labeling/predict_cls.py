"""Run a trained YOLO-cls classifier over undecided images and rank them by the
task's positive class (``db.CLS_POSITIVE``: cleandone -> dirty, xytable ->
powder) — active-learning triage: feed the ranked list into
``labeling/priority/<task>.txt`` so a human reviews the model's most-likely-
positive candidates first, where a weak/early model finds the most signal.

Does NOT write to `classifications` — this is a read-only scoring pass; the
human is still the label of record (mirrors predict_batch.py's separation of
model prediction from human approval, minus the review-queue machinery since
there's just one label per image here, not polygons to edit).

Run under the conda python (has ultralytics + torch); CPU, matching train_cls.py.

  /opt/miniconda3/bin/python labeling/predict_cls.py --task cleandone
  /opt/miniconda3/bin/python labeling/predict_cls.py --task cleandone \
      --ids 223743-229285 --weights runs/cleandone-cls/cleandone-nano-v1/weights/best.pt \
      --write-priority
  /opt/miniconda3/bin/python labeling/predict_cls.py --task xytable --write-priority
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db


def _parse_ids(spec: str) -> list[int]:
    """'223743-229285' -> range; '1,2,3' -> explicit list; combinable with commas."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def _targets(conn, task: str, ids: list[int] | None) -> list[int]:
    """Undecided, downloaded images for this task's category, optionally
    restricted to an explicit id set (still filtered to undecided+on-disk)."""
    category = db.TASK_CATEGORY[task]
    rows = conn.execute(
        "SELECT f.file_id FROM files f "
        "LEFT JOIN classifications c ON c.file_id = f.file_id AND c.task = ? "
        "WHERE f.downloaded = 1 AND f.category_name = ? AND c.file_id IS NULL",
        (task, category),
    ).fetchall()
    all_undecided = {r["file_id"] for r in rows}
    if ids is not None:
        wanted = set(ids) & all_undecided
    else:
        wanted = all_undecided
    return sorted(
        fid for fid in wanted if (db.IMAGES_DIR / f"{fid}.jpg").exists()
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score undecided images by P(positive class) for triage")
    ap.add_argument("--task", required=True, choices=db.CLS_TASKS)
    ap.add_argument("--weights", default=None,
                    help="default: db.CLS_WEIGHTS[task] (deployed checkpoint)")
    ap.add_argument("--ids", default=None,
                    help="restrict to this id set, e.g. '223743-229285' or '1,2,3' "
                    "(default: every undecided image for the task)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--write-priority", action="store_true",
                    help="write the ranked (highest P(positive) first) ids into "
                    "labeling/priority/<task>.txt, prepended before any existing "
                    "entries not already covered by this run")
    args = ap.parse_args()

    weights = Path(args.weights) if args.weights else db.CLS_WEIGHTS[args.task]
    if not weights.exists():
        sys.exit(f"weights not found: {weights} — train + deploy first")

    conn = db.connect()
    ids = _parse_ids(args.ids) if args.ids else None
    targets = _targets(conn, args.task, ids)
    if args.limit:
        targets = targets[: args.limit]
    if not targets:
        print(f"[{args.task}] nothing to score")
        return
    print(f"[{args.task}] scoring {len(targets)} images with {weights}")

    from ultralytics import YOLO
    model = YOLO(str(weights))
    # Class order comes from the checkpoint's own names, not our label tuple —
    # ultralytics indexes folder-per-class datasets alphabetically, so relying
    # on db.cls_labels() order here would silently rank the wrong class.
    positive = db.CLS_POSITIVE[args.task]
    pos_idx = next((i for i, name in model.names.items() if name == positive), None)
    if pos_idx is None:
        sys.exit(f"{weights} has classes {sorted(model.names.values())} — "
                 f"no '{positive}' class; wrong task's weights?")

    paths = [str(db.IMAGES_DIR / f"{fid}.jpg") for fid in targets]
    scored: list[tuple[int, float]] = []
    results = model.predict(paths, device="cpu", verbose=False, stream=True)
    for fid, r in zip(targets, results):
        p_pos = float(r.probs.data[pos_idx])
        scored.append((fid, p_pos))

    scored.sort(key=lambda t: t[1], reverse=True)
    print(f"[{args.task}] top 10 by P({positive}):")
    for fid, p in scored[:10]:
        print(f"  {fid}: {p:.3f}")

    if args.write_priority:
        path = db.ROOT / "priority" / f"{args.task}.txt"
        ranked_ids = [fid for fid, _ in scored]
        existing: list[int] = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line and line.isdigit() and int(line) not in ranked_ids:
                    existing.append(int(line))
        lines = [
            f"# predict_cls.py --task {args.task}: {len(ranked_ids)} images "
            f"ranked by P({positive}) desc",
        ] + [str(fid) for fid in ranked_ids]
        if existing:
            lines.append("# carried over from prior priority list")
            lines.extend(str(fid) for fid in existing)
        path.write_text("\n".join(lines) + "\n")
        print(f"[{args.task}] wrote {len(ranked_ids)} ranked + {len(existing)} carried-over ids -> {path}")


if __name__ == "__main__":
    main()
