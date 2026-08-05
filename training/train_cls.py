"""Train a YOLO-cls classifier for a classification-labeler dataset.

Tasks (see labeling/db.py CLS_TASK_LABELS):
    cleandone -> dirty | clean          on 'CleanDone' photos
    xytable   -> powder | no_powder     on 'XYTable' photos

Run from the ``training/`` directory (paths below are cwd-relative to training/):

    python labeling/export_cls.py --task cleandone
    /opt/miniconda3/bin/python train_cls.py --task cleandone
    /opt/miniconda3/bin/python train_cls.py --task xytable

Runs save to ``training/runs/<task>-cls/<name>/weights/{best,last}.pt``.

Deploy target: ``../active_pipeline/checkpoints/`` (labeling + inference not yet
wired to it — see plan; training/checkpoints/ mirror is for parity with the
seg pipeline's predict/flag tools if a review pass is added later).

Standalone from train_multi.py: classification uses a folder-per-class dataset
ROOT (no data.yaml, no polygon/nc/names assumptions) and reports top1/top5
accuracy instead of seg mAP, so it isn't a --mode fitting into MODE_CFG.

Prereqs:
    - Conda base env has ultralytics + torch
    - Base checkpoint yolo11n-cls.pt (ultralytics auto-downloads if missing)
    - Dataset exported for the task (labeling/export_cls.py)
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Resolve relative to this file so train works regardless of cwd.
_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
_ACTIVE_CKPT = _REPO / "active_pipeline" / "checkpoints"
_TRAIN_CKPT = _TRAINING / "checkpoints"

TASK_CFG: dict[str, dict[str, str]] = {
    "cleandone": {
        "data":    str(_TRAINING / "labeling/datasets/cleandone_cls_dataset"),
        "project": str(_TRAINING / "runs/cleandone-cls"),
        "deploy":  str(_ACTIVE_CKPT / "best_cleaning.pt"),
    },
    "xytable": {
        "data":    str(_TRAINING / "labeling/datasets/xytable_cls_dataset"),
        "project": str(_TRAINING / "runs/xytable-cls"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_xytable_cls.pt"),
    },
}

BASE_MODEL = "yolo11n-cls.pt"
DEVICE = "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-task YOLO-cls classifier")
    ap.add_argument("--task", required=True, choices=list(TASK_CFG),
                    help="which classification-labeler task / dataset to train on")
    ap.add_argument("--name", help="run name (default: <task>-nano-v1)")
    ap.add_argument("--base", default=BASE_MODEL, help=f"base weights (default: {BASE_MODEL})")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=DEVICE, help="cpu | 0 | mps (mps unsupported for seg; cls untested here)")
    args = ap.parse_args()

    cfg = TASK_CFG[args.task]
    run_name = args.name or f"{args.task}-nano-v1"
    data_dir = Path(cfg["data"])
    if not data_dir.exists() or not any((data_dir / "train").glob("*/*.jpg")):
        raise FileNotFoundError(
            f"{data_dir} has no train images — export the dataset first:\n"
            f"    cd training && python labeling/export_cls.py --task {args.task}"
        )

    from ultralytics import YOLO

    _ACTIVE_CKPT.mkdir(parents=True, exist_ok=True)
    _TRAIN_CKPT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base)
    model.train(
        data=str(data_dir.resolve()),  # cls dataset ROOT dir, not a yaml
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=cfg["project"],
        name=run_name,
        exist_ok=False,
    )

    # Re-validation is informational only — best.pt is already written by
    # model.train() above. Mirrors train_multi.py's non-fatal guard.
    try:
        metrics = model.val()
        print(f"\n[{args.task}] top1 accuracy: {metrics.top1:.4f}")
        print(f"[{args.task}] top5 accuracy: {metrics.top5:.4f}")
    except Exception as e:
        print(f"\n[{args.task}] (re-validation skipped after training succeeded: {e})")

    weights_dirs = sorted(Path(cfg["project"]).glob(f"{run_name}*/weights/best.pt"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    best = str(weights_dirs[0]) if weights_dirs else f"{cfg['project']}/{run_name}/weights/best.pt"
    print(f"[{args.task}] Weights: {best}")
    print(f"[{args.task}] To deploy: cp {best} {cfg['deploy']}")
    mirror = _TRAIN_CKPT / Path(cfg["deploy"]).name
    print(f"[{args.task}] Also mirror for labeling tools: cp {best} {mirror}")


if __name__ == "__main__":
    main()
