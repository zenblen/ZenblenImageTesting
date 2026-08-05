"""Train a YOLO-seg model for a multi-mode labeler dataset (spill / logo / standard / chunk / unmixed).

Run from the ``training/`` directory (paths below are cwd-relative to training/):

    python labeling/export_multi.py --mode spill
    /opt/miniconda3/bin/python train_multi.py --mode spill

Runs save to ``training/runs/<mode>-seg/<name>/weights/{best,last}.pt``.

Deploy targets:
  standard / spill / chunk → ``../active_pipeline/checkpoints/``
  logo                     → ``checkpoints/yolo_logo_seg.pt`` (training-only)

Prereqs:
    - Conda base env has ultralytics + torch
    - Base checkpoint yolo11n-seg.pt (ultralytics auto-downloads if missing)
    - Dataset exported for the mode
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Resolve relative to this file so train works regardless of cwd.
_TRAINING = Path(__file__).resolve().parent
_REPO = _TRAINING.parent
_ACTIVE_CKPT = _REPO / "active_pipeline" / "checkpoints"
_TRAIN_CKPT = _TRAINING / "checkpoints"

MODE_CFG: dict[str, dict[str, str]] = {
    "standard": {
        "data":    str(_TRAINING / "labeling/datasets/smoothie_dataset_std/data.yaml"),
        "project": str(_TRAINING / "runs/standard-seg"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_standard_seg.pt"),
    },
    "spill": {
        "data":    str(_TRAINING / "labeling/datasets/spill_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/spill-seg"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_spill_seg.pt"),
    },
    "logo": {
        "data":    str(_TRAINING / "labeling/datasets/logo_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/logo-seg"),
        "deploy":  str(_TRAIN_CKPT / "yolo_logo_seg.pt"),
    },
    "chunk": {
        "data":    str(_TRAINING / "labeling/datasets/chunk_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/chunk-seg"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_chunk_seg.pt"),
    },
    "unmixed": {
        "data":    str(_TRAINING / "labeling/datasets/unmixed_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/unmixed-seg"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_unmixed_seg.pt"),
    },
    # Powder spilled on the XY table. Localised deposits on a steel plate --
    # segmentation, because whole-image classification could not see them
    # (a ~60x40px deposit survives a 224px resize as ~10x7px).
    "powder": {
        "data":    str(_TRAINING / "labeling/datasets/powder_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/powder-seg"),
        "deploy":  str(_TRAIN_CKPT / "yolo_powder_seg.pt"),
    },
    # Inverse-of-chunks experiment: class 'blended' = ROI - human chunks. Built
    # by experimentation/inverse_blend/build_inverse_dataset.py (NOT a hand mode,
    # so it is absent from db.MODES). See that script for the known caveat.
    "blended": {
        "data":    str(_TRAINING / "labeling/datasets/blended_dataset/data.yaml"),
        "project": str(_TRAINING / "runs/blended-seg"),
        "deploy":  str(_ACTIVE_CKPT / "yolo_blended_seg.pt"),
    },
}

BASE_MODEL = "yolo11n-seg.pt"
DEVICE = "cpu"


def _data_yaml_with_abs_path(data_yaml: Path) -> Path:
    """Rewrite ``path:`` to an absolute dataset root for Ultralytics."""
    root = data_yaml.resolve().parent
    names_line = "chunk"
    text = data_yaml.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("0:"):
            names_line = line.split(":", 1)[1].strip()
            break
    out = root / "_train_data.yaml"
    out.write_text(
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {names_line}\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-mode YOLO-seg model")
    ap.add_argument("--mode", required=True, choices=list(MODE_CFG),
                    help="which labeler mode / dataset to train on")
    ap.add_argument("--name", help="run name (default: <mode>-nano-v1)")
    ap.add_argument("--base", default=BASE_MODEL, help=f"base weights (default: {BASE_MODEL})")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=DEVICE, help="cpu | 0 | mps (mps unsupported for seg)")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision — workaround for CUDA "
                    "'illegal memory access' crashes seen on some GPU/driver combos")
    args = ap.parse_args()

    cfg = MODE_CFG[args.mode]
    run_name = args.name or f"{args.mode}-nano-v1"
    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — export the dataset first:\n"
            f"    cd training && python labeling/export_multi.py --mode {args.mode}"
        )
    train_yaml = _data_yaml_with_abs_path(data_yaml)

    from ultralytics import YOLO

    _ACTIVE_CKPT.mkdir(parents=True, exist_ok=True)
    _TRAIN_CKPT.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base)
    model.train(
        data=str(train_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=cfg["project"],
        name=run_name,
        exist_ok=False,
        amp=not args.no_amp,
    )

    # Re-validation is informational only — best.pt is already written by
    # model.train() above. A flaky Windows DataLoader worker here must not
    # crash the script and lose an otherwise-successful run.
    try:
        metrics = model.val()
        print(f"\n[{args.mode}] mAP50 (mask):    {metrics.seg.map50:.4f}")
        print(f"[{args.mode}] mAP50-95 (mask): {metrics.seg.map:.4f}")
    except Exception as e:
        print(f"\n[{args.mode}] (re-validation skipped after training succeeded: {e})")

    weights_dirs = sorted(Path(cfg["project"]).glob(f"{run_name}*/weights/best.pt"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    best = str(weights_dirs[0]) if weights_dirs else f"{cfg['project']}/{run_name}/weights/best.pt"
    print(f"[{args.mode}] Weights: {best}")
    print(f"[{args.mode}] To deploy: cp {best} {cfg['deploy']}")
    # Keep training/checkpoints mirrors for labeling predict/flag tools
    if args.mode != "logo":
        mirror = _TRAIN_CKPT / Path(cfg["deploy"]).name
        print(f"[{args.mode}] Also mirror for labeling: cp {best} {mirror}")


if __name__ == "__main__":
    main()
