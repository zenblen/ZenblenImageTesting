# Zenblen Image Testing

Computer-vision QA for a smoothie vending rig. A fixed camera photographs each cup
on the blender seat; YOLO-seg models judge blend quality, spills, and fill level.

**Working here as a coding agent?** Start with
[AGENTS.md](AGENTS.md) / [CLAUDE.md](CLAUDE.md), which point into [docs/](docs/).

| Folder | Purpose |
|--------|---------|
| `active_pipeline/` | Deployable YOLO runtime (blend/chunk + spill) + weights. Ships to the Jetson |
| `training/` | Labeling UIs, datasets, `train_multi.py`, training runs |
| `experimentation/` | Exploratory studies; not deployed |
| `docs/` | Canonical documentation |
| `scripts/` | Repo maintenance helpers |

## Documentation

| Doc | Covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Components, data flow, entry points, dependency boundaries |
| [docs/repository-map.md](docs/repository-map.md) | Where each kind of change belongs |
| [docs/testing.md](docs/testing.md) | Validation commands and real coverage |
| [docs/agent-guidelines.md](docs/agent-guidelines.md) | Rules for coding agents |
| [docs/liquid-level.md](docs/liquid-level.md) | Underfill-flag spec and calibration |
| [REMOTE_TRAIN.md](REMOTE_TRAIN.md) | GPU training via push/pull |
| [training/labeling/README.md](training/labeling/README.md) | Detailed labeling workflows |

## Runtime (Jetson / inference)

```bash
cd active_pipeline
python run.py --pipeline blend --image <img.jpg>
python run.py --pipeline spill --image <img.jpg>
```

A directory argument runs batch mode and writes
`outputs/<timestamp>__<pipeline>/comparison.csv`.

## Label + train

```bash
cd training
python labeling/app_multi.py                      # label            (port 5001)
python labeling/export_multi.py --mode chunk      # export dataset
python train_multi.py --mode chunk                # train (CPU; MPS segfaults)

# deploy — both copies are required:
cp runs/chunk-seg/<run>/weights/best.pt ../active_pipeline/checkpoints/yolo_chunk_seg.pt
cp runs/chunk-seg/<run>/weights/best.pt checkpoints/yolo_chunk_seg.pt   # labeling mirror
```

Logo weights stay under `training/checkpoints/` (not shipped in `active_pipeline`).

## Validation

```bash
pytest                                   # test suite
ruff check .                             # lint (pip install ruff)
python scripts/check_agent_docs.py       # agent-doc drift check
```

Use the conda base interpreter (`/opt/miniconda3/bin/python`) — it is the only
environment with `torch` + `ultralytics`.
