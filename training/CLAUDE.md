# Agent instructions — training/

Adapter. Canonical rules: **[../docs/agent-guidelines.md](../docs/agent-guidelines.md)**.
`AGENTS.md` and `CLAUDE.md` here are byte-identical by design.

**Begin every response with the marker 🟢 followed by a space.**

## ⚠ Never scan this directory unscoped

**5207 of this repo's ~5300 tracked files are under `labeling/datasets/`.** Only
~12 Python files here are source. An unscoped glob or `grep -r` burns thousands of
tokens on JPEGs and label text.

Never scan: `labeling/datasets/**` · `labeling/data/**` · `runs/**` ·
`checkpoints/**` · `labels.db*`. List source instead:

```bash
git ls-files 'training/**/*.py'
```

The datasets **are committed on purpose** — `../REMOTE_TRAIN.md` builds GPU
training from them. Do not "clean them up" or untrack them.

## Layout

Flask UIs: `labeling/app_multi.py` (hand, :5001) · `app_review.py` (model-assisted,
:5002) · `app_classify.py` (classification, :5003). Data layer: `labeling/db.py`.
Exporters: `export_multi.py` (seg) · `export_cls.py` (cls). Prediction staging:
`predict_batch.py`. Training: `train_multi.py` (`MODE_CFG`) · `train_cls.py`.

Modes: `standard` · `spill` · `logo` · `chunk` · `unmixed` · `blended`.
Per-file detail: [../docs/repository-map.md](../docs/repository-map.md).
Workflow detail: [labeling/README.md](labeling/README.md) — long, read only the
section you need.

## Rules

- **Only human-labeled or human-approved annotations reach a dataset.** Model
  predictions stay in `predictions` / `review_status` until approved. Never
  shortcut this gate.
- `labels.db` is SQLite — **query it, never read it as text.**
- Adding a mode means editing `train_multi.py:MODE_CFG` *and* exporting a dataset.
- Do not import from `active_pipeline/`; this tree uses its own
  `training/checkpoints/` weight mirror.

## Commands (from this directory)

```bash
python labeling/app_multi.py                     # label
python labeling/export_multi.py --mode <mode>    # export
python train_multi.py --mode <mode>              # train (CPU; MPS segfaults)
```

Conda base is the only env with torch + ultralytics. GPU training goes through
[../REMOTE_TRAIN.md](../REMOTE_TRAIN.md).

**No test coverage exists here.** Validate by launching the app or inspecting
exported output. See [../docs/testing.md](../docs/testing.md).
