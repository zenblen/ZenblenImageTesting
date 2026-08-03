# Agent instructions — active_pipeline/

Adapter. Canonical rules: **[../docs/agent-guidelines.md](../docs/agent-guidelines.md)**.
`AGENTS.md` and `CLAUDE.md` here are byte-identical by design.

**Begin every response with the marker 🟢 followed by a space.**

## What this directory is

The **only deployable unit** — copied to the Jetson Nano on its own. See
[../docs/architecture.md](../docs/architecture.md) for data flow.

**This package is a dependency leaf.** It may import stdlib, `cv2`, `numpy`, and
`ultralytics` only. Never import from `training/`, `experimentation/`, or root
scripts — they do not exist on the device.

## Layout

| Path | Role |
|---|---|
| `run.py` | CLI entry: `--pipeline blend\|spill --image <file\|dir>` |
| `smoothie_cv/config.py` | `Config` — all tunables. CLI > `config.yaml` > defaults |
| `smoothie_cv/detection/` | Detectors: `yolo`, `chunk`, `chunk_yolo`, `spill`, `common` |
| `smoothie_cv/pipelines/` | `base.py` contracts + `blend.py`, `spill.py` |
| `smoothie_cv/scoring/metrics.py` | Blend score, overlays |
| `smoothie_cv/tests/` | The repo's only test suite |
| `checkpoints/*.pt` | Deploy weights — binary, never read |

## Contracts to reuse, not reinvent

- Detector: `(image, config) -> (mask uint8 255=inside, bbox | None)`
- Pipeline: `analyze(image, roi_mask=None) -> BlendResult | SpillResult`
- New tunables go on `Config` with a behavior-preserving default

## Traps that have cost real time

- **`run.py:write_run_manifest`** has a hard-coded `fieldnames` list with
  `extrasaction="ignore"` — new metadata is *silently dropped* from
  `comparison.csv`. Add your key there too.
- **`BlendPipeline.analyze`** substitutes an all-255 full-frame mask when
  `roi_mask is None`. Confirm a real ROI arrived before any geometry check, or a
  no-detection frame scores as a full cup.
- **Score one image at a time.** Batching mixed-orientation frames through
  ultralytics rescales the batch and corrupts mask geometry.
- Deploying weights means copying to **both** `active_pipeline/checkpoints/` and
  `training/checkpoints/`.

## Validate

```bash
pytest -k Blend                                   # targeted (from repo root)
pytest                                            # full suite, 9 tests
ruff check active_pipeline/
python run.py --pipeline blend --image <img.jpg>   # smoke test, from this dir
```

Tests use synthetic arrays and never load weights — they cannot catch a model
regression. See [../docs/testing.md](../docs/testing.md).
