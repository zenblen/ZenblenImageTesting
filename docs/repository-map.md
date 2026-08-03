# Repository map

Where things live, where changes belong, and what to run. Read this instead of
exploring. See also: [architecture](architecture.md) ·
[agent guidelines](agent-guidelines.md) · [testing](testing.md)

**Scale check:** ~5300 tracked files, but only **51 Python files / ~6,600 lines**.
Everything else is training data. List source with `git ls-files '*.py'`.

---

## Directories

| Path | Purpose | Scan? |
|---|---|---|
| `active_pipeline/` | **Deployable runtime.** `run.py` + `smoothie_cv/` + deploy weights. Ships to the Jetson | Yes |
| `active_pipeline/smoothie_cv/detection/` | Per-target detectors (`yolo`, `chunk`, `chunk_yolo`, `spill`) + `common.py` helpers | Yes |
| `active_pipeline/smoothie_cv/pipelines/` | `base.py` contracts, `blend.py`, `spill.py` | Yes |
| `active_pipeline/smoothie_cv/scoring/` | `metrics.py` — blend score, overlays | Yes |
| `active_pipeline/smoothie_cv/tests/` | The only test suite | Yes |
| `training/labeling/` | Flask labeling UIs, `db.py`, exporters, prediction staging | Yes |
| `training/train_multi.py`, `train_cls.py` | Training entry points | Yes |
| `experimentation/` | Exploratory studies; not deployed, nothing imports it | Only if asked |
| `docs/` | Canonical documentation (this directory) | Yes |
| `scripts/` | Repo maintenance helpers | Yes |
| `training/labeling/datasets/` | **5207 files.** Exported YOLO datasets. Committed on purpose — `REMOTE_TRAIN.md` needs them | **Never** |
| `training/labeling/data/` | Raw image pool (gitignored) | **Never** |
| `training/runs/`, `training/checkpoints/` | Run artifacts + weight mirror | **Never** |
| `**/outputs/` | Pipeline outputs (gitignored) | **Never** |

## Where common changes belong

| Task | Edit | Then run |
|---|---|---|
| Tune a threshold / add a flag | `smoothie_cv/config.py` (`Config`) | `pytest active_pipeline/smoothie_cv/tests/ -q` |
| Change blend scoring | `smoothie_cv/scoring/metrics.py` | `pytest -q -k Blend` |
| New/changed detector | `smoothie_cv/detection/<target>.py`, export in `detection/__init__.py` | `pytest -q` |
| New pipeline | `smoothie_cv/pipelines/<name>.py` + `run.py:load_pipeline` + `PIPELINE_NAMES` | `python run.py --pipeline <name> --image <img>` |
| New result field | `pipelines/base.py` dataclass, then `run.py` record **and** `write_run_manifest` `fieldnames` | `python run.py --pipeline blend --image <dir>` |
| Labeling UI change | `training/labeling/app_*.py` (+ `templates/`, `static/`) | Launch the app; no test coverage |
| Schema / query change | `training/labeling/db.py` | Manual — inspect `labels.db` |
| Dataset export rules | `training/labeling/export_multi.py` (or `export_cls.py`) | `python labeling/export_multi.py --mode <m>` |
| New training mode | `training/train_multi.py:MODE_CFG` | `python train_multi.py --mode <m>` |
| Docs / agent rules | `docs/*.md` | `python scripts/check_agent_docs.py` |

### Gotchas that cost real debugging time

- **`run.py:write_run_manifest`** has a hard-coded `fieldnames` list with
  `extrasaction="ignore"` — new metadata is *silently dropped* from
  `comparison.csv`. Add the key there too.
- **`BlendPipeline.analyze`** substitutes an all-255 full-frame mask when
  `roi_mask is None`. Any geometry check must first confirm a real ROI arrived, or
  a no-detection frame reads as a full cup.
- **Weights are mirrored** in two places. Deploying means copying to
  `active_pipeline/checkpoints/` *and* `training/checkpoints/`.
- **Training runs on conda python**, not the default interpreter:
  `/opt/miniconda3/bin/python`. MPS segfaults on YOLO-seg; local training is CPU.

## Validation commands

From the repo root (config lives in `pyproject.toml`):

```bash
pytest -q                                        # full suite
pytest -q -k Blend                               # targeted
ruff check .                                     # lint (optional install)
python scripts/check_agent_docs.py               # agent-doc drift check
python -m compileall -q active_pipeline training  # syntax-only, no deps
```

Runtime smoke test:

```bash
cd active_pipeline && python run.py --pipeline blend --image <img.jpg>
```

Full details and environment setup: [testing.md](testing.md).

## Reference docs

| Doc | Covers |
|---|---|
| `README.md` | Project overview, quickstart |
| `REMOTE_TRAIN.md` | GPU training via push/pull |
| `docs/liquid-level.md` | Underfill-flag spec and calibration |
| `training/labeling/README.md` | Detailed labeling workflows (long) |
| `active_pipeline/README.md` | Jetson deploy notes |
