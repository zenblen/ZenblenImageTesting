# Testing and validation

Run checks in escalating order and stop at the first failure. Every command below
was verified from the repo root. See also:
[agent guidelines](agent-guidelines.md) · [repository map](repository-map.md)

---

## Environment

All Python work uses the **conda base** interpreter — it is the only environment
with `torch` + `ultralytics`. On this machine `python` already resolves to it:

```bash
python -c "import sys; print(sys.executable)"   # -> /opt/miniconda3/bin/python
```

If `python` resolves elsewhere, use the explicit path `/opt/miniconda3/bin/python`.
Verified present in conda base: `cv2`, `numpy`, `ultralytics`, `torch`, `flask`,
`pytest`, `yaml`, `requests`.

- `ruff` is installed but is **not** in any `requirements.txt` — treat lint as a
  convenience, not a hard gate. Install with `pip install ruff`.
- No type checker is configured. There is no `mypy`/`pyright` setup; do not claim
  a type check ran. `python -m compileall` is the available static gate.
- Training needs CPU: **MPS segfaults on YOLO-seg.** GPU training goes through
  `REMOTE_TRAIN.md`.
- Model-running scripts need `active_pipeline/checkpoints/*.pt` present.

## 1. Targeted checks (seconds — do these first)

```bash
pytest -k Blend                       # one class
pytest -k "LiquidLevel or RoiCrop"    # a few classes
pytest active_pipeline/smoothie_cv/tests/test_pipeline.py::TestBlendPipeline -v
ruff check active_pipeline/smoothie_cv/detection/   # lint just what you touched
```

`pyproject.toml` sets `pythonpath = ["active_pipeline"]`, so `pytest` works from
the repo root. It also sets `norecursedirs` to keep collection away from the 5207
dataset files.

## 2. Package-level checks

```bash
pytest                                            # whole suite (9 tests, ~0.1s)
python -m compileall -q active_pipeline           # syntax, no deps needed
ruff check active_pipeline/
```

Runtime smoke test — exercises the real models end to end:

```bash
cd active_pipeline && python run.py --pipeline blend --image <img.jpg>
cd active_pipeline && python run.py --pipeline spill --image <img.jpg>
```

A directory argument triggers batch mode and writes
`outputs/<timestamp>__blend/comparison.csv`.

## 3. Full validation

```bash
pytest                                   # 9 passed
ruff check .                             # All checks passed
python scripts/check_agent_docs.py       # agent-doc drift
python -m compileall -q active_pipeline training experimentation
```

## Coverage reality

Be precise about what is and is not covered — do not imply more than this.

| Area | Coverage |
|---|---|
| `active_pipeline/smoothie_cv/` | 9 smoke tests in `smoothie_cv/tests/test_pipeline.py` — ROI crop/paste round-trip, blend pipeline contract, score range, mask shape/dtype |
| `training/labeling/` (3 Flask apps, `db.py`, exporters) | **None.** Validate manually by launching the app |
| `training/train_multi.py`, `train_cls.py` | **None** |
| `experimentation/` | **None** |
| Model accuracy | No automated check. Thresholds were tuned by measuring on real frames — see below |

Tests use synthetic arrays and do **not** load model weights, so they pass without
checkpoints and cannot catch a model regression.

## Validating a model-behavior change

Threshold and gate changes are empirical claims, not code changes.

1. Run before and after over a real frame directory; report counts.
2. **Score one image at a time.** Batching mixed-orientation frames through
   ultralytics rescales the whole batch — measured to change mask width on 223 of
   1087 frames. See [architecture.md](architecture.md#inference-invariants).
3. Calibrate through `detection/yolo.py:get_yolo_roi()`, the same function the
   runtime uses (it applies `fill_holes()`).
4. Report false-positive counts. Note explicitly when a rate is unmeasurable —
   e.g. the underfill flag has no true positives in the corpus, so its sensitivity
   is unknown ([liquid-level.md](liquid-level.md)).

## Adding a test

Add to `active_pipeline/smoothie_cv/tests/test_pipeline.py`, following the
existing helper style (`_solid_image`, `_patchy_image`, `_roi`). Build synthetic
masks rather than loading fixtures so tests stay dependency-light and fast.

## Known findings (accepted, not failures)

- `training/labeling/predict_cls.py:101` — `zip()` without `strict=`, silenced via
  a **file-scoped** ignore in `pyproject.toml`. If `targets` and the streamed
  ultralytics `results` ever differ in length, file_ids misalign with predictions.
  Deferred because fixing it changes prediction semantics. New `zip()` calls
  elsewhere are still flagged.
