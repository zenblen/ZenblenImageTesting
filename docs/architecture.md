# Architecture

Computer-vision QA for a smoothie vending rig. A fixed camera photographs each cup
on the blender seat; YOLO-seg models decide whether the drink is properly blended,
whether material spilled, and (shadow mode) whether the fill level is low.

See also: [agent guidelines](agent-guidelines.md) ·
[repository map](repository-map.md) · [testing](testing.md)

---

## Major components

| Component | Path | Role |
|---|---|---|
| **Runtime** | `active_pipeline/` | The only deployable unit. Ships to the Jetson Nano. Self-contained: imports nothing else in the repo |
| **Training** | `training/` | Labeling UIs, dataset export, model training, run artifacts |
| **Experiments** | `experimentation/` | Exploratory studies. Not deployed, nothing imports it |
| **Ops scripts** | `gpu_train.py`, `gpu_train.sh`, `smoothie_mqtt.py` | Remote GPU training driver; MQTT client that consumes the runtime |

## Dependency boundaries

```
                    smoothie_mqtt.py ──┐
                                       ├──> active_pipeline/smoothie_cv  (leaf)
                    experimentation/ ──┘
                    training/  ── (independent; own checkpoint mirror)
```

**One hard rule: `active_pipeline/` is a leaf.** It imports only stdlib, `cv2`,
`numpy`, and `ultralytics`. It must never import from `training/`,
`experimentation/`, or root scripts — it is copied to the Jetson on its own.

- `training/` is independent of `active_pipeline/`. It keeps its own mirror in
  `training/checkpoints/` so labeling tools can predict without the runtime.
- `experimentation/` may read from anywhere; nothing may import *it*.
- `smoothie_mqtt.py` injects `active_pipeline/` on `sys.path` and consumes
  `Config`, `detect_container`, `BlendPipeline`, `SpillPipeline`.

## Entry points

| Command | Entry | Purpose |
|---|---|---|
| `python run.py --pipeline blend --image X` | `active_pipeline/run.py` | Blend/chunk analysis |
| `python run.py --pipeline spill --image X` | `active_pipeline/run.py` | Spill detection |
| `python smoothie_mqtt.py` | root | MQTT-driven live inference |
| `python labeling/app_multi.py` | `training/` | Hand labeling UI, port 5001 |
| `python labeling/app_review.py --mode M` | `training/` | Model-assisted review UI, port 5002 |
| `python labeling/app_classify.py` | `training/` | Classification UI, port 5003 |
| `python labeling/export_multi.py --mode M` | `training/` | Build a YOLO-seg dataset |
| `python train_multi.py --mode M` | `training/` | Train a segmentation model |
| `python train_cls.py --task T` | `training/` | Train a classifier |

`run.py` accepts a file *or* a directory; a directory triggers batch mode and
writes `comparison.csv`.

## Runtime data flow

```
image (BGR, 480x640 portrait)
  │
  ├─ _classify_smoothie()            detection/common.py    -> RED_PINK | VIVID_YELLOW | PALE_YELLOW
  │                                  (run.py collapses these to two output folders: red_pink | yellow)
  │
  ├─ detect_container()              detection/__init__.py   -> ROI mask + bbox
  │    └─ detect_yolo()              detection/yolo.py       [yolo_standard_seg.pt]
  │         └─ get_yolo_roi()  highest-confidence instance, resized, fill_holes()
  │
  ├── BLEND pipeline                 pipelines/blend.py
  │     ├─ detect_chunk()            detection/chunk.py -> chunk_yolo.py [yolo_chunk_seg.pt]
  │     └─ compute_blend_score()     scoring/metrics.py      -> BlendResult
  │
  └── SPILL pipeline                 pipelines/spill.py      (full frame, no ROI)
        └─ detect_spill()            detection/spill.py      [yolo_spill_seg.pt]
                                                             -> SpillResult
```

Outputs land in `active_pipeline/outputs/<timestamp>__<pipeline>/` — per-image
`_result.json`, mask/ROI overlays, plus `comparison.csv`, `run_info.json`, and a
`README.md` summary.

### Shared contracts

- **Detectors**: `(image, config) -> (roi_mask: HxW uint8, bbox | None)`.
  `255` = inside. Add new detectors to this signature.
- **Pipelines**: implement `BlendPipeline.analyze(image, roi_mask=None)` from
  `pipelines/base.py`, returning `BlendResult` or `SpillResult`.
- **Config**: all tunables live on `smoothie_cv/config.py:Config`
  (CLI flags > `config.yaml` > dataclass defaults).
- **ROI cropping**: `roi.py:crop_to_roi()` / `paste_mask()` convert between
  full-frame and crop coordinates.

## Inference invariants

These are load-bearing. Violating them corrupts results silently.

1. **Score one image at a time.** Ultralytics rect-letterboxes a *batch* to a
   common shape. 16 of the corpus frames are landscape (480x640 instead of
   640x480), so one landscape frame in a batch rescales every other image in it —
   measured to change mask width on 223 of 1087 frames. `run.py` is safe
   (single-image). Any batching must group by frame shape.
2. **Calibrate through the same path you infer through.** Use
   `detection/yolo.py:get_yolo_roi()`, which applies `fill_holes()`. Thresholding
   the raw mask yields different geometry.
3. **Frame geometry is fixed** (bolted camera, fixed gasket seat). Seat line
   `bot_y` has sd ≈ 9 px, so pixel constants are valid — but they are tied to
   480x640 portrait and this camera pose.

## Training data flow

```
Zenblen Files API
  └─ labeling/download.py ─────> labels.db(files) + labeling/data/images/<id>.jpg
       │
       ├─ app_multi.py (hand)  ──┐
       ├─ predict_batch.py       │ -> predictions -> app_review.py (approve/reject)
       │                         ├──> labels.db(annotations, mode_status)
       └─ app_classify.py ───────┴──> labels.db(classifications)
             │
             ├─ export_multi.py --mode M ─> labeling/datasets/<M>_dataset/  (YOLO-seg)
             └─ export_cls.py   --task T ─> labeling/datasets/<T>_cls_dataset/
                    │
                    ├─ train_multi.py --mode M ─> runs/<M>-seg/<name>/weights/best.pt
                    └─ train_cls.py   --task T ─> runs/<T>-cls/<name>/weights/best.pt
                           │
                           └─ deploy: cp best.pt -> active_pipeline/checkpoints/
                                             (+ training/checkpoints/ mirror)
```

Only hand-labeled or human-approved annotations reach a dataset — model
predictions stay in `predictions` / `review_status` until approved.

`labels.db` is a tracked SQLite file. **Query it; never read it as text.**

## Models

| Weights | Trained by | Used by |
|---|---|---|
| `yolo_standard_seg.pt` | `--mode standard` | ROI / liquid mask (runtime + level flag) |
| `yolo_chunk_seg.pt` | `--mode chunk` | Unblended-chunk detection |
| `yolo_spill_seg.pt` | `--mode spill` | Spill detection |
| `yolo_logo_seg.pt` | `--mode logo` | Wordmark suppression (training-side only) |
| `best_cleaning.pt` | `train_cls.py --task cleandone` | Clean/dirty classifier — **not yet wired into runtime** |

Weights are binary: never read them, only copy and reference by path.
