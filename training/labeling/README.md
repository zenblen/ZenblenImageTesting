# Labeling tools

## Multi-mode labeler (current) — `app_multi.py`

The from-scratch hand-labeling pipeline (see also the faster model-assisted
**review** pipeline below, once a mode has a trained model). Three independent
segmentation passes over the same image pool, each training its own single-class
YOLO11n-seg model:

| mode | key | segments | dataset | model weights |
|------|-----|----------|---------|---------------|
| standard | `1` | smoothie **inside** the cup | `labeling/datasets/smoothie_dataset_std/` | `checkpoints/yolo_standard_seg.pt` |
| spill | `2` | smoothie **outside** the cup | `labeling/datasets/spill_dataset/` | `checkpoints/yolo_spill_seg.pt` |
| logo | `3` | the zenblen logo/wordmark | `labeling/datasets/logo_dataset/` | `checkpoints/yolo_logo_seg.pt` |

Each mode is strictly separate: one source image labeled in all three modes
yields three separate image+label pairs, never a mixed-class file. The 200
previous container labels were migrated into `standard` mode.

**Label → export → train** (run from ``training/``):

```bash
# 1. Label. Amber/violet/green banner shows the active mode. Switch with 1/2/3.
python labeling/app_multi.py                       # http://127.0.0.1:5001
#    click = drop points · N = new shape · Enter = save · K = mark clean · S = skip
#    ← / → = prev / next; ← reaches ANY earlier image (even ones decided on a
#            previous run) so you can always go back and re-edit — save replaces
#            that image's shapes for the active mode.
#    Optional: labeling/priority/<mode>.txt (one file_id per line) is served
#    FIRST by /api/next — used to bump hard spill lookalikes to the front.

# 2. Export the per-mode dataset (single class, clean images -> empty labels).
python labeling/export_multi.py --mode spill       # or logo / standard / (omit = all)

# 3. Train that mode's YOLO-nano (conda env — MPS segfaults, runs on CPU).
/opt/miniconda3/bin/python training/train_multi.py --mode spill
#    -> runs/spill-seg/spill-nano-v1/weights/best.pt   (bump --name each retrain)

# 4. Deploy the weights (path printed at end of training).
cp runs/spill-seg/spill-nano-v1/weights/best.pt checkpoints/yolo_spill_seg.pt
```

Data lives in the shared `labels.db` (additive `annotations` / `mode_status`
tables).

---

## Model-assisted review — `predict_batch.py` + `app_review.py` (current)

A SECOND, separate pipeline that is FASTER than drawing from scratch once a mode
has a trained model. It runs the mode's YOLO-seg model over the raw images and
lets a human **Approve / Reject / Edit** each prediction; approvals flow into the
SAME training dataset the hand labeler feeds (via the unchanged `export_multi.py`).
This is human-in-the-loop pseudo-labeling — the human is the quality gate, so the
model's mistakes never silently become training labels.

Predictions live in their OWN tables (`predictions`, `review_status`) and do NOT
enter training until approved. Approval writes `annotations` (tagged
`source='model'`) + `mode_status='labeled'`; a rejected image is left undecided so
the hand labeler (`app_multi.py`) re-serves it (and it's pushed to
`priority/<mode>.txt` so it jumps that queue).

```bash
# 0. Deploy the mode's weights (or pass --weights a run's best.pt below):
cp runs/spill-seg/spill-nano-v1/weights/best.pt checkpoints/yolo_spill_seg.pt

# 1. Stage predictions over the raw (undecided-for-this-mode) images.
#    Conda python (needs ultralytics/torch); runs on CPU (MPS segfaults on seg).
/opt/miniconda3/bin/python labeling/predict_batch.py --mode spill
#    --weights runs/spill-seg/spill-nano-v1/weights/best.pt   # if not deployed
#    --conf 0.25   --limit 50 (quick trial)
#    Every processed image is staged 'pending' — including zero-detection ones,
#    so the reviewer can also catch false-negatives.

# 2. Review. Opens on the model prediction, pre-loaded as an editable polygon.
python labeling/app_review.py --mode spill            # http://127.0.0.1:5002
#    A = approve (as-is or after dragging vertices) -> into training
#    R = reject  (wrong) -> back to the hand labeler
#    K = clean   (no target here) -> empty negative sample
#    N new shape · D delete shape · X clear · Z undo · ←/→ prev/next · S skip
#    Queue is LOWEST-confidence-first by default (?sort=conf_asc) so you spend
#    effort where the model is weakest; ?sort=file for file_id order.

# 3. Export + train exactly as the hand pipeline (approved labels are included):
python labeling/export_multi.py --mode spill
/opt/miniconda3/bin/python training/train_multi.py --mode spill --name spill-nano-v2

# Ablation — prove the pseudo-labels help, not hurt, on the disjoint eval:
python labeling/export_multi.py --mode spill --source hand   # hand labels only
python labeling/export_multi.py --mode spill --source model  # model-approved only
```

**Why lowest-confidence-first / reject → hand:** approving only the model's
confident hits teaches it nothing on the tail (pale/tan cups, clipped wordmarks),
so we surface uncertain predictions first and route the model's failures to manual
labeling — that's where new signal comes from.

The review UI is READ-ONLY (no polygon editing): each prediction shows as a thin
contour + faint fill, judged with **A** accept / **R** reject / **S** skip. Accept
on a zero-detection image confirms it clean (empty negative).

Navigation is resume-friendly: on load (and after each Accept/Reject) it jumps to
the **first/next pending** image, so you can quit any time and come back exactly
where the unreviewed work is — decisions persist in `labels.db`. **← / →** step
one image at a time through the whole mode in file order (INCLUDING already-decided
ones), so you can go back and **change any past decision** — reversing a reject to
accept also pulls that image out of the hand-labeler priority queue. Skip leaves an
image pending (it resurfaces later). Default order is file order; `?sort=conf_asc`
gives a lowest-confidence-first triage pass instead.

### Machinery / no-smoothie filtering — `flag_smoothie_presence.py`

Machinery / empty-rig shots (blender interior, no cup) are NOT a separate
category in the Files API — every image is `UserGrab`/`CleanDone` — so category
filtering can't remove them. Instead, gate on the CONTAINER model: an image with
zero smoothie detections is flagged `no_smoothie` and excluded EVERYWHERE (review
queues, the hand labeler `/api/next`, and `export_multi.py`). Run it after each
`download.py` pull (uses the deployed `yolo_smoothie_seg.pt`):

```bash
/opt/miniconda3/bin/python labeling/flag_smoothie_presence.py
#   --conf 0.30   raise the presence threshold    --limit N   trial run
```

Idempotent: images where a smoothie IS found get any stale flag cleared, so
re-running after a better container model un-hides recovered cups. Measured on the
1,123-image pool: 31 flagged (verified genuine machinery — the spill model had
been false-firing "spill" on the hardware, which the gate now keeps out of the
dataset). LIMIT: a smoothie the container model itself misses could be
false-flagged; keep the threshold low (default 0.25 = any detection).

---

## Classification track (current) — `app_classify.py`

A FOURTH, self-contained pipeline: whole-image classification instead of
polygons. One label per image (no drawing), trained as a YOLO11n-cls model
rather than YOLO-seg. Shares the `files` registry + `data/images/` with the rest
of the tool but writes only to its own `classifications` table
(`labeling/db.py`) — the seg tables are untouched.

Each task is **independent**: its own Files-API category, its own label set, its
own dataset, its own checkpoint. Nothing is shared but the image registry.

| Task | Category | Labels | Hotkeys | Checkpoint |
|---|---|---|---|---|
| `cleandone` | `CleanDone` | `dirty` / `clean` | D / C | `best_cleaning.pt` |
| `xytable` | `XYTable` | `powder` / `no_powder` | P / N | `yolo_xytable_cls.pt` |

The three per-task registries that define a task live in `labeling/db.py`:
`CLS_TASK_LABELS` (vocabulary), `TASK_CATEGORY` (image source), `CLS_POSITIVE`
(which class `predict_cls.py` ranks by). The buttons/hotkeys/colours live in
`app_classify.TASK_UI`, validated against `db.cls_labels()` at launch.

```bash
# 1. Pull the task's category (download.py already filters by category).
python labeling/download.py --start '2026-07-07 00:00:00' \
                            --end   '2026-08-04 00:00:00' --category XYTable
#    XYTable also emits video/mp4; the default --type image/jpg excludes it.
#    Volume is modest — ~100 jpgs / 2 weeks, ~409 over 4 months. Widen the
#    range (in <=2-week chunks; the API 500s on multi-month windows) to build
#    up a few hundred.

# 2. Label. Per-label hotkeys · S = skip (no save) · Z = undo · ←/→ prev/next.
python labeling/app_classify.py                    # http://127.0.0.1:5003
#    ?task=xytable selects the task (default: first in db.CLS_TASKS); the
#    banner carries a switch link. ← reaches ANY earlier image to re-classify.

# 3. Export a folder-per-class dataset (no data.yaml — that's a cls-only layout).
python labeling/export_cls.py --task xytable
#    -> datasets/xytable_cls_dataset/{train,val,test}/{powder,no_powder}/*.jpg

# 4. Train (conda env; yolo11n-cls, imgsz 224).
/opt/miniconda3/bin/python train_cls.py --task xytable
#    -> runs/xytable-cls/xytable-nano-v1/weights/best.pt

# 5. Deploy (paths printed at end of training).
cp runs/xytable-cls/xytable-nano-v1/weights/best.pt checkpoints/yolo_xytable_cls.pt
cp runs/xytable-cls/xytable-nano-v1/weights/best.pt ../active_pipeline/checkpoints/yolo_xytable_cls.pt

# 6. Optional — active-learning triage once a first model exists: rank the
#    undecided pool by P(powder) so the next labeling pass hits the signal.
/opt/miniconda3/bin/python labeling/predict_cls.py --task xytable --write-priority
```

### Adding another classification task

1. `db.py` — add to `CLS_TASKS`, `CLS_TASK_LABELS`, `CLS_POSITIVE`,
   `CLS_WEIGHTS`, `TASK_CATEGORY`.
2. `app_classify.TASK_UI` — title + one `{name, key, color}` per label. Keys
   `s`/`z`/`x` are reserved; a mismatch with `CLS_TASK_LABELS` raises at launch.
3. `export_cls.TASK_DIRS` and `train_cls.TASK_CFG` — one entry each.

No template or JS change is needed — `label_classify.html` renders whatever
label set the server hands it.

Neither task is wired into `active_pipeline/run.py` / `smoothie_cv` inference —
this track currently only produces `best.pt`. Runtime integration is a
deliberate follow-up once each classifier's accuracy is validated.

Note: these tasks deliberately do **not** use the `no_smoothie` /
`flag_smoothie_presence.py` gate — that gate keys on the smoothie/container
detector and would wrongly exclude the empty-station and bare-table shots that
CleanDone and XYTable images are made of.

---

## Shared data-prep — `download.py` / optional chunk seeds

```bash
# 1. Download every image/jpg in a time range (start NARROW to test).
python labeling/download.py --start '2026-06-29 00:00:00' --end '2026-06-30 00:00:00'
#    optional: --category CleanDone   --type image/jpg   --list-only
#    -> labeling/data/images/<file_id>.jpg   (+ files table in labels.db)

# 2. Optional: seed chunk-mode candidates (YOLO chunk detector).
/opt/miniconda3/bin/python labeling/run_chunk_seed.py --limit 20
#    -> data/polygons_chunk_seed/*.json
```

The API key goes in `.env` at repo root (git-ignored): `ZENBLEN_API_KEY=...`.

Standard / spill / logo modes are free-draw (or use `predict_batch.py` +
`app_review.py` for model-assisted labeling). Chunk mode optionally loads seeds
from `run_chunk_seed.py`.
