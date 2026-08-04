# Nano patch — ROI instance selection

**One file. One function. Two lines of logic.**
`active_pipeline/smoothie_cv/detection/yolo.py` → `get_yolo_roi()`

Select the ROI mask by **area**, not confidence.

---

## Why

Confidence says how sure the model is that a region *is* smoothie — not how much
of the smoothie it *covers*. When the model splits the liquid into overlapping
instances, the most-confident one can be a fragment.

Measured on 1,088 frames:

- **46.3%** of frames return more than one instance
- On **0.6% (6 frames)** the most-confident instance is not the largest
- When it happens it is severe. Frame `225728`:

| instance | conf | height | area |
|---|---|---|---|
| **was selected** | **0.89** | 351 px | 82,503 |
| ignored | 0.31 | 227 px | 9,552 |
| ignored | **0.30** | **483 px** | **118,034** |

The ROI was truncated by ~30%, which produced a false underfill flag.

**This affects every pipeline**, not just the level flag — `detect_container()`
feeds the blend-score denominator, so a truncated ROI inflates the chunk-area
fraction.

---

## The change

Find this in `get_yolo_roi()`:

```python
    confs = result.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    raw = result.masks.data[idx].cpu().numpy()
```

Replace with:

```python
    masks = result.masks.data.cpu().numpy() > 0.5
    idx = int(masks.reshape(len(masks), -1).sum(axis=1).argmax())
    raw = result.masks.data[idx].cpu().numpy()
```

And update the docstring first line from
`"""Extract the highest-confidence instance mask ...`
to
`"""Extract the LARGEST instance mask ...`
with a note that selection is by area, not confidence, and why.

Nothing else changes: the resize, the `> 0.5` threshold, and `fill_holes()` all
stay exactly as they were.

### Cost on device

Moves each instance's low-res mask to CPU to count pixels. Instance counts are
1–3 and masks are ~160×160, so this is negligible next to the forward pass. If
you want to avoid the transfer entirely, the torch-side equivalent is:

```python
    idx = int((result.masks.data > 0.5).flatten(1).sum(dim=1).argmax())
```

Both give the same index; the numpy form is used above because it does not depend
on torch tensor-API details across JetPack builds.

---

## Why area is safe (and union is not)

Distribution over all 1,088 frames:

| policy | area p50 | area p99 | area max | height p50 |
|---|---|---|---|---|
| argmax-conf (old) | 118,752 | 136,326 | 156,096 | 483 |
| **largest-area (new)** | **118,793** | **136,326** | **156,096** | **484** |
| union of instances | 122,040 | 144,230 | 169,844 | 503 |

Largest-area is **distributionally identical** to the old behaviour — 0.03%
difference at the median, same p99, same max — so it changes nothing except the 6
truncated frames, where it is a clear improvement.

**Do not use the union.** It raises median height by 20 px, meaning it
systematically pulls in extra regions, and it would force a full recalibration.

---

## Verify on the Nano

```bash
cd active_pipeline
python -m pytest ../ -q                 # or: pytest -q from repo root
python run.py --pipeline blend --image <a-known-frame.jpg>
```

Expected: **9 passed**, and blend scores unchanged on normal frames.

Targeted check — if you have frame `225728` on the device:

```bash
python - <<'PY'
from ultralytics import YOLO
import numpy as np
from smoothie_cv.detection.yolo import get_yolo_roi
r = YOLO("checkpoints/yolo_standard_seg.pt")("225728.jpg", verbose=False)[0]
m = get_yolo_roi(r, r.orig_shape) > 0
ys = np.where(m.any(axis=1))[0]
print("height:", ys.max() - ys.min(), "area:", m.sum())
PY
```

Expected **height 483, area 119,308**. Before the patch it was height 351.

---

## Optional: level constants

If the underfill flag from `docs/liquid-level.md` is implemented, its constants
shift by less than a pixel and **do not require updating**:

| | before | after |
|---|---|---|
| gated n | 681 | 682 |
| median height | 483.0 | 483.0 |
| sd | 30.74 | 30.31 |
| cutoff (median − 1.5σ) | 436.9 | **437.5** |
| frames flagged | 11 | **10** |

If you want the tighter value, set `CUTOFF_H = 437.5` in
`smoothie_cv/detection/level.py`. Leaving it at 436.9 is also correct — the
difference does not change any verdict in the 1,088-frame set.

---

## Not fixed by this patch

The other 10 flags are **not** truncation artifacts. They are full cups carrying
thick foam heads, sitting at 419–435 px against the 437 cutoff. The mask excludes
foam by the labeling standard, so liquid-only height reads short on foamy recipes.
That is a separate, known limitation — see `docs/liquid-level.md` §4.
