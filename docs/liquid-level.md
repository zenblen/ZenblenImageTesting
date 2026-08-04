# Liquid-level (underfill) flag — implementation spec

Flag cups whose liquid height is abnormally low. **No new model, no new labels,
no training** — pure geometry over the mask `yolo_standard_seg.pt` already
produces.

Calibrated on 1108 `UserGrab` frames, 2026-08-03, using single-image inference
(the path `run.py` actually uses).

---

## 1. The method

```python
mask   = yolo_standard_seg(frame)          # already in the pipeline
w_max  = max over rows of (mask pixels in that row)
top_y, bot_y = first / last row containing mask
height = bot_y - top_y

if w_max < 290 or bot_y < 560:   -> UNKNOWN   # not upright & seated, don't score
if height < 435:                 -> LOW
else:                            -> OK
```

### Constants — one cup class, 480x640 portrait frames

Re-measured through `detect_container()` after `get_yolo_roi()` switched to
largest-area instance selection (see [NANO_PATCH.md](NANO_PATCH.md)). The earlier
argmax-confidence values were median 483.0 / σ 30.74 / cutoff 436.9 — within
2 px, and the flag set is the same size, so either calibration works. Use these.

| | value |
|---|---|
| **median height** | **484.0 px** |
| **σ** | **32.38** |
| **cutoff (median − 1.5σ)** | **435.4 px** |
| trips at | 10.0% below median |
| flags on the calibration set | 10 / 686 scored (1.46%) |
| coverage (scored, not `unknown`) | 686 / 1108 = **62%** (63% of portrait-only frames) |

`shortfall = 1 − height / 484` — log this and trend it. Read it as "N% below a
typical cup", **not** "N% empty": the cup rim sits at or above the frame edge on
full cups (`top_y` clips at 2 px in 30 frames), so there is no rim datum to
compute true percent-full against.

### Reference values — verify your rig matches before trusting the cutoff

| | p5 | p50 | p95 |
|---|---|---|---|
| `w_max` | 297 | **308** | 321 |
| `bot_y` (seat line) | 567 | **581** | 597 |
| `top_y` (liquid top) | 42 | **94** | 142 |
| `height` | 443 | **484** | 539 |

If your numbers fall outside these bands the calibration does not transfer — see
§5.

---

## 2. Why the two gates exist

Both are load-bearing; dropping either produces false alarms.

- **`w_max >= 290`** — rejects mask failures. Width is tightly unimodal
  (p1=290, p50=307, p99=324), so anything narrower is a broken segmentation.
- **`bot_y >= 560`** — rejects tipped cups. Without it, **21 of 32 flags are
  tipped cups**, not underfills. Costs coverage (99% → 63%) and it is worth it.

Log the `unknown` reason rather than discarding: off-seat cups are a real fault
condition you do not currently detect.

---

## 3. ⚠ Never batch mixed-orientation frames

**16 of 1108 frames are landscape (480x640) instead of portrait (640x480).**
Ultralytics rect-letterboxes a batch to a common shape, so one landscape frame
rescales every other image in its batch. Effect measured directly: batching in
groups of 16 changed `w_max` on **223 / 1087 frames** (mean +15.7 px, max 85) and
manufactured a phantom second "cup size" cluster at `w_max` ≈ 230 that does not
exist under single-image inference.

Consequences for implementation:

1. **`run.py` is safe** — it processes one image at a time. Keep it that way.
2. If you ever add batching for throughput, **group by frame shape** or the
   constants above silently stop applying.
3. **Skip non-portrait frames.** Assert `mask.shape == (640, 480)` before
   scoring; 16 landscape frames exist in the corpus and their geometry constants
   are different.

This is also why there is **one** cup class here and not two.

---

## 4. Known limits

- **Sensitivity floor ~10%.** Anything subtler is inside the noise. This is an
  anomaly flag, not a volume gauge.
- **Sensitivity is UNMEASURED.** There are **zero real underfills** in the 1108
  frames. All 10 flags are false — full cups with thick foam heads. You know the
  rule rarely cries wolf; you do **not** know it catches a real shortfall.
- **Foam.** The mask excludes foam by the labeling standard, so `height` measures
  liquid under the head. Pale/foamy recipes read ~5% lower than vivid ones, so
  they trip earlier. `_classify_smoothie()` (already called in `run_single()`)
  gives the pale/vivid axis if you later want a per-shade baseline.
- **At 1.5σ you get ~10x more flags than at 2σ** (10 vs 1). With no true positives
  in the data, every extra flag is currently a false one. `median − 2σ = 419.2`
  if you want the conservative variant.

### Measured dead — do not retry
- **Logo as a scale reference**: corr(cup `w_max`, logo width) = **−0.011**,
  logo width CV 0.58. The wordmark wraps a cylinder, so apparent width tracks
  rotation, not scale.
- **Background differencing** against a median of 396 bare-deck `CleanDone`
  frames: contents-top pinned at row 0 on all 10 test frames. Clear headspace
  shows white steel through it.

---

## 5. Recalibration (run on the Jetson's own camera)

```python
# Usage: python recalibrate_level.py <dir-of-NORMAL-FILL-frames>
import sys, glob
import numpy as np
from ultralytics import YOLO
from smoothie_cv.detection.yolo import get_yolo_roi

model = YOLO("checkpoints/yolo_standard_seg.pt")
rows = []
for p in sorted(glob.glob(f"{sys.argv[1]}/*.jpg")):
    r = model(p, verbose=False)[0]              # ONE at a time -- see section 3
    m = get_yolo_roi(r, r.orig_shape) > 0       # same mask path as the runtime
    if not m.any() or m.shape != (640, 480):
        continue
    ys = np.where(m.any(axis=1))[0]
    rows.append((int(m.sum(axis=1).max()), int(ys.min()), int(ys.max())))

w = np.array([r[0] for r in rows]); t = np.array([r[1] for r in rows])
b = np.array([r[2] for r in rows])
print("w_max histogram -- expect ONE tight cluster:")
h, e = np.histogram(w, bins=30)
for c, lo in zip(h, e):
    if c:
        print(f"{lo:6.0f} {c:4d} " + "#" * int(c / 8))

W_MIN, B_MIN = 290, 560     # <-- update from the percentiles printed below
print(f"w_max p1={np.percentile(w,1):.0f} p50={np.percentile(w,50):.0f} "
      f"| bot_y p10={np.percentile(b,10):.0f} p50={np.percentile(b,50):.0f}")
ok = (w >= W_MIN) & (b >= B_MIN)
if ok.sum() < 50:
    sys.exit(f"only {ok.sum()} frames pass the gate -- need >=50 for a stable sd")
hgt = (b - t)[ok]
med, sd = np.median(hgt), hgt.std(ddof=1)
print(f"n={ok.sum()} median={med:.1f} sd={sd:.2f} "
      f"cutoff(1.5sd)={med - 1.5 * sd:.1f} coverage={100 * ok.mean():.0f}%")
```

Feed it **normal-fill frames only** — the baseline must describe healthy cups.

**To measure sensitivity** (the missing piece): stage pours at 100 / 85 / 70 / 55
/ 40% on the rig, ~25 frames. Half an hour, and it gives you a real calibration
curve plus a validation set.

---

## 6. Drop-in module

Save as `active_pipeline/smoothie_cv/detection/level.py`:

```python
"""Abnormally-low liquid-level flag from the standard-seg mask.

The rig is fixed (seat line sd ~9 px), so the gasket seat is a calibration
constant and fill level reduces to mask height. Constants are for 480x640
portrait frames on the current rig -- see docs/liquid-level.md section 5 to
recalibrate, and section 3 for why frames must be scored one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_SHAPE = (640, 480)    # portrait; landscape frames have different geometry
W_MIN = 290                 # below this the mask is broken (width is unimodal)
BOT_MIN = 560               # below this the cup is off-seat / tipped
MEDIAN_H = 484.0
CUTOFF_H = 435.4            # median - 1.5 * sd (sd = 32.38)


@dataclass(frozen=True)
class LevelResult:
    status: str                 # "ok" | "low" | "unknown"
    reason: str                 # why unknown, else ""
    height: int | None          # bot_y - top_y
    shortfall: float | None     # fraction below MEDIAN_H, 0.0 if at/above
    w_max: int
    bot_y: int


def assess_level(mask: np.ndarray) -> LevelResult:
    """Flag an abnormally low fill. Returns "unknown" (never "low") when the cup
    is not upright and seated -- a tipped cup yields a short mask that would
    otherwise read as underfilled. Callers should log "unknown" separately: it is
    mostly tipped/displaced cups, a real fault condition worth surfacing."""
    if mask.shape != FRAME_SHAPE:
        return LevelResult("unknown", f"frame shape {mask.shape}", None, None, 0, 0)

    m = mask > 0
    if not m.any():
        return LevelResult("unknown", "empty mask", None, None, 0, 0)

    rows = np.where(m.any(axis=1))[0]
    top_y, bot_y = int(rows.min()), int(rows.max())
    w_max = int(m.sum(axis=1).max())

    if w_max < W_MIN:
        return LevelResult("unknown", f"mask too narrow (w_max {w_max})",
                           None, None, w_max, bot_y)
    if bot_y < BOT_MIN:
        return LevelResult("unknown", f"off-seat (bot_y {bot_y})",
                           None, None, w_max, bot_y)

    height = bot_y - top_y
    shortfall = max(0.0, 1.0 - height / MEDIAN_H)
    status = "low" if height < CUTOFF_H else "ok"
    return LevelResult(status, "", height, shortfall, w_max, bot_y)
```

---

## 7. Wiring — three edits

### 7a. `smoothie_cv/pipelines/blend.py`

`analyze()` substitutes an all-255 full-frame mask when `roi_mask is None`.
**That mask scores as a full cup** (`w_max`=480, `bot_y`=639, height=639 → "ok"),
silently reporting a healthy level for a frame where no cup was detected. Capture
whether a real ROI arrived *before* the substitution:

```python
    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        have_roi = roi_mask is not None                          # <-- ADD
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        from smoothie_cv.detection.chunk import detect_chunk
        from smoothie_cv.detection.level import assess_level     # <-- ADD

        unblended, chunk_detector = detect_chunk(image, roi_mask, self.config)
        score = compute_blend_score(unblended, roi_mask)
        passed = score >= self.config.threshold

        level = assess_level(roi_mask) if have_roi else None      # <-- ADD

        return BlendResult(
            blend_score=score,
            passed=passed,          # NOTE: level deliberately does NOT affect this
            mask=unblended,
            pipeline_name=self.name,
            metadata={
                "chunk_detector": chunk_detector,
                "chunk_yolo_input": getattr(self.config, "chunk_yolo_input", "full_filter"),
                # --- ADD ---
                "level_status": level.status if level else "unknown",
                "level_reason": level.reason if level else "no roi mask",
                "level_height": level.height if level else None,
                "level_shortfall": round(level.shortfall, 3) if level and level.shortfall is not None else None,
            },
        )
```

`run.py` already writes `"metadata": result.metadata` per image, so these surface
in `<stem>_blend_result.json` with no further change.

### 7b. `run.py` — get the fields into `comparison.csv`

**Required for shadow mode, not optional.** `comparison.csv` has a hard-coded
`fieldnames` list with `extrasaction="ignore"`, so level fields are *silently
dropped* from the batch CSV — the one artifact you need to review a few hundred
cups. In `write_run_manifest()`:

```python
    if records:
        csv_path = run_dir / "comparison.csv"
        fieldnames = ["image", "pipeline", "blend_score", "passed", "threshold",
                      "runtime_ms",
                      "level_status", "level_height",
                      "level_shortfall", "level_reason"]           # <-- ADD
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:                                      # <-- REPLACE writerows
                md = r.get("metadata") or {}
                writer.writerow({**r, **{k: v for k, v in md.items()
                                         if k.startswith("level_")}})
```

### 7c. `smoothie_cv/tests/test_pipeline.py`

```python
class TestLiquidLevel:
    def _mask(self, w: int, bot_y: int, height: int) -> np.ndarray:
        m = np.zeros((640, 480), dtype=np.uint8)
        x0 = (480 - w) // 2
        m[max(0, bot_y - height):bot_y + 1, x0:x0 + w] = 255
        return m

    def test_normal_fill_is_ok(self):
        from smoothie_cv.detection.level import assess_level
        assert assess_level(self._mask(308, 581, 483)).status == "ok"

    def test_underfill_is_low(self):
        from smoothie_cv.detection.level import assess_level
        r = assess_level(self._mask(308, 581, 400))     # 400 < 435.4
        assert r.status == "low" and r.shortfall > 0.1

    def test_tipped_cup_is_unknown_not_low(self):
        from smoothie_cv.detection.level import assess_level
        r = assess_level(self._mask(308, 540, 483))     # off-seat
        assert r.status == "unknown" and "off-seat" in r.reason

    def test_broken_narrow_mask_is_unknown(self):
        from smoothie_cv.detection.level import assess_level
        assert assess_level(self._mask(230, 581, 483)).status == "unknown"

    def test_empty_mask_is_unknown(self):
        from smoothie_cv.detection.level import assess_level
        assert assess_level(np.zeros((640, 480), np.uint8)).status == "unknown"

    def test_landscape_frame_is_unknown(self):
        from smoothie_cv.detection.level import assess_level
        assert assess_level(np.ones((480, 640), np.uint8) * 255).status == "unknown"

    def test_full_frame_mask_never_reports_ok(self):
        """A no-ROI fallback mask must not read as a healthy full cup."""
        from smoothie_cv.config import Config
        from smoothie_cv.pipelines.blend import BlendPipeline
        res = BlendPipeline(Config()).analyze(np.zeros((640, 480, 3), np.uint8))
        assert res.metadata["level_status"] == "unknown"
```

### 7d. Verify

```bash
pytest smoothie_cv/tests/test_pipeline.py -v -k "LiquidLevel or Blend"
```

Then on a directory of real frames — verified over all 1108: **670 ok, 11 low
(1.62% of scored), 427 unknown**:

```bash
python run.py --pipeline blend --image <dir-of-frames>
```

---

## 8. Shadow mode → live

`passed` must stay independent of `level_status` until §5 gives a measured
true-positive rate. When you promote it, add `level_gate: bool = False` to
`Config` and gate on that — do not hard-wire it into `passed`.

## 9. Cost

One `sum`, one `max`, two `where` calls on a mask the pipeline already computes.
No extra weights, no new dependencies (numpy only).
