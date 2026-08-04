# Texture-blendedness experiment

Chunk-**independent** blendedness metrics. Instead of counting discrete chunks,
these measure how uniform the liquid colour is. Seeds/chia/blueberry skin are
small isolated speckle (normal recipe texture) and must NOT lower the score;
genuine unblended streaks/patches are large and contiguous, so they should.

Uses `yolo_standard_seg.pt` (liquid ROI) + `yolo_logo_seg.pt` (logo, excluded).
NO chunk model, NO labels.

## Methods (methods/*.py — one file each, all share the same interface)
Each exposes `NAME` and `score(image_bgr, roi_mask, logo_mask) -> (score_0_100, flag01)`.
- **dev_area** — LAB ΔE from a blurred local baseline → threshold → morph-open →
  score by the *flagged-area fraction* (border eroded to drop meniscus/rim).
- **entropy** — local Shannon entropy of quantised a*/b* over a seed-sized+ window.
- **robust_spread** — global robust colour dispersion (trimmed-std + MAD, chroma-weighted).
- **frequency** — multi-scale band-pass (DoG) mid-scale energy via masked convolution.
- **clusters** — k-means dominant colour; coherent off-colour area after size-filtering.

Add a new method by dropping another file in `methods/` with the same interface.

## Shared infra
- `common.py` — YOLO ROI/logo extraction (cached to `outputs/mask_cache/`),
  `scored_region()` border helper, `render_panel()` overlay.

## Run the comparison
```
/opt/miniconda3/bin/python experimentation/texture_blendedness/compare.py --n 100
/opt/miniconda3/bin/python experimentation/texture_blendedness/gen_report.py
open experimentation/texture_blendedness/outputs/comparison/index.html
```
`compare.py` runs every method on the same images (masks cached, so re-runs are
fast) and writes:
- `outputs/comparison/scores_all.csv` — one row per image, one column per method
- `outputs/comparison/panels/<rank>_<name>.jpg` — original | each method (outlined + score)
`gen_report.py` builds `index.html` — a sortable browser gallery (sort by any
method, by mean, or by "spread" = how much the methods disagree).

Flags: `compare.py --only dev_area entropy` restricts to some methods;
`--sort <NAME>` sets the ranking method.

## Single-method runner (legacy)
`run_experiment.py` runs just the original dev-area method with overlay modes
(`--overlay outline|heat|sidebyside`). `tune.py` sweeps dev-area params on a
curated set.
