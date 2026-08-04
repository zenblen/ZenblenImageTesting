# press_assets/

Figures and a self-contained deck for talks about the vision stack.

`zenblen_models_deck.html` opens in any browser — no server, no login, no network.
Every image is inlined as a data URI.

## Figures

| File | What it shows |
|---|---|
| `fig1_pipeline_stages.png` | The four blend stages on one real failing cup (score 0.876) |
| `ex_roi.png` | `standard-seg` — liquid mask |
| `ex_chunk.png` | `chunk-seg` — unblended regions |
| `ex_spill.png` | `spill-seg` — material outside the cup |
| `ex_logo.png` | `logo-seg` — printed wordmark |
| `ex_blended.png` | `blended-seg` — the smooth region |
| `ex_cleandone.png` | `cleandone-cls` — clean vs dirty station |
| `ex_level.png` | Fill level — mask height against the gasket seat |

Each `ex_*` figure is **three held-out validation frames**, chosen by highest IoU
against the hand-labelled polygons and spaced apart in time so no two tiles are
near-duplicate frames. White outline is the prediction, green is the hand label.

## Regenerating

```bash
/opt/miniconda3/bin/python press_assets/make_model_examples.py /tmp/report.json
/opt/miniconda3/bin/python press_assets/make_level_fig.py <level_probe.json> 226246,222344,225709
```

`make_model_examples.py` also prints the per-model IoU numbers quoted in the deck.
Both scripts score **one image at a time** — batching mixed-orientation frames
through ultralytics rescales the batch and corrupts mask geometry
(see [../docs/liquid-level.md](../docs/liquid-level.md) §3).

The level script needs a probe JSON of `{"rows": [{id, top_y, bot_y, h, w_max}],
"median", "sd", "cutoff"}`; regenerate it with the recalibration snippet in
[../docs/liquid-level.md](../docs/liquid-level.md) §5.

Numbers in the deck are point-in-time and will drift as the datasets grow.
