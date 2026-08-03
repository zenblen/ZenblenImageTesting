"""Visual panels for manually checking the trained 'blended' model.

For each image: draws the ROI outline (green) and, more importantly, the
region INSIDE the ROI the model predicts is NOT blended (red fill) — that is
the "hole" it thinks it found. For chunk-labeled images the TRUE chunk polygon
is also outlined (yellow) so you can see whether the model's red region lines
up with the real chunk or misses it.

Groups (same as evaluate_blended_model.py):
  A-streak  chunk-clean, texture-score flagged as likely streaky/unmixed
  B-chunk   chunk-labeled (real chunks) -- does the red region match the yellow?
  C-clean   chunk-clean control -- should show almost no red

Run (needs ultralytics/torch -> conda python):
  /opt/miniconda3/bin/python visualize_blended_model.py
Output: experimentation/inverse_blend/outputs/panels/*.jpg + outputs/index.html
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texture_blendedness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_inverse_dataset as B  # noqa: E402
import common  # noqa: E402
from evaluate_blended_model import (  # noqa: E402
    DB,
    GROUP_A,
    WEIGHTS,
    chunk_clean_good_ids,
    chunk_labeled_ids,
    chunk_polys,
    predict_blended_mask,
)

OUT = Path(__file__).resolve().parent / "outputs"
PANELS = OUT / "panels"


def render(image, roi, pred_mask, chunk_mask, label, pct):
    vis = image.copy()
    roi_b = roi > 0
    not_blended = roi_b & (pred_mask == 0)

    overlay = vis.copy()
    overlay[not_blended] = (0, 0, 255)  # red = model says NOT blended
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    roi_cnts, _ = cv2.findContours(roi_b.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, roi_cnts, -1, (0, 255, 0), 2)  # green ROI outline

    if chunk_mask is not None:
        c_cnts, _ = cv2.findContours(chunk_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, c_cnts, -1, (0, 255, 255), 2)  # yellow true chunk

    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, f"{label}  blended={pct:.1f}%", (8, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return vis


def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    PANELS.mkdir(parents=True, exist_ok=True)
    group_b = chunk_labeled_ids(8)
    group_c = chunk_clean_good_ids(8, exclude=set(GROUP_A))

    rows = []  # (group, fid, relpath, pct, chunk_excl)
    for group, ids in (("A-streak", GROUP_A), ("B-chunk", group_b), ("C-clean", group_c)):
        for fid in ids:
            path = common.IMAGES_DIR / f"{fid}.jpg"
            if not path.exists():
                continue
            image, roi, _logo = common.get_masks(path)
            h, w = roi.shape
            roi_px = int((roi > 0).sum())
            if roi_px < 500:
                continue
            pred = predict_blended_mask(model, image, (h, w))
            pct = 100.0 * (pred > 0).sum() / roi_px

            chunk_mask = None
            chunk_excl = ""
            if group == "B-chunk":
                polys = chunk_polys(conn, fid)
                if polys:
                    chunk_mask = B._rasterize(
                        [[v for xy in p for v in xy] for p in polys], (h, w)) > 0
                    chunk_px = int(chunk_mask.sum())
                    excluded = int((chunk_mask & (pred == 0)).sum())
                    chunk_excl = f"{100.0 * excluded / max(1, chunk_px):.1f}% excluded"

            vis = render(image, roi, pred, chunk_mask, f"{group} #{fid}", pct)
            fname = f"{group}_{fid}.jpg"
            cv2.imwrite(str(PANELS / fname), vis)
            rows.append((group, fid, f"panels/{fname}", pct, chunk_excl))
            print(f"  wrote {fname}  blended={pct:.1f}%  {chunk_excl}")

    sections = {}
    for group, fid, rel, pct, chunk_excl in rows:
        sections.setdefault(group, []).append((fid, rel, pct, chunk_excl))

    html = ["<html><head><meta charset='utf-8'><title>blended model check</title>",
           "<style>body{font-family:sans-serif;background:#14161a;color:#eee}",
           "h2{border-bottom:1px solid #444;padding-bottom:6px}",
           ".grid{display:flex;flex-wrap:wrap;gap:14px}",
           ".card{width:280px}img{width:100%;border-radius:4px}",
           ".cap{font-size:12px;color:#bbb;margin-top:4px}</style></head><body>",
           "<p>Red = model predicts NOT blended (the 'hole' it found). "
           "Green = ROI outline. Yellow = TRUE labeled chunk (group B only).</p>"]
    titles = {"A-streak": "A: streak candidates (chunk-clean, texture-flagged) "
                          "&mdash; red should appear if model catches streaks",
             "B-chunk": "B: real chunk-labeled images &mdash; red should "
                       "overlap the yellow outline",
             "C-clean": "C: clean control &mdash; red should be near-absent"}
    for group in ("A-streak", "B-chunk", "C-clean"):
        html.append(f"<h2>{titles.get(group, group)}</h2><div class='grid'>")
        for fid, rel, pct, chunk_excl in sections.get(group, []):
            html.append(
                f"<div class='card'><img src='{rel}'>"
                f"<div class='cap'>#{fid} — blended {pct:.1f}% {chunk_excl}</div></div>")
        html.append("</div>")
    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))
    print(f"\nwrote {OUT / 'index.html'}  ({len(rows)} panels)")


if __name__ == "__main__":
    main()
