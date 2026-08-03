"""Validate the texture-blendedness methods against EXISTING chunk labels.

No new labeling. Uses labels.db chunk mode_status as a free (partial) ground
truth: status='labeled' => a real defect is present (expect LOW blend score),
status='clean' => no chunk (expect HIGH score). For each method we measure how
well the score separates the two groups (AUC = P(clean scores > labeled score);
1.0 = perfect, 0.5 = no separation).

The payoff cases are printed too: chunk-CLEAN images that still score LOW are
candidates for the non-chunk unblending (streaks/patches) the chunk model
structurally misses — the whole reason this metric exists.

Run (needs YOLO -> conda python):
  /opt/miniconda3/bin/python validate_vs_chunklabels.py            # all chunk-decided
  /opt/miniconda3/bin/python validate_vs_chunklabels.py --limit 200
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
import pkgutil

import common  # noqa: E402
import methods as methods_pkg  # noqa: E402

DB = common.REPO / "training" / "labeling" / "labels.db"
OUT = Path(__file__).resolve().parent / "outputs" / "validation"


def load_methods():
    mods = []
    for m in pkgutil.iter_modules(methods_pkg.__path__):
        mod = importlib.import_module(f"methods.{m.name}")
        if hasattr(mod, "score") and hasattr(mod, "NAME"):
            mods.append(mod)
    mods.sort(key=lambda x: x.NAME)
    return mods


def chunk_decided(limit=None):
    """(file_id, status) for chunk-decided images that exist on disk."""
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_id, status FROM mode_status "
        "WHERE mode='chunk' AND status IN ('labeled','clean') ORDER BY file_id"
    ).fetchall()
    out = []
    for r in rows:
        p = common.IMAGES_DIR / f"{r['file_id']}.jpg"
        if p.exists():
            out.append((r["file_id"], r["status"], p))
    return out[:limit] if limit else out


def auc(pos_scores, neg_scores):
    """P(neg > pos) via Mann-Whitney; here pos=labeled(defect), neg=clean.
    A blend score should be HIGHER for clean, so a good metric gives AUC>0.5."""
    pos = np.asarray(pos_scores, float)
    neg = np.asarray(neg_scores, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1  # 1-based ranks (ties ~ok)
    r_neg = ranks[len(pos):].sum()
    u = r_neg - len(neg) * (len(neg) + 1) / 2
    return u / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    mods = load_methods()
    names = [m.NAME for m in mods]
    items = chunk_decided(args.limit)
    print(f"scoring {len(items)} chunk-decided images x {len(names)} methods: {names}",
          flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (fid, status, path) in enumerate(items):
        try:
            image, roi, logo = common.get_masks(path)
        except Exception as e:
            print(f"  [{fid}] mask fail: {e}", flush=True)
            continue
        if (roi > 0).sum() < 500:  # no usable ROI
            continue
        rec = {"file_id": fid, "status": status}
        for mod in mods:
            try:
                s, _ = mod.score(image, roi, logo)
            except Exception as e:
                s = float("nan")
                print(f"  [{fid}] {mod.NAME} fail: {e}", flush=True)
            rec[mod.NAME] = round(float(s), 2)
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(items)}", flush=True)

    # write raw scores
    csv_path = OUT / "scores_vs_chunk.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file_id", "status"] + names)
        w.writeheader()
        w.writerows(rows)

    labeled = [r for r in rows if r["status"] == "labeled"]
    clean = [r for r in rows if r["status"] == "clean"]
    print(f"\nscored {len(rows)} usable  (labeled={len(labeled)} clean={len(clean)})\n")
    print(f"{'method':16} {'AUC':>6}  {'med_labeled':>11} {'med_clean':>9}  separation")
    print("-" * 62)
    ranking = []
    for nm in names:
        lab = [r[nm] for r in labeled if not np.isnan(r[nm])]
        cln = [r[nm] for r in clean if not np.isnan(r[nm])]
        a = auc(lab, cln)
        ml, mc = (np.median(lab) if lab else float("nan"),
                  np.median(cln) if cln else float("nan"))
        ranking.append((a, nm, ml, mc))
    for a, nm, ml, mc in sorted(ranking, reverse=True):
        print(f"{nm:16} {a:6.3f}  {ml:11.1f} {mc:9.1f}  {mc - ml:+.1f}")

    best = max(ranking)[1]
    print(f"\nbest separator: {best}")
    # payoff: chunk-CLEAN images scoring LOW on the best method = missed streaks
    cln_sorted = sorted((r for r in clean if not np.isnan(r[best])),
                        key=lambda r: r[best])[:15]
    print(f"\nchunk-CLEAN but LOW {best} (candidate non-chunk unblending the "
          f"chunk model missed):")
    for r in cln_sorted:
        print(f"  {r['file_id']}  {best}={r[best]:.1f}")
    print(f"\nraw scores -> {csv_path}")


if __name__ == "__main__":
    main()
