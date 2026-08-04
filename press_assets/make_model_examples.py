"""Pick and render 3 good validation examples per model.

Every candidate is scored against its own hand-labelled ground truth, so the
"working" claim in the deck is checkable rather than cherry-picked by eye.
One image at a time -- batching mixed-orientation frames rescales the batch.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path("/Users/jasonli/Documents/GitHub/ZenblenImageTesting")
DS = ROOT / "training/labeling/datasets"
OUT = ROOT / "press_assets"
OUT.mkdir(exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = (250, 250, 250)
GT_COL = (120, 230, 130)          # ground truth -- green, thin
MIN_ID_GAP = 25                   # frame ids are timestamps; avoid near-duplicates

SEG = [
    # key, label, weights, dataset, tint colour (BGR), min GT px
    ("roi",     "standard-seg",  ROOT / "active_pipeline/checkpoints/yolo_standard_seg.pt",
     "smoothie_dataset_std", (255, 170, 40),  20000),
    ("chunk",   "chunk-seg",     ROOT / "active_pipeline/checkpoints/yolo_chunk_seg.pt",
     "chunk_dataset",        (60, 60, 255),   1200),
    ("spill",   "spill-seg",     ROOT / "active_pipeline/checkpoints/yolo_spill_seg.pt",
     "spill_dataset",        (60, 200, 255),  1200),
    ("logo",    "logo-seg",      ROOT / "training/checkpoints/yolo_logo_seg.pt",
     "logo_dataset",         (235, 120, 220), 1500),
    ("blended", "blended-seg",
     ROOT / "training/runs/blended-seg/blended-nano-v1/weights/best.pt",
     "blended_dataset",      (120, 220, 120), 20000),
]


def label(img, text, y=26, scale=0.5, color=INK):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, 2)
    cv2.rectangle(img, (8, y - h - 8), (8 + w + 14, y + 9), (18, 18, 18), -1)
    cv2.putText(img, text, (15, y), FONT, scale, color, 2, cv2.LINE_AA)
    return img


def tint(img, mask, color, alpha=0.42):
    over = img.copy()
    over[mask > 0] = color
    return cv2.addWeighted(over, alpha, img, 1 - alpha, 0)


def outline(img, mask, color, thick=2):
    cs, _ = cv2.findContours((mask > 0).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, thick)
    return img


def gt_mask(txt: Path, shape) -> np.ndarray:
    """Union of every polygon in a YOLO-seg label file."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    if not txt.exists():
        return m
    for line in txt.read_text().split("\n"):
        v = line.split()
        if len(v) < 7:
            continue
        pts = np.array(v[1:], float).reshape(-1, 2) * (w, h)
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m


def pred_mask(res, shape) -> np.ndarray:
    """Union of predicted instances, resized to the frame."""
    h, w = shape
    if res.masks is None or len(res.masks) == 0:
        return np.zeros((h, w), np.uint8)
    acc = np.zeros((h, w), np.uint8)
    for i in range(len(res.masks)):
        m = (res.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
        acc |= cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return acc


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def fid(p: Path) -> int:
    return int("".join(c for c in p.stem if c.isdigit()) or 0)


def spread(cands, n=3):
    """Highest-IoU picks, forced apart in time so tiles aren't near-duplicates."""
    out = []
    for c in sorted(cands, key=lambda c: -c["iou"]):
        if all(abs(c["id"] - o["id"]) >= MIN_ID_GAP for o in out):
            out.append(c)
        if len(out) == n:
            break
    return out


def strip(tiles, title, sub):
    grid = np.hstack([cv2.resize(t, (330, 440)) for t in tiles])
    banner = np.full((58, grid.shape[1], 3), 18, np.uint8)
    cv2.putText(banner, title, (16, 25), FONT, 0.62, INK, 2, cv2.LINE_AA)
    cv2.putText(banner, sub, (16, 46), FONT, 0.44, (150, 155, 145), 1, cv2.LINE_AA)
    return np.vstack([banner, grid])


def do_seg(key, name, weights, dsname, col, min_gt):
    model = YOLO(str(weights))
    imgs = sorted((DS / dsname / "images/val").glob("*.jpg"))
    cands = []
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None or img.shape[:2] != (640, 480):
            continue                                  # portrait only
        g = gt_mask(DS / dsname / "labels/val" / f"{ip.stem}.txt", img.shape[:2])
        if int((g > 0).sum()) < min_gt:
            continue
        r = model(str(ip), verbose=False)[0]           # ONE image at a time
        p = pred_mask(r, img.shape[:2])
        if not p.any():
            continue
        conf = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        cands.append({"path": ip, "id": fid(ip), "iou": iou(p, g),
                      "px": int((p > 0).sum()), "conf": conf})

    picks = spread(cands)
    tiles = []
    for c in picks:
        img = cv2.imread(str(c["path"]))
        r = model(str(c["path"]), verbose=False)[0]
        p = pred_mask(r, img.shape[:2])
        g = gt_mask(DS / dsname / "labels/val" / f"{c['path'].stem}.txt", img.shape[:2])
        v = tint(img, p, col)
        v = outline(v, p, (255, 255, 255), 2)
        v = outline(v, g, GT_COL, 1)
        v = label(v, f"IoU {c['iou']:.2f}   conf {c['conf']:.2f}", y=26, scale=0.5)
        v = label(v, f"{c['px']:,} px", y=54, scale=0.46)
        tiles.append(v)

    ious = [c["iou"] for c in picks]
    cv2.imwrite(str(OUT / f"ex_{key}.png"), strip(
        tiles, f"{name} - three held-out validation frames",
        "white = prediction   green = hand-labelled ground truth"))
    return {"model": name, "n_scored": len(cands),
            "picks": [{"id": c["path"].stem, "iou": round(c["iou"], 3),
                       "conf": round(c["conf"], 3), "px": c["px"]} for c in picks],
            "mean_iou_picked": round(float(np.mean(ious)), 3) if ious else 0.0,
            "median_iou_all": round(float(np.median([c["iou"] for c in cands])), 3)}


def do_cls():
    model = YOLO(str(ROOT / "active_pipeline/checkpoints/best_cleaning.pt"))
    names = model.names
    want = {"clean": 2, "dirty": 1}                     # show both classes
    picked, scored = [], 0
    for cls, k in want.items():
        cands = []
        for ip in sorted((DS / "cleandone_cls_dataset/val" / cls).glob("*.jpg")):
            img = cv2.imread(str(ip))
            if img is None:
                continue
            r = model(str(ip), verbose=False)[0]
            top = int(r.probs.top1)
            scored += 1
            if names[top] != cls:                       # only correct calls
                continue
            cands.append({"path": ip, "id": fid(ip), "cls": cls,
                          "conf": float(r.probs.top1conf)})
        cands.sort(key=lambda c: -c["conf"])
        chosen = []
        for c in cands:
            if all(abs(c["id"] - o["id"]) >= MIN_ID_GAP for o in chosen):
                chosen.append(c)
            if len(chosen) == k:
                break
        picked += chosen

    tiles = []
    for c in picked:
        img = cv2.imread(str(c["path"]))
        good = c["cls"] == "clean"
        col = (90, 220, 90) if good else (60, 60, 255)
        cv2.rectangle(img, (2, 2), (img.shape[1] - 3, img.shape[0] - 3), col, 5)
        img = label(img, f"{c['cls'].upper()}  {c['conf'] * 100:.1f}%", y=30, scale=0.6, color=col)
        img = label(img, f"label {c['cls']}  (correct)", y=58, scale=0.44)
        tiles.append(img)
    cv2.imwrite(str(OUT / "ex_cleandone.png"), strip(
        tiles, "cleandone-cls - three held-out validation frames",
        "predicted class and confidence; every call matches its hand label"))
    return {"model": "cleandone-cls", "n_scored": scored,
            "picks": [{"id": c["path"].stem, "cls": c["cls"],
                       "conf": round(c["conf"], 4)} for c in picked]}


if __name__ == "__main__":
    report = [do_seg(*s) for s in SEG] + [do_cls()]
    json.dump(report, open(sys.argv[1], "w"), indent=1)
    for r in report:
        print(json.dumps(r))
