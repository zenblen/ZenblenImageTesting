"""
Shared infrastructure for the blendedness-method comparison.

Every scoring method lives in methods/<name>.py and exposes:

    NAME: str
    def score(image_bgr, roi_mask, logo_mask) -> (score_0_100, flag01)

where:
    image_bgr : HxWx3 uint8 BGR
    roi_mask  : HxW uint8, 255 = liquid (from YOLO standard-seg, hole-filled)
    logo_mask : HxW uint8, 255 = printed logo (from YOLO logo-seg) — EXCLUDE these
    returns   : (score in [0,100]  where 100 = perfectly blended,
                 flag01 HxW float32 in [0,1] = per-pixel "unblendedness",
                        0 outside the scored region — used only for the overlay)

Methods must NOT run YOLO themselves — masks are passed in. Use core_roi() if you
want to drop the meniscus/gasket border. Keep methods self-contained (numpy/cv2).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "training" / "labeling" / "data" / "images"
STD_WEIGHTS = REPO / "training" / "checkpoints" / "yolo_standard_seg.pt"
LOGO_WEIGHTS = REPO / "training" / "checkpoints" / "yolo_logo_seg.pt"
CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "mask_cache"

_STD = None
_LOGO = None


def _models():
    global _STD, _LOGO
    if _STD is None:
        from ultralytics import YOLO
        _STD = YOLO(str(STD_WEIGHTS))
        _LOGO = YOLO(str(LOGO_WEIGHTS))
    return _STD, _LOGO


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def _roi_from_model(image: np.ndarray) -> np.ndarray:
    std, _ = _models()
    h, w = image.shape[:2]
    r = std(image, verbose=False, device="cpu")[0]
    if r.masks is None or len(r.masks) == 0:
        return np.zeros((h, w), np.uint8)
    raw = r.masks.data[int(np.argmax(r.boxes.conf.cpu().numpy()))].cpu().numpy()
    m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return _fill_holes(((m > 0.5) * 255).astype(np.uint8))


def _logo_from_model(image: np.ndarray) -> np.ndarray:
    _, logo = _models()
    h, w = image.shape[:2]
    r = logo(image, verbose=False, device="cpu", conf=0.25)[0]
    mask = np.zeros((h, w), np.uint8)
    if r.masks is None:
        return mask
    for raw in r.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask


def get_masks(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_bgr, roi_mask, logo_mask), caching masks to disk so methods
    and the comparison harness never re-run YOLO for the same image."""
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{image_path.stem}.npz"
    if cache.exists():
        d = np.load(cache)
        return image, d["roi"], d["logo"]
    roi = _roi_from_model(image)
    logo = _logo_from_model(image)
    np.savez_compressed(cache, roi=roi, logo=logo)
    return image, roi, logo


# --- glare / specular-highlight suppression -------------------------------
# Glare off the plastic cup comes in TWO forms here:
#   (1) hard specular — bright AND desaturated (near white/gray), often clipped.
#   (2) colored bloom — a broad bright reflection that KEEPS the smoothie's hue,
#       so it is NOT desaturated. This is the common case on these cups.
# Both are LUMINANCE events: L* jumps up while a*/b* (hue) stay ~constant. A real
# unblended patch instead SHIFTS colour (large a*/b* change). So we flag pixels
# that are much brighter than a large-scale local baseline while their hue barely
# moves, plus the classic bright+desaturated specular core. Excluding these from
# the scored region stops glare masquerading as a colour "deviation".
# NOTE: dilution (water/ice) is also brighter-same-hue and can be caught here —
# an inherent optics ambiguity; the real fix is a polarizer (see memory).
GLARE_SUPPRESS = True
GLARE_L_MIN = 232        # hard specular: LAB L* (0..255) this bright or more
GLARE_CHROMA_MAX = 14    # hard specular: this desaturated or less
GLARE_BASE_KERNEL = 201  # large blur = broad local baseline for bloom detection
GLARE_DL_MIN = 11.0      # bloom: L* this much brighter than the local baseline
GLARE_DHUE_MAX = 8.0     # bloom: a*/b* shift stays under this (hue preserved)
GLARE_DILATE_PX = 4      # also drop the soft halo around each hot spot
# Bloom removal is only SAFE on coloured cups: there glare bloom is distinct from
# the saturated liquid. On PALE cups a bright unmixed patch looks exactly like a
# bright bloom (both bright + near-neutral hue), so bloom removal would delete
# real defect signal (see memory: bottom-cream-not-separable-on-device). So we
# gate it on the liquid's median chroma; below the gate we keep specular only.
GLARE_BLOOM_MIN_CHROMA = 18.0


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def glare_mask(image_bgr: np.ndarray,
               roi_mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of specular highlights + colored bloom to EXCLUDE. Bloom is
    only added when the liquid (roi_mask) is coloured enough to distinguish glare
    from a pale unmixed patch; otherwise specular-only is used."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)

    # (1) hard specular core: bright + desaturated — always safe
    specular = (L >= GLARE_L_MIN) & (chroma <= GLARE_CHROMA_MAX)

    # (2) colored bloom — only on coloured cups (saturation gate)
    colored = True
    if roi_mask is not None:
        m = roi_mask > 0
        colored = bool(m.any()) and float(np.median(chroma[m])) >= GLARE_BLOOM_MIN_CHROMA
    if colored:
        k = _odd(GLARE_BASE_KERNEL)
        L_base = cv2.GaussianBlur(L, (k, k), 0)
        a_base = cv2.GaussianBlur(a, (k, k), 0)
        b_base = cv2.GaussianBlur(b, (k, k), 0)
        d_hue = np.sqrt((a - a_base) ** 2 + (b - b_base) ** 2)
        bloom = ((L - L_base) >= GLARE_DL_MIN) & (d_hue <= GLARE_DHUE_MAX)
    else:
        bloom = np.zeros_like(specular)

    g = specular | bloom
    if GLARE_DILATE_PX > 0:
        kd = 2 * GLARE_DILATE_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd))
        g = cv2.dilate(g.astype(np.uint8), kernel).astype(bool)
    return g


def scored_region(roi_mask: np.ndarray, logo_mask: np.ndarray | None,
                  border_erode_px: int = 0,
                  image_bgr: np.ndarray | None = None) -> np.ndarray:
    """Boolean region a method should score: ROI minus logo, optionally with the
    border eroded to drop the meniscus band / gasket rim. If image_bgr is given
    and GLARE_SUPPRESS is on, specular highlights are excluded too."""
    region = roi_mask > 0
    if logo_mask is not None:
        region &= ~(logo_mask > 0)
    if image_bgr is not None and GLARE_SUPPRESS:
        region &= ~glare_mask(image_bgr, roi_mask)
    if border_erode_px and border_erode_px > 0:
        k = 2 * border_erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        region = cv2.erode(region.astype(np.uint8), kernel).astype(bool)
    return region


def sample_images(n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images under {IMAGES_DIR}")
    if len(files) <= n:
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def render_panel(image: np.ndarray, flag01: np.ndarray, roi_mask: np.ndarray,
                 score: float, label: str) -> np.ndarray:
    """One comparison panel: image with flagged regions OUTLINED (see-through) +
    ROI outline + score + method label. Consistent visual across all methods."""
    vis = image.copy()
    flagged = (flag01 > 0.33).astype(np.uint8)
    severe = (flag01 > 0.66).astype(np.uint8)
    for m, col in ((flagged, (0, 255, 255)), (severe, (0, 0, 255))):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, col, 2)
    roi_cnts, _ = cv2.findContours((roi_mask > 0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, roi_cnts, -1, (0, 255, 0), 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(vis, f"{label}: {score:.1f}", (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis
