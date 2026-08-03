#!/usr/bin/env python3
"""
Smoothie blendedness pipeline runner.

Usage (single image):
    python run.py --pipeline blend --image data/images/test.jpg

Usage (batch — directory of images):
    python run.py --pipeline blend --image data/images/

Outputs per run:
    outputs/<stem>_<pipeline>_mask.png      - unblended region overlay
    outputs/<stem>_<pipeline>_roi.png       - detected container boundary
    outputs/<stem>_<pipeline>_result.json   - score, passed, metadata
    outputs/comparison.csv                  - per-image scores (batch)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent))

from smoothie_cv.config import Config
from smoothie_cv.detection import (
    SmoothieType,
    _classify_smoothie,
    detect_container,
    draw_container_overlay,
)
from smoothie_cv.scoring.metrics import overlay_mask

PIPELINE_NAMES = ["blend", "spill"]


def load_pipeline(name: str, config: Config):
    if name == "blend":
        from smoothie_cv.pipelines.blend import BlendPipeline
        return BlendPipeline(config)
    if name == "spill":
        from smoothie_cv.pipelines.spill import SpillPipeline
        return SpillPipeline(config)
    raise ValueError(f"Unknown pipeline: {name!r}. Choose from {PIPELINE_NAMES}")


def _shade_subfolder(smoothie_type: SmoothieType) -> str:
    """Map smoothie type to output subfolder name."""
    if smoothie_type == SmoothieType.RED_PINK:
        return "red_pink"
    return "yellow"


def run_spill_single(
    image_path: Path,
    config: Config,
) -> dict:
    """Run the spill pipeline (no container ROI — spill is defined on the whole
    frame). Writes a spill overlay + JSON and returns the record."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pipeline = load_pipeline("spill", config)
    t0 = time.perf_counter()
    result = pipeline.analyze(image)
    runtime_ms = (time.perf_counter() - t0) * 1000

    smoothie_dir = config.output_dir / image_path.stem
    smoothie_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_spill"

    mask_path = smoothie_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), overlay_mask(image, result.mask))

    record = {
        "image": str(image_path),
        "pipeline": "spill",
        "spill_detected": result.spill_detected,
        "spill_area_px": result.spill_area_px,
        "confidence": round(result.confidence, 4),
        "passed": not result.spill_detected,  # "pass" = clean (no spill)
        "runtime_ms": round(runtime_ms, 1),
        "mask_path": str(mask_path),
        "metadata": result.metadata,
    }
    (smoothie_dir / f"{stem}_result.json").write_text(json.dumps(record, indent=2))

    verdict = "SPILL" if result.spill_detected else "clean"
    print(
        f"[spill] {image_path.name}  {verdict}  "
        f"area={result.spill_area_px}px conf={result.confidence:.2f}  "
        f"({runtime_ms:.0f} ms)"
    )
    return record


def run_single(
    image_path: Path,
    pipeline_name: str,
    config: Config,
    detector: str | None = None,
) -> dict:
    if pipeline_name == "spill":
        return run_spill_single(image_path, config)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    smoothie_type = _classify_smoothie(image)

    # ROI detection — YOLO-only active path.
    roi_mask, _bbox, det = detect_container(
        image, config, prefer=detector, return_meta=True
    )

    pipeline = load_pipeline(pipeline_name, config)

    t0 = time.perf_counter()
    result = pipeline.analyze(image, roi_mask)
    runtime_ms = (time.perf_counter() - t0) * 1000

    # One subfolder per smoothie, grouped by shade: <run>/<shade>/<stem>/
    shade_dir = config.output_dir / _shade_subfolder(smoothie_type)
    smoothie_dir = shade_dir / image_path.stem
    smoothie_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{pipeline_name}"

    # ROI overlay from the chosen detector
    roi_path = smoothie_dir / f"{stem}_roi.png"
    cv2.imwrite(str(roi_path), draw_container_overlay(image, roi_mask))

    # mask overlay (unblended region highlighted), computed on the ROI
    mask_vis = overlay_mask(image, result.mask)
    mask_path = smoothie_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_vis)

    # write JSON result
    record = {
        "image": str(image_path),
        "pipeline": pipeline_name,
        "smoothie_type": smoothie_type.value,
        "blend_score": round(result.blend_score, 4),
        "passed": result.passed,
        "threshold": config.threshold,
        "runtime_ms": round(runtime_ms, 1),
        "detector": det["detector"],
        "detector_fallback": det["fallback"],
        "top_roughness": det["roughness"],
        "roi_path": str(roi_path),
        "mask_path": str(mask_path),
        "metadata": result.metadata,
    }
    json_path = smoothie_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(record, indent=2))

    status = "PASS" if result.passed else "FAIL"
    det_tag = f"  det={det['detector']}" + ("(fallback)" if det["fallback"] else "")
    print(
        f"[{pipeline_name}] {image_path.name}  "
        f"score={result.blend_score:.3f}  {status}  "
        f"({runtime_ms:.0f} ms){det_tag}"
    )
    print(f"  dir   → {smoothie_dir}")
    return record


def run_batch(
    image_dir: Path,
    pipeline_names: list[str],
    config: Config,
    detector: str | None = None,
) -> list[dict]:
    images = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.png"))
    if not images:
        print(f"No .jpg/.png images found in {image_dir}")
        return []

    rows: list[dict] = []
    for img_path in images:
        for pname in pipeline_names:
            try:
                row = run_single(img_path, pname, config, detector=detector)
                rows.append(row)
            except NotImplementedError as e:
                print(f"[{pname}] SKIP — {e}")
            except Exception as e:
                print(f"[{pname}] ERROR on {img_path.name}: {e}")
    return rows


def make_run_dir(output_root: Path, pipeline_names: list[str], started: datetime) -> Path:
    """Create a unique timestamped subfolder for this run and return it."""
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    label = "all" if len(pipeline_names) > 1 else pipeline_names[0]
    run_dir = output_root / f"{stamp}__{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_run_readme(info: dict) -> str:
    """Build a short human-readable summary; full data lives in comparison.csv."""
    run_id = info["run_id"]
    pipelines = ", ".join(info["pipelines"])
    multi_pipeline = len(info["pipelines"]) > 1
    summary = info["summary"]
    started = datetime.fromisoformat(info["started"]).strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {run_id}",
        "",
        f"{pipelines} · `{info['image_source']}` · threshold **{info['threshold']:.2f}** · "
        f"{info['duration_s']}s · {started}",
        "",
        f"**{summary['pass']} pass / {summary['fail']} fail** "
        f"({info['num_images']} images) · "
        f"scores {summary['min_score']:.3f}–{summary['max_score']:.3f}, "
        f"mean {summary['mean_score']:.3f}",
    ]

    def _metric(r: dict) -> str:
        # chunk pipeline reports a blend score; spill reports spilled area.
        if "blend_score" in r:
            return f"{r['blend_score']:.3f}"
        return f"{r.get('spill_area_px', 0)}px"

    failures = [r for r in info["results"] if not r.get("passed")]
    if failures:
        lines.extend(["", "## Failures", ""])
        if multi_pipeline:
            lines.extend(["| Image | Pipeline | Score |", "|---|---|---|"])
            for r in failures:
                lines.append(
                    f"| `{Path(r['image']).name}` | {r['pipeline']} | {_metric(r)} |"
                )
        else:
            lines.extend(["| Image | Score |", "|---|---|"])
            for r in failures:
                lines.append(f"| `{Path(r['image']).name}` | {_metric(r)} |")

    lines.extend(["", "→ [comparison.csv](comparison.csv) · [run_info.json](run_info.json)", ""])
    return "\n".join(lines)


def write_run_manifest(
    run_dir: Path,
    records: list[dict],
    pipeline_names: list[str],
    image_source: Path,
    config: Config,
    started: datetime,
) -> None:
    """Write run_info.json (machine) + README.md (human) describing this run."""
    finished = datetime.now()
    passed = sum(1 for r in records if r.get("passed"))
    failed = len(records) - passed
    # blend_score is chunk-pipeline-only; spill records omit it.
    scores = [r["blend_score"] for r in records if "blend_score" in r] or [0.0]

    info = {
        "run_id": run_dir.name,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 2),
        "pipelines": pipeline_names,
        "threshold": config.threshold,
        "image_source": str(image_source),
        "num_images": len({r["image"] for r in records}),
        "num_results": len(records),
        "summary": {
            "pass": passed,
            "fail": failed,
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "mean_score": round(sum(scores) / len(scores), 4),
        },
        "results": records,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    (run_dir / "README.md").write_text(format_run_readme(info))

    if records:
        csv_path = run_dir / "comparison.csv"
        fieldnames = ["image", "pipeline", "blend_score", "passed", "threshold", "runtime_ms"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothie blendedness pipeline runner")
    parser.add_argument(
        "--pipeline", default="blend",
        choices=PIPELINE_NAMES,
        help="Analysis pipeline (default: blend).",
    )
    parser.add_argument("--image", required=True,
                        help="Path to an image file or directory of images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Pass/fail threshold (0–1). Overrides config default (0.90).")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (optional)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write outputs (default: outputs/)")
    parser.add_argument("--detector", choices=["auto", "yolo"],
                        default="auto",
                        help="ROI detector (YOLO-only). 'auto' and 'yolo' are equivalent.")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.threshold is not None:
        config.threshold = args.threshold
    detector = None if args.detector == "auto" else args.detector
    output_root = Path(args.output_dir) if args.output_dir is not None else config.output_dir

    image_path = Path(args.image)
    pipeline_name = args.pipeline
    pipeline_names = [pipeline_name]

    if not (image_path.is_dir() or image_path.is_file()):
        print(f"Error: {image_path} is not a file or directory.")
        sys.exit(1)

    # every invocation gets its own timestamped subfolder under outputs/
    started = datetime.now()
    run_dir = make_run_dir(output_root, pipeline_names, started)
    config.output_dir = run_dir
    print(f"Run directory → {run_dir}\n")

    records: list[dict] = []
    if image_path.is_dir():
        records = run_batch(
            image_path, pipeline_names, config, detector=detector,
        )
    else:
        for pname in pipeline_names:
            try:
                records.append(run_single(
                    image_path, pname, config, detector=detector,
                ))
            except NotImplementedError as e:
                print(f"[{pname}] SKIP — {e}")

    write_run_manifest(run_dir, records, pipeline_names, image_path, config, started)
    print(f"\nManifest → {run_dir / 'run_info.json'}")
    print(f"Summary  → {run_dir / 'README.md'}")


if __name__ == "__main__":
    main()
