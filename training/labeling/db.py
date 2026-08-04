"""Shared SQLite helpers for the labeling tool.

One database (``labeling/labels.db``) holds image registry + multi-mode
annotations, keyed by ``file_id`` (the Files-API file id, an integer PK).
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# All labeling artefacts live under this package directory, next to the scripts.
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "labels.db"
IMAGES_DIR = ROOT / "data" / "images"
CHUNK_SEED_DIR = ROOT / "data" / "polygons_chunk_seed"

# --- multi-mode labeler (app_multi.py) --------------------------------------
# A SECOND, self-contained pipeline that shares this DB's image registry but
# writes ONLY to the additive `annotations` / `mode_status` tables below — the
# `labels` table above (the container/standard container-detection dataset) is
# never touched. Each mode is an INDEPENDENT segmentation pass over the same
# image pool and exports to its OWN single-class dataset (see export_multi.py):
#   standard -> smoothie inside the cup   -> class 0: smoothie
#   spill    -> smoothie outside the cup  -> class 0: spill
#   logo     -> the zenblen logo/wordmark -> class 0: logo
#   chunk    -> an unblended lump/chunk   -> class 0: chunk
#   unmixed  -> non-chunk unblending      -> class 0: unmixed
#               (streaks, unmixed color patches, gradients/layering, watery/
#               diluted zones). Small isolated speckle — seeds/chia/blueberry
#               skin — is normal recipe texture and is NOT marked. This is the
#               failure mode the chunk model structurally misses (no discrete
#               blob to find); it is a SEPARATE additive class, chunk untouched.
# One image labeled in all five modes yields five SEPARATE image+label pairs,
# one per dataset — never a single file with mixed-class labels.
MODES = ("standard", "spill", "logo", "chunk", "unmixed")
MODE_CLASS_NAMES = {"standard": "smoothie", "spill": "spill", "logo": "logo",
                    "chunk": "chunk", "unmixed": "unmixed"}
# Persisted statuses. "skip" is deliberately NOT stored (Skip = advance without
# writing state, so the image stays undecided and reappears later).
MODE_STATUSES = ("labeled", "clean")

# --- model-assisted review pipeline (predict_batch.py + app_review.py) -------
# A THIRD, separate pipeline: run each mode's trained YOLO-seg model over the raw
# (undecided-for-that-mode) images, store predictions, and let a human
# Approve / Reject / Edit them. Predictions live in their OWN tables below and
# do NOT enter training until APPROVED — approval writes into the shared
# `annotations` (tagged source='model') + `mode_status='labeled'` so the existing
# export_multi.py picks them up unchanged. Reject leaves the image undecided so
# the hand pipeline (app_multi.py) re-serves it.
REVIEW_STATUSES = ("pending", "approved", "rejected")

# Per-image content flags (image_flags table). 'no_smoothie' = a machinery /
# empty-rig shot with no smoothie in frame; excluded from every pipeline.
NO_SMOOTHIE = "no_smoothie"

# Provenance values for annotations.source (see the ADD COLUMN migration below).
SOURCE_HAND = "hand"    # drawn in app_multi.py (default) / migrated legacy labels
SOURCE_MODEL = "model"  # a model prediction approved (possibly edited) in review

# Deployed per-mode weights (under training/checkpoints/). After train_multi,
# copy standard/spill/chunk best.pt into ../active_pipeline/checkpoints/ as well.
# predict_batch.py falls back to a run's best.pt via --weights when missing.
CHECKPOINTS_DIR = ROOT.parent / "checkpoints"
MODE_WEIGHTS = {
    "standard": CHECKPOINTS_DIR / "yolo_standard_seg.pt",
    "spill": CHECKPOINTS_DIR / "yolo_spill_seg.pt",
    "logo": CHECKPOINTS_DIR / "yolo_logo_seg.pt",
    "chunk": CHECKPOINTS_DIR / "yolo_chunk_seg.pt",
    "unmixed": CHECKPOINTS_DIR / "yolo_unmixed_seg.pt",
}

# --- classification labeler (app_classify.py) --------------------------------
# A FOURTH, self-contained pipeline: whole-image classification (no polygons),
# for tasks where the label is a single verdict about the whole frame rather
# than a region within it. Shares the `files` image registry above but writes
# ONLY to its own `classifications` table (see schema below) — the seg tables
# (`labels`, `annotations`, `mode_status`, `predictions`, `review_status`) are
# never touched, and `db.MODES` is not extended with classification tasks.
#   cleandone -> is this CleanDone station photo dirty or clean?
CLS_TASKS = ("cleandone",)
CLS_LABELS = ("dirty", "clean")
CLS_WEIGHTS = {
    "cleandone": CHECKPOINTS_DIR / "best_cleaning.pt",
}
# Files-API category each task's images are drawn from (files.category_name).
TASK_CATEGORY = {
    "cleandone": "CleanDone",
}


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the labeling DB, creating the schema on first use.

    ``db_path`` resolves to the module-level ``DB_PATH`` at call time (not bound
    as a default), so tests can point it elsewhere by reassigning ``db.DB_PATH``.
    """
    conn = sqlite3.connect(str(db_path if db_path is not None else DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            file_id       INTEGER PRIMARY KEY,
            order_id      INTEGER,
            file_name     TEXT,
            file_url      TEXT,
            category_name TEXT,
            file_type     TEXT,
            created_at    TEXT,
            downloaded    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS labels (
            file_id    INTEGER PRIMARY KEY REFERENCES files(file_id),
            verdict    TEXT NOT NULL,
            polygon    TEXT,               -- JSON list of [x, y] pixel points
            labeled_at TEXT NOT NULL
        );

        -- Multi-mode labeler (app_multi.py). Additive; the tables above are
        -- untouched. `annotations` is multi-instance: one row per polygon, so
        -- an image can hold several same-class shapes for a given mode.
        CREATE TABLE IF NOT EXISTS annotations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            mode       TEXT NOT NULL,      -- one of db.MODES
            polygon    TEXT NOT NULL,      -- JSON list of [x, y] pixel points
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_annotations_file_mode
            ON annotations(file_id, mode);

        -- One decision per (image, mode). status is one of db.MODE_STATUSES.
        -- No row for a (file, mode) means "undecided" -> served again as next.
        CREATE TABLE IF NOT EXISTS mode_status (
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            mode       TEXT NOT NULL,
            status     TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (file_id, mode)
        );

        -- Model-assisted review pipeline (predict_batch.py + app_review.py).
        -- Model predictions awaiting human review. Multi-instance like
        -- `annotations`; polygon is pixel-space so it round-trips identically
        -- when approved. An image with a prediction row but no accepted
        -- polygons (zero detections) still gets a review_status row so the
        -- reviewer can catch false-negatives.
        CREATE TABLE IF NOT EXISTS predictions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            mode       TEXT NOT NULL,      -- one of db.MODES
            polygon    TEXT NOT NULL,      -- JSON list of [x, y] pixel points
            confidence REAL,               -- model confidence for this instance
            model_run  TEXT,               -- weights file the prediction came from
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_predictions_file_mode
            ON predictions(file_id, mode);

        -- One review decision per (image, mode). status is one of
        -- db.REVIEW_STATUSES. Independent of mode_status: a 'pending' review
        -- has no mode_status row (so it stays out of training); 'approved'
        -- writes mode_status; 'rejected' leaves mode_status empty so the hand
        -- pipeline re-serves the image.
        CREATE TABLE IF NOT EXISTS review_status (
            file_id     INTEGER NOT NULL REFERENCES files(file_id),
            mode        TEXT NOT NULL,
            status      TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            PRIMARY KEY (file_id, mode)
        );

        -- Per-image content flags, independent of mode. flag='no_smoothie' marks
        -- machinery / empty-rig shots (the container model finds no smoothie) so
        -- they are excluded EVERYWHERE: review queues, hand labeler, and export.
        -- Set by flag_smoothie_presence.py.
        CREATE TABLE IF NOT EXISTS image_flags (
            file_id    INTEGER PRIMARY KEY REFERENCES files(file_id),
            flag       TEXT NOT NULL,      -- one of db.IMAGE_FLAGS
            confidence REAL,               -- top smoothie confidence seen (0 if none)
            created_at TEXT NOT NULL
        );

        -- Whole-image classification labeler (app_classify.py). One label per
        -- (image, task) — the classification analogue of `mode_status`, but a
        -- single verdict instead of a polygon set. No `annotations` row is ever
        -- written for a classification task.
        CREATE TABLE IF NOT EXISTS classifications (
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            task       TEXT NOT NULL,      -- one of db.CLS_TASKS
            label      TEXT NOT NULL,      -- one of db.CLS_LABELS
            labeled_at TEXT NOT NULL,
            PRIMARY KEY (file_id, task)
        );
        """
    )
    # Provenance column on annotations (additive; older DBs predate it). Guarded
    # so re-running is a no-op. Existing rows default to 'hand'.
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(annotations)")}
    if "source" not in cols:
        conn.execute(
            f"ALTER TABLE annotations ADD COLUMN source TEXT DEFAULT '{SOURCE_HAND}'"
        )
    conn.commit()


def load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Dependency-free (no python-dotenv). Looks for ``labeling/.env`` first, then
    the repo-root ``.env``. Existing environment variables win, so an explicit
    ``export`` or ``--api-key`` still overrides the file. Missing file is fine.
    """
    for env_path in (ROOT / ".env", ROOT.parent / ".env", ROOT.parent.parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def ensure_dirs() -> None:
    """Create the on-disk data directories used by the pipeline stages."""
    for d in (IMAGES_DIR, CHUNK_SEED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# Every table that carries a `file_id` FK back to `files` — kept in one place
# so a hard delete never leaves an orphaned row behind in a table added later.
_FILE_ID_TABLES = ("labels", "annotations", "mode_status", "predictions",
                    "review_status", "image_flags", "classifications")


def delete_file(conn: sqlite3.Connection, file_id: int) -> None:
    """Permanently remove ``file_id`` everywhere: every DB row that
    references it (across ALL pipelines — seg + classification, not just the
    one that called this) and its jpg on disk. IRREVERSIBLE — callers are
    responsible for confirming with the user first.
    """
    for table in _FILE_ID_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE file_id = ?", (file_id,))
    conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    conn.commit()
    path = IMAGES_DIR / f"{file_id}.jpg"
    if path.exists():
        path.unlink()
