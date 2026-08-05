"""Whole-image classification labeler — one verdict per frame, no polygons.

A FOURTH, self-contained pipeline alongside app_multi.py (polygons) and
predict_batch.py/app_review.py (model-assisted review): no drawing, just one
verdict per image. Shares the `files` image registry with the rest of the
labeling tool but writes ONLY to the additive `classifications` table (see
labeling/db.py) — nothing here touches `annotations` / `mode_status`.

Tasks (each scoped to its own ``files.category_name``, its own label set, its
own dataset — see db.CLS_TASK_LABELS / db.TASK_CATEGORY):
    cleandone -> dirty | clean          on 'CleanDone' photos
    xytable   -> powder | no_powder     on 'XYTable' photos

Deliberately does NOT join ``image_flags``/``no_smoothie`` — that gate keys on
the smoothie/container detector and would wrongly exclude the empty-station and
bare-table shots these tasks are about.

Run (from ``training/``):
    python labeling/app_classify.py                # http://127.0.0.1:5003
    #    ?task=xytable  switches task (default: the first in db.CLS_TASKS)
    #    per-label hotkeys (see TASK_UI) · S = skip (no save) · ←/→ prev/next
    #    Optional: labeling/priority/<task>.txt (one file_id per line) is
    #    served FIRST by /api/next — used to bump specific images to the
    #    front of the queue (e.g. re-review candidates).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

app = Flask(__name__, template_folder=str(db.ROOT / "templates"),
            static_folder=str(db.ROOT / "static"))

# Presentation only — hotkey + colour per label, plus banner text. Kept here
# rather than in db.py because none of it affects stored data or the dataset.
# Label names MUST match db.CLS_TASK_LABELS (asserted by _task_ui below), and
# hotkeys must avoid the fixed globals: S skip, Z undo, X delete, ←/→ navigate.
TASK_UI: dict[str, dict] = {
    "cleandone": {
        "title": "CleanDone classify",
        "labels": [{"name": "dirty", "key": "d", "color": "#e0524a"},
                   {"name": "clean", "key": "c", "color": "#39c07a"}],
    },
    "xytable": {
        "title": "XYTable powder classify",
        "labels": [{"name": "powder", "key": "p", "color": "#e0a33f"},
                   {"name": "no_powder", "key": "n", "color": "#39c07a"}],
    },
}
_RESERVED_KEYS = {"s", "z", "x"}


def _check_task(task: str | None) -> str:
    if task not in db.CLS_TASKS:
        abort(400, f"task must be one of {db.CLS_TASKS}")
    return task


def _task_ui(task: str) -> dict:
    """UI config for a task, validated against the DB's label set so a rename
    in db.CLS_TASK_LABELS can't silently leave an unreachable button here."""
    ui = TASK_UI[task]
    names = tuple(lab["name"] for lab in ui["labels"])
    if names != db.cls_labels(task):
        raise RuntimeError(
            f"TASK_UI[{task!r}] labels {names} != db.cls_labels({task!r}) "
            f"{db.cls_labels(task)} — keep them in sync"
        )
    keys = [lab["key"] for lab in ui["labels"]]
    clash = _RESERVED_KEYS.intersection(keys)
    if clash or len(set(keys)) != len(keys):
        raise RuntimeError(f"TASK_UI[{task!r}] hotkeys {keys} clash ({clash or 'duplicate'})")
    return ui


@app.route("/")
def index() -> str:
    """Render the labeler for ``?task=`` (default: first task in db.CLS_TASKS).

    One template serves every task; the label buttons, hotkeys and progress
    line are generated from ``TASK_UI`` + ``db.cls_labels()``.
    """
    task = _check_task(request.args.get("task", db.CLS_TASKS[0]))
    ui = _task_ui(task)
    return render_template("label_classify.html", task=task, title=ui["title"],
                           dataset=f"{task}_cls_dataset", labels=ui["labels"],
                           tasks=list(db.CLS_TASKS))


@app.route("/image/<int:file_id>")
def image(file_id: int):
    path = db.IMAGES_DIR / f"{file_id}.jpg"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")


def _item_payload(conn, file_id: int, task: str) -> dict:
    row = conn.execute(
        "SELECT label FROM classifications WHERE file_id = ? AND task = ?",
        (file_id, task),
    ).fetchone()
    return {"file_id": file_id, "task": task, "label": row["label"] if row else None,
            "priority": file_id in set(_priority_ids(task))}


def _priority_ids(task: str) -> list[int]:
    """Optional front-of-queue file_ids from ``labeling/priority/<task>.txt``
    (one integer id per line, # comments / blanks ignored) — mirrors
    app_multi.py's ``_priority_ids``, scoped to classification tasks instead of
    seg modes (no name collision: tasks and modes are separate namespaces)."""
    path = db.ROOT / "priority" / f"{task}.txt"
    if not path.exists():
        return []
    out: list[int] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            continue
    return out


@app.route("/api/next")
def api_next():
    """First image in this task's category with NO classification yet.

    ``?after=<id>`` steps forward past a given id (mirrors app_multi's
    /api/next). If ``labeling/priority/<task>.txt`` exists, undecided ids
    listed there are served first (in file order), then the normal ascending
    file_id walk. Scoped to the task's category_name; no image_flags gate —
    see module docstring.
    """
    task = _check_task(request.args.get("task"))
    category = db.TASK_CATEGORY[task]
    after = request.args.get("after", type=int)
    conn = db.connect()

    # Priority queue: serve undecided ids from priority/<task>.txt first. When
    # ?after= points at a priority id, continue AFTER that entry so Skip still
    # advances; once the remaining priority list is exhausted, fall through.
    priority = _priority_ids(task)
    if priority:
        undecided = {
            r["file_id"]
            for r in conn.execute(
                "SELECT f.file_id FROM files f "
                "LEFT JOIN classifications c ON c.file_id = f.file_id AND c.task = ? "
                "WHERE f.downloaded = 1 AND f.category_name = ? AND c.file_id IS NULL",
                (task, category),
            )
        }
        start = 0
        if after is not None:
            try:
                start = priority.index(after) + 1
            except ValueError:
                start = 0
        for fid in priority[start:]:
            if fid in undecided and (db.IMAGES_DIR / f"{fid}.jpg").exists():
                return jsonify(_item_payload(conn, fid, task))

    base = (
        "SELECT f.file_id FROM files f "
        "LEFT JOIN classifications c ON c.file_id = f.file_id AND c.task = ? "
        "WHERE f.downloaded = 1 AND f.category_name = ? AND c.file_id IS NULL "
    )
    params: tuple = (task, category)
    if after is not None:
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1",
            params + (after,),
        ).fetchone()
    else:
        row = conn.execute(base + "ORDER BY f.file_id ASC LIMIT 1", params).fetchone()

    # Only serve images whose jpg is actually on disk.
    while row is not None:
        if (db.IMAGES_DIR / f"{row['file_id']}.jpg").exists():
            return jsonify(_item_payload(conn, row["file_id"], task))
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1",
            params + (row["file_id"],),
        ).fetchone()
    return jsonify({"done": True})


@app.route("/api/item/<int:file_id>")
def api_item(file_id: int):
    task = _check_task(request.args.get("task"))
    conn = db.connect()
    if conn.execute("SELECT 1 FROM files WHERE file_id = ?", (file_id,)).fetchone() is None:
        abort(404)
    return jsonify(_item_payload(conn, file_id, task))


@app.route("/api/seek")
def api_seek():
    """Adjacent ALREADY-CLASSIFIED image in this task's category, by file_id —
    lets you always walk back through every past decision, persisting across
    page reloads (the client's local session history only covers images
    viewed this browser session; this covers every classified image ever).

    Restricted to classified images on purpose: the priority queue serves
    undecided images in P(dirty)-ranked order, not file_id order, so a raw
    file_id walk over ALL images (the original design, matching app_multi.py)
    would land on an arbitrary, usually-undecided neighbor with no relation
    to what you were just looking at. Undecided images are still reachable —
    that's what /api/next (priority-aware) is for.

    Returns the same payload as /api/item, or {"edge": True} at the pool edge.
    """
    task = _check_task(request.args.get("task"))
    category = db.TASK_CATEGORY[task]
    file_id = request.args.get("file_id", type=int)
    direction = request.args.get("dir", "prev")
    if file_id is None or direction not in ("prev", "next"):
        abort(400, "file_id required and dir must be 'prev' or 'next'")
    op, order = ("<", "DESC") if direction == "prev" else (">", "ASC")
    sql = (
        f"SELECT f.file_id FROM files f "
        f"JOIN classifications c ON c.file_id = f.file_id AND c.task = ? "
        f"WHERE f.downloaded = 1 AND f.category_name = ? AND f.file_id {op} ? "
        f"ORDER BY f.file_id {order} LIMIT 1"
    )
    conn = db.connect()
    cur = file_id
    while True:
        row = conn.execute(sql, (task, category, cur)).fetchone()
        if row is None:
            return jsonify({"edge": True})
        if (db.IMAGES_DIR / f"{row['file_id']}.jpg").exists():
            return jsonify(_item_payload(conn, row["file_id"], task))
        cur = row["file_id"]


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Persist a decision for (file_id, task).

    Body: {file_id, task, label in db.cls_labels(task)}. Skip is client-only
    (advance without writing), so it never reaches here.
    """
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    task = _check_task(data.get("task"))
    label = data.get("label")
    labels = db.cls_labels(task)
    if file_id is None or label not in labels:
        abort(400, f"file_id required and label must be one of {labels}")

    conn = db.connect()
    conn.execute(
        """
        INSERT INTO classifications (file_id, task, label, labeled_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(file_id, task) DO UPDATE SET
            label=excluded.label, labeled_at=excluded.labeled_at
        """,
        (file_id, task, label),
    )
    conn.commit()
    return jsonify({"ok": True})


@app.route("/api/unclassify", methods=["POST"])
def api_unclassify():
    """Clear any decision for (file_id, task) — back to unlabeled.

    Used by the client for Undo (revert the last classify() that had no prior
    label). Skip never calls this — Skip is purely client-side, advance
    without writing.
    """
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    task = _check_task(data.get("task"))
    if file_id is None:
        abort(400, "file_id required")

    conn = db.connect()
    conn.execute(
        "DELETE FROM classifications WHERE file_id = ? AND task = ?",
        (file_id, task),
    )
    conn.commit()
    return jsonify({"ok": True})


@app.route("/api/delete", methods=["POST"])
def api_delete():
    """Permanently delete an image everywhere — every DB row referencing
    file_id (across every pipeline, not just this task) plus the jpg on disk.
    IRREVERSIBLE. For corrupt/wrong/unusable images you never want to see
    again; for "I don't want this in the dataset but the file is fine" use
    Skip or Undo instead, since those are reversible.
    """
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    if file_id is None:
        abort(400, "file_id required")

    conn = db.connect()
    if conn.execute("SELECT 1 FROM files WHERE file_id = ?", (file_id,)).fetchone() is None:
        abort(404)
    db.delete_file(conn, file_id)
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    """Per-task counts: how many of this task's category carry each label, and
    how many are still undecided. Keys are the task's OWN label names (so
    cleandone reports dirty/clean and xytable reports powder/no_powder) plus
    ``labels`` — the ordered name list, so the client can render without
    knowing the vocabulary.

    Also reports this task's OWN priority-queue counts (from
    ``labeling/priority/<task>.txt``), mirroring app_multi's ``/api/progress``:
      ``priority_total``     — ids listed in that task's file
      ``priority_remaining`` — of those, how many are still undecided.
    """
    conn = db.connect()
    tasks = [request.args["task"]] if request.args.get("task") else list(db.CLS_TASKS)
    out: dict = {"tasks": {}}
    for t in tasks:
        _check_task(t)
        category = db.TASK_CATEGORY[t]
        total = conn.execute(
            "SELECT COUNT(*) c FROM files WHERE downloaded = 1 AND category_name = ?",
            (category,),
        ).fetchone()["c"]
        labels = db.cls_labels(t)
        counts = {label: 0 for label in labels}
        for r in conn.execute(
            "SELECT c.label, COUNT(*) n FROM classifications c "
            "JOIN files f ON f.file_id = c.file_id "
            "WHERE c.task = ? AND f.category_name = ? GROUP BY c.label",
            (t, category),
        ):
            if r["label"] in counts:  # defensive: ignore stale/renamed labels
                counts[r["label"]] = r["n"]
        decided = sum(counts.values())
        priority = _priority_ids(t)
        priority_remaining = 0
        if priority:
            undecided = {
                r["file_id"]
                for r in conn.execute(
                    "SELECT f.file_id FROM files f "
                    "LEFT JOIN classifications c ON c.file_id = f.file_id AND c.task = ? "
                    "WHERE f.downloaded = 1 AND f.category_name = ? AND c.file_id IS NULL",
                    (t, category),
                )
            }
            priority_remaining = sum(1 for fid in priority if fid in undecided)
        out["tasks"][t] = {**counts, "labels": list(labels),
                            "total": total, "decided": decided,
                            "remaining": total - decided,
                            "priority_total": len(priority),
                            "priority_remaining": priority_remaining}
    return jsonify(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classification labeling UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    db.ensure_dirs()
    db.connect().close()  # create schema (incl. classifications table) up front
    # Validate every task's UI config now, so a label/hotkey mismatch fails at
    # launch instead of on the first request to that task.
    for task in db.CLS_TASKS:
        _task_ui(task)
        print(f"  {task:<10} ({'/'.join(db.cls_labels(task))})"
              f"  ->  http://{args.host}:{args.port}/?task={task}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
