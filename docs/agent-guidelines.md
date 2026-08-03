# Agent guidelines

Canonical rules for any coding agent in this repo (Claude Code, Codex, Cursor, or
other). Root and nested `AGENTS.md` / `CLAUDE.md` files are thin adapters that
point here — this file is the single source of truth.

Companion docs: [architecture](architecture.md) ·
[repository map](repository-map.md) · [testing](testing.md)

---

## 0. Session convention

Begin every response with the marker 🟢 followed by a space.

## 1. Exploration

- **Start with the smallest relevant module**, not the repo root.
- **Read the nearest applicable `AGENTS.md`** before editing (nested ones exist in
  `active_pipeline/` and `training/`).
- **Search for symbols before opening large files.** `grep -rn "<symbol>" --include='*.py'`
  beats reading a 371-line Flask app.
- **Avoid repo-wide scans.** See §2 — an unscoped glob here returns thousands of
  dataset files.
- Prefer [repository-map.md](repository-map.md) over re-deriving structure.

## 2. Never scan these paths

**98% of tracked files (5207 of ~5300) are training data, not code.** There are
only 51 Python files and ~6,600 lines of source. An unscoped `glob '**/*'`,
`ls-files`, or `grep -r` wastes thousands of tokens on JPEGs and label text.

Excluded from *every* search unless the task is explicitly about dataset contents:

| Path | What it is |
|---|---|
| `training/labeling/datasets/**` | Exported YOLO datasets — 3197 `.jpg`, 2004 `.txt`, 10 `.cache` |
| `training/labeling/data/**` | Raw downloaded image pool (gitignored) |
| `training/runs/**` | Training-run artifacts and weights |
| `**/outputs/**` | Pipeline run outputs (gitignored) |
| `**/__pycache__/**`, `*.pyc` | Bytecode |
| `*.pt` | Model weights — binary, never readable |
| `training/labeling/labels.db*` | SQLite — query it, never read it as text |

Scope searches to a directory (`grep -rn X active_pipeline/`) or a glob
(`--include='*.py'`). To list source files:

```bash
git ls-files '*.py'
```

These datasets **are deliberately committed** — `REMOTE_TRAIN.md` depends on them
for GPU training. Do not "clean them up".

## 3. Editing

- **Reuse existing types, utilities, and patterns.** Detector functions share one
  contract: `(image, config) -> (mask, bbox)`. Pipelines return `BlendResult` /
  `SpillResult` from `smoothie_cv/pipelines/base.py`. Config flows through
  `smoothie_cv/config.py:Config` — do not add ad-hoc parameters.
- **Keep edits limited to the requested scope.** Do not rewrite unrelated working
  code or reformat files you are not changing.
- New tunables go on `Config` with a default that preserves current behavior.
- **Update docs when architecture changes** — this file, `architecture.md`, and
  `repository-map.md`.

## 4. Validation

Run in this order; stop at the first failure. Details in [testing.md](testing.md).

1. **Targeted test** — the class or test touching your change
2. **Package check** — full test file, plus `ruff check <paths>` if installed
3. **Full validation** — `pytest` + `ruff check .` from the repo root

**Never claim a check passed unless it actually passed.** Paste or summarize real
output. If a check was skipped or a tool is missing, say so explicitly.

**Keep command output concise** — pipe verbose runs through `tail`, `-q`, or a
count. Never dump a full training log into context.

## 5. Model-behavior changes need evidence

Detector thresholds and gates in this repo were tuned against measured
false-positive rates on real frames. Changing one is an empirical claim.

- Measure before and after on a real image set; report counts, not impressions.
- Match the runtime inference path exactly when calibrating: **one image at a
  time**. Batching mixed-orientation frames through ultralytics rescales the
  batch and silently corrupts mask geometry (see [architecture.md](architecture.md#inference-invariants)).
- Prefer additive, default-off flags over changing a tuned default.

## 6. Parallelism and subagents

- **Do not use subagents for simple work** — a single file read or a scoped grep
  is faster inline.
- **Give subagents narrow, non-overlapping scopes.** One directory or one
  question each; never two agents on the same files.
- Batch independent tool calls into one message. Do not parallelize calls where
  one's input depends on another's output.

## 7. Completion

Before reporting done:

- [ ] Targeted tests pass (real output seen)
- [ ] Change is within requested scope
- [ ] Docs updated if structure or behavior changed
- [ ] `python scripts/check_agent_docs.py` passes if you touched `AGENTS.md`,
      `CLAUDE.md`, or `docs/`

State plainly what you ran, what passed, and what you did not do. Do not commit,
push, or deploy unless asked.
