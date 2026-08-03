# Agent instructions — ZenblenImageTesting

Adapter file. Canonical rules live in **[docs/agent-guidelines.md](docs/agent-guidelines.md)** —
read it before non-trivial work. `AGENTS.md` and `CLAUDE.md` are byte-identical by
design; edit `docs/` instead, and verify with `python scripts/check_agent_docs.py`.

| Doc | Read it when |
|---|---|
| [docs/agent-guidelines.md](docs/agent-guidelines.md) | Always — exploration, editing, validation, completion rules |
| [docs/repository-map.md](docs/repository-map.md) | "Where does this change go?" |
| [docs/architecture.md](docs/architecture.md) | Components, data flow, entry points, boundaries |
| [docs/testing.md](docs/testing.md) | What to run, and what is actually covered |

## Non-negotiables

**Begin every response with the marker 🟢 followed by a space.**

**Never scan these paths** — 98% of tracked files (5207 of ~5300) are training
data, not code. There are only 51 Python files:

```
training/labeling/datasets/**   training/labeling/data/**   training/runs/**
**/outputs/**   **/__pycache__/**   *.pt   *.db   *.cache
```

Scope every search: `grep -rn X active_pipeline/ --include='*.py'`, or list source
with `git ls-files '*.py'`. These datasets are committed on purpose
(`REMOTE_TRAIN.md` needs them) — do not "clean them up".

**`active_pipeline/` is a leaf.** It ships to the Jetson alone and must never
import from `training/`, `experimentation/`, or root scripts.

**One image at a time.** Batching mixed-orientation frames through ultralytics
rescales the batch and silently corrupts mask geometry.

**Never claim a check passed unless it actually passed.**

## Commands

```bash
pytest -k Blend                      # targeted
pytest                               # full suite (9 tests)
ruff check .                         # lint (optional install)
python scripts/check_agent_docs.py   # agent-doc drift
```

Use the conda base interpreter — it is the only env with torch + ultralytics.

Nested adapters with directory-specific rules: `active_pipeline/`, `training/`.
