#!/usr/bin/env python3
"""Guard against AGENTS.md / CLAUDE.md drift.

Every agent adapter in this repo is intentionally a thin pointer to docs/. The
failure mode we care about is one vendor's file being updated and the other's
going stale, so the invariant is simple: at each directory, AGENTS.md and
CLAUDE.md must be byte-identical, and every doc they link to must exist.

Run:  python scripts/check_agent_docs.py
Exit: 0 = ok, 1 = drift found. Stdlib only, no dependencies.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANONICAL_DOCS = [
    "docs/agent-guidelines.md",
    "docs/architecture.md",
    "docs/repository-map.md",
    "docs/testing.md",
]
MAX_ADAPTER_LINES = 60  # adapters are pointers; rules belong in docs/

# Skip generated/vendored trees so this never walks the 5000-file dataset dirs.
SKIP_DIRS = {".git", "__pycache__", "datasets", "data", "runs", "outputs", "node_modules"}


def find_adapters() -> list[Path]:
    """Locate every AGENTS.md, skipping generated trees."""
    found = []
    for path in ROOT.rglob("AGENTS.md"):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        found.append(path)
    return sorted(found)


def main() -> int:
    errors: list[str] = []

    # 1. The canonical docs must all exist.
    for doc in CANONICAL_DOCS:
        if not (ROOT / doc).is_file():
            errors.append(f"missing canonical doc: {doc}")

    adapters = find_adapters()
    if not adapters:
        errors.append("no AGENTS.md found anywhere")

    for agents in adapters:
        rel = agents.relative_to(ROOT)
        claude = agents.with_name("CLAUDE.md")

        # 2. Paired and byte-identical -- this is the anti-drift invariant.
        if not claude.is_file():
            errors.append(f"{rel}: no CLAUDE.md sibling")
            continue
        a_bytes, c_bytes = agents.read_bytes(), claude.read_bytes()
        if a_bytes != c_bytes:
            errors.append(
                f"{rel}: AGENTS.md and CLAUDE.md differ "
                f"({len(a_bytes)} vs {len(c_bytes)} bytes) -- they must be identical"
            )

        text = a_bytes.decode("utf-8", "replace")

        # 3. Stay a pointer, not a second source of truth.
        n_lines = len(text.splitlines())
        if n_lines > MAX_ADAPTER_LINES:
            errors.append(
                f"{rel}: {n_lines} lines exceeds {MAX_ADAPTER_LINES} -- "
                f"move rules into docs/ and link instead"
            )

        # 4. Must point at the canonical guidelines.
        if "agent-guidelines.md" not in text:
            errors.append(f"{rel}: does not link docs/agent-guidelines.md")

        # 5. Every relative markdown link must resolve.
        for target in re.findall(r"\]\(([^)#:]+\.md)[^)]*\)", text):
            if (agents.parent / target).resolve().is_file():
                continue
            if (ROOT / target).is_file():
                continue
            errors.append(f"{rel}: broken link -> {target}")

    if errors:
        print("agent-doc check FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"agent-doc check passed: {len(adapters)} adapter pair(s), "
          f"{len(CANONICAL_DOCS)} canonical docs, all links resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
