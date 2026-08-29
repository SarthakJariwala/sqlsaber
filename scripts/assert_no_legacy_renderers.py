#!/usr/bin/env python3
"""Fail if sqlsaber still imports Rich, Questionary, or Rich Console helpers.

Run from the repo root. Reviewers rerun this to check the migration.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = (
    ROOT / "src",
    ROOT / "plugins",
    ROOT / "tests",
)
SKIP_PARTS = {".venv", "node_modules", "__pycache__", ".audit"}
FORBIDDEN = re.compile(
    r"^\s*(?:from\s+rich(?:\.\S+)?\s+import|import\s+rich(?:\.\S+)?"
    r"|from\s+questionary(?:\.\S+)?\s+import|import\s+questionary(?:\.\S+)?)\b"
)
CREATE_CONSOLE = re.compile(r"\bcreate_console\b")

# cyclopts still imports Rich for --help. That is allowed only inside cyclopts.
# This script only scans our trees.


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files.append(path)
    return sorted(files)


def main() -> int:
    hits: list[str] = []
    for path in iter_python_files():
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN.search(line):
                hits.append(f"{rel}:{lineno}:{line.rstrip()}")
            elif CREATE_CONSOLE.search(line) and "assert_no_legacy_renderers" not in str(
                rel
            ):
                hits.append(f"{rel}:{lineno}:{line.rstrip()}")
    if hits:
        print(f"legacy renderer imports: {len(hits)}")
        for hit in hits:
            print(hit)
        return 1
    print("legacy renderer imports: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
