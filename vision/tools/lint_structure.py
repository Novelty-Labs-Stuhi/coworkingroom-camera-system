"""Structural guard: catch the rules a normal linter cannot see.

Two checks, run over the Python files passed on the command line:

1. No god-files       -- a single module may not exceed MAX_LINES.
2. No embedded DSLs    -- no HTML/SQL/CSS glued into a host-language string literal.

Exit code is non-zero if any file violates a rule, so it can gate a pre-commit hook.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_LINES = 300

# Substrings that betray a foreign language hiding inside a Python string literal.
FOREIGN_MARKERS = (
    "<html",
    "<!doctype",
    "<div",
    "<span",
    "<table",
    "select ",
    "insert into",
    "update ",
    "delete from",
    "create table",
)


def _too_long(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_LINES:
        return f"{path}: {len(lines)} lines (limit {MAX_LINES}) -- split by responsibility"
    return None


def _embedded_language(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            hit = next((m for m in FOREIGN_MARKERS if m in lowered), None)
            if hit is not None:
                problems.append(
                    f"{path}:{node.lineno}: string literal contains '{hit.strip()}' "
                    "-- move foreign languages into their own files"
                )
    return problems


def check(path: Path) -> list[str]:
    problems: list[str] = []
    if (long := _too_long(path)) is not None:
        problems.append(long)
    # This module necessarily contains the marker substrings in FOREIGN_MARKERS;
    # scanning it for them would be a false positive, so skip the DSL check for it.
    if path.resolve() != Path(__file__).resolve():
        problems.extend(_embedded_language(path))
    return problems


def main(argv: list[str]) -> int:
    problems = [p for arg in argv for p in check(Path(arg))]
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
