# Linting & quality tooling

Quality tooling is wired in before feature code, and runs automatically on every
commit.

## What runs

| Tool | Purpose | Config |
|------|---------|--------|
| **Ruff** (lint) | pyflakes, imports, bugbear, complexity (C901), size limits (PLR09xx) | `pyproject.toml` `[tool.ruff]` |
| **Ruff** (format) | consistent formatting | `pyproject.toml` |
| **structural guard** | rules Ruff can't see: no god-files (>300 lines), no foreign languages in string literals | `tools/lint_structure.py` |
| **pytest** | unit tests for the pure logic | `pyproject.toml` `[tool.pytest.ini_options]` |

## One-time setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows;  source .venv/bin/activate on Unix
pip install -e ".[dev]"
pre-commit install
```

## Running by hand

```bash
ruff check .              # lint
ruff format .             # format
python tools/lint_structure.py src/stuhi_vision/*.py   # structural guard
pytest                    # tests (pure logic; no models needed)
pre-commit run --all-files
```

The pure-logic tests (`test_embeddings`, `test_doorway`, `test_ledger`) need only
numpy, so they run fast in CI without downloading any vision model.
