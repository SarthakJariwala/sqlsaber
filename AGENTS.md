# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/sqlsaber/`
  - `cli/`: CLI entry (`saber`, `sqlsaber`), REPL, prompts.
  - `agents/`: agent implementations (pydantic‑ai).
  - `tools/`: SQL, introspection, registry.
  - `database/`: connection, resolver, schema utilities.
  - `memory/`, `conversation/`: state and persistence.
  - `config/`: settings, API keys, DB configs.
- Tests: `tests/` mirror modules (`test_cli/`, `test_tools/`, …).
- Docs & assets: `docs/`, `sqlsaber.gif`, `sqlsaber.svg`.

## Build, Test, and Development Commands
- Install (editable): `uv sync`
- Lint: `uv run ruff check .`
- Type check (all): `uvx ty check src/`
- Type check (targeted): `uvx ty check <file>`
- Format: `uv run ruff format .`
- Tests (all): `uv run pytest -q`
- Tests (targeted): `uv run pytest tests/test_tools -q`
- Run CLI (dev): `uv run saber` or `uv run python -m sqlsaber`


Note: Prefer `uv run ruff ...` over `uvx ruff ...` to avoid hitting user-level uv caches that may be restricted in sandboxed environments.

## Performance: CLI Startup Time
- CLI startup must stay fast (<0.5s for `--help`). Avoid eager imports of heavy modules (`pydantic_ai`, `google.genai`, `openai`, etc.) at module load time.
- Use lazy `__getattr__` imports in `__init__.py` files for heavy exports (see `sqlsaber/__init__.py` and `sqlsaber/config/__init__.py`).
- CLI command modules should defer heavy imports to inside command functions, not at module top-level.
- Test with `time uv run saber --help` before merging changes that touch imports.

## Coding Style & Naming Conventions
- Python 3.12+, 4‑space indent, strictly use modern (3.12+) type hints approach.
  - Type check must pass without errors always.
- Ruff is the linter/formatter; code must be clean and formatted.
- Naming: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- Keep public CLI surfaces in `cli/`; factor reusable logic into modules under `sqlsaber/`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio`.
- Place tests under `tests/`, name files `test_*.py` and mirror package layout.
- Include tests for new behavior and bug fixes; prefer async tests for async code.
- Use fixtures from `tests/conftest.py` where possible.
- Tests must pass without errors always.
