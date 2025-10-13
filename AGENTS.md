# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/sqlsaber/`
  - `cli/`: CLI entry (`saber`, `sqlsaber`), REPL, prompts.
  - `agents/`: agent implementations (pydantic‑ai).
  - `tools/`: SQL, introspection, registry.
  - `database/`: connection, resolver, schema utilities.
  - `memory/`, `conversation/`: state and persistence.
  - `config/`: settings, API keys, OAuth, DB configs.
- Tests: `tests/` mirror modules (`test_cli/`, `test_tools/`, …).
- Docs & assets: `docs/`, `sqlsaber.gif`, `sqlsaber.svg`.

## Build, Test, and Development Commands
- Install (editable): `uv sync`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Tests (all): `uv run pytest -q`
- Tests (targeted): `uv run pytest tests/test_tools -q`
- Run CLI (dev): `uv run saber` or `uv run python -m sqlsaber`


Note: Prefer `uv run ruff ...` over `uvx ruff ...` to avoid hitting user-level uv caches that may be restricted in sandboxed environments.

## Coding Style & Naming Conventions
- Python 3.12+, 4‑space indent, use modern type hints.
- Ruff is the linter/formatter; code must be clean and formatted.
- Naming: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`; constants `UPPER_SNAKE`.
- Keep public CLI surfaces in `cli/`; factor reusable logic into modules under `sqlsaber/`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio`.
- Place tests under `tests/`, name files `test_*.py` and mirror package layout.
- Include tests for new behavior and bug fixes; prefer async tests for async code.
- Use fixtures from `tests/conftest.py` where possible.

## Commit & Pull Request Guidelines
- Commits: short, imperative; prefer conventional prefixes (e.g., `feat:`, `fix:`, `docs:`). Reference issues/PRs when relevant.
- PRs must: describe the change and rationale, include tests, pass `ruff` and `pytest`, and update docs/README for user‑visible changes.
- For CLI UX changes, include before/after samples (command + output snippet).

## Security & Configuration Tips
- Never commit secrets. Configure via CLI: `saber db add` and `saber models set` (credentials stored via system keyring).
- Queries run read‑only by default; avoid introducing mutating behavior in tools without explicit safeguards.
