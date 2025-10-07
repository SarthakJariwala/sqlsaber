## Commands

- **Run Python**: `uv run python`
- **Run tests**: `uv run python -m pytest`
- **Run single test**: `uv run python -m pytest tests/test_path/test_file.py::test_function`
- **Lint**: `uv run ruff check --fix`
- **Format**: `uv run ruff format`

## Architecture

- **CLI App**: Agentic SQL assistant for natural language to SQL
- **Core modules**: `agents/` (AI logic), `cli/` (commands), `database/` (connections)
- **Database support**: PostgreSQL, SQLite, MySQL via asyncpg/aiosqlite/aiomysql

## Code Style

- **Imports**: stdlib → 3rd party → local, use relative imports within modules
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants, `_private` methods
- **Types**: Always use modern type hints (3.12+), async functions for I/O
- **Errors**: Use try/finally for cleanup
- **Docstrings**: Triple-quoted with Args/Returns sections
