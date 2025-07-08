# SQLSaber Agent Guide

## Commands

- **Run Python**: `uv run python`
- **Run tests**: `uv run python -m pytest`
- **Run single test**: `uv run python -m pytest tests/test_path/test_file.py::test_function`
- **Lint**: `uv run ruff check --fix`
- **Format**: `uv run ruff format`

## Architecture

- **CLI App**: Agentic SQL assistant for natural language to SQL
- **Core modules**: `agents/` (AI logic), `cli/` (commands), `clients/` (LLM clients), `database/` (connections), `mcp/` (Model Context Protocol server)
- **Database support**: PostgreSQL, SQLite, MySQL via asyncpg/aiosqlite/aiomysql
- **MCP integration**: Exposes tools via `sqlsaber-mcp` for Claude Code and other MCP clients

## Code Style

- **Imports**: stdlib → 3rd party → local, use relative imports within modules
- **Naming**: snake_case functions/vars, PascalCase classes, UPPER_SNAKE constants, `_private` methods
- **Types**: Always use modern type hints (3.12+), async functions for I/O
- **Errors**: Custom exception hierarchy from `LLMClientError`, use try/finally for cleanup
- **Docstrings**: Triple-quoted with Args/Returns sections
