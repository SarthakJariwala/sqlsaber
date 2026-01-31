---
title: Changelog
description: Release history and notable changes to SQLsaber
---

All notable changes to SQLsaber will be documented here.

### Unreleased

---

### v0.52.0

#### Added

- **Plugin system for tools** via `sqlsaber.tools` entry points
  - External packages can register custom tools by adding entry points in `pyproject.toml`
  - Supports direct Tool class registration or factory functions
  - Automatic plugin discovery on startup
- `Tool.requires_ctx` class attribute for tools needing `RunContext` access
- `execute_sql` results now include `file` key with `tool_call_id` for result traceability
- Display manager shows JSON output for non-SQL tool inputs

---

### v0.51.0

#### Added

- `/handoff` command for starting fresh threads with context
  - Generates a context-aware prompt summarizing the current conversation
  - Extracts database schema, SQL queries, and key findings
  - Presents an editable draft before starting the new thread
  - Usage: `/handoff <goal>` (e.g., `/handoff optimize this query for performance`)

---

### v0.50.0

#### Changed

- `execute_sql` no longer accepts a `limit` argument; queries without LIMIT now receive a server-side cap of 1,000 rows
- JSON results now return full database rows (no preview truncation)
- Updated system prompt to reflect the limit changes

---

### v0.49.1

#### Fixed

- Fixed "File name too long" error when passing long memory text strings (>255 chars) to the Python API

---

### v0.49.0

#### Added

- Python API now accepts `thinking_level` parameter for extended thinking models
  - Accepts string values: `"minimal"`, `"low"`, `"medium"`, `"high"`, `"maximum"`
  - Automatically enables thinking when `thinking_level` is provided
  - Example: `SQLSaber(database="my.db", thinking_level="high")`

### Changed

- Default recommended model for new configurations is now Claude Opus 4.5 (changed from Sonnet 4.5)

---

### v0.48.0

#### Added

- Configurable thinking levels across all providers
  - New `ThinkingLevel` enum with 5 levels: `minimal`, `low`, `medium`, `high`, `maximum`
  - Each level maps to provider-specific configurations:
    - **Anthropic**: `budget_tokens` (1,024 → 100,000)
    - **OpenAI**: `reasoning_effort` (minimal → xhigh)
    - **Google**: `thinking_level` (MINIMAL → HIGH)
    - **Groq**: Binary (any level enables reasoning when thinking is on)
  - `saber models set` now prompts for thinking level after model selection
  - `saber models current` displays both model and thinking configuration
  - `/thinking` slash command enhanced:
    - `/thinking` - Show current status and level
    - `/thinking on` - Enable with current level
    - `/thinking off` - Disable thinking
    - `/thinking <level>` - Set level (minimal, low, medium, high, maximum)
  - Config schema upgraded to v2 with automatic migration from v1
  - `--thinking` / `--no-thinking` now override saved defaults when running `saber`

---

### v0.47.2

#### Fixed

- Fixed JSON serialization of `Decimal`, `datetime`, `date`, `time`, `UUID`, and `bytes` types returned by database drivers

---

### v0.47.1

#### Fixed

- Fixed rendering of update text

---

### v0.47.0 - 2026-01-12

#### Added

- CLI: allow multiple CSVs via repeated `-d` (each CSV becomes its own DuckDB view)
- CLI: `saber threads resume` now accepts multiple `-d` CSVs
- Python API: `SQLSaber(database=[...])` now accepts multiple CSVs
- CSV view names are derived from file stems; collisions are suffixed (`_2`, `_3`, ...)

---

### v0.46.1

#### Fixed

- Fixed rendering of update text

---

### v0.46.0 - 2026-01-11

#### Added

- Python API: add `memory` parameter to inject session context (accepts text or a file path)
- Agent: prefer passed `memory` over saved database memories

---

### v0.45.0 - 2026-01-11

#### Added

- New version check on startup

---

### v0.44.0 - 2026-01-10

#### Removed

- Removed Anthropic OAuth (Claude Pro/Max subscription) authentication support; Anthropic now requires an API key.

#### Changed

- `saber auth setup` / onboarding now configure API keys only
- `saber auth reset` now removes stored API keys only

---

### v0.43.0 - 2025-01-08

#### Added

- `saber threads export` command to export thread transcripts as shareable HTML files
  - Dark and light theme support with toggle button
  - Syntax highlighting for SQL queries
  - Renders markdown content with code blocks, tables, and formatting
  - Tool results displayed as collapsible sections with formatted tables
  - Schema introspection results show columns, types, keys, and comments
  - Includes thread metadata: database, model, timestamps
  - Usage: `saber threads export <thread-id> [-o output.html]`

---

### v0.42.0 - 2025-01-05

#### Added

- Session usage summary displayed on exit
  - Shows input context size (current tokens), total output tokens generated
  - Displays request count and tool call count
  - Summary shown when exiting via `/exit`, `/quit`, or Ctrl+D
  - Also displayed after single-query (non-interactive) mode

---

### v0.41.0 - 2025-12-25

#### Changed

- Simplified event stream handling in agent and API modules

### v0.40.0 - 2025-12-24

#### Added

- `--allow-dangerous` flag to enable INSERT/UPDATE/DELETE and DDL operations
  - Allows a restricted set of DML operations: INSERT, UPDATE, DELETE, MERGE, REPLACE
  - Allows non-destructive DDL operations: CREATE, ALTER
  - Always blocks dangerous operations: DROP, TRUNCATE, transaction control, GRANT/REVOKE
  - Works in both single query and interactive modes

### v0.39.0 - 2025-12-15

#### Added

- Python API now supports `model_name` and `api_key` parameters for programmatic model configuration
  - Override configured model: `SQLSaber(database="...", model_name="anthropic:claude-sonnet-4-20250514")`
  - Override configured model and API key: `SQLSaber(database="...", model_name="anthropic:claude-sonnet-4-20250514", api_key="sk-...")`
  - Falls back to OAuth if `api_key` is not provided but OAuth is configured


### v0.38.0 - 2025-12-01

#### Changed

- Consolidated display formatting logic into `DisplayManager` for cleaner separation of concerns
  - Markdown table export now handled internally by display methods
  - `show_table_list`, `show_schema_info`, and `show_query_results` auto-detect terminal vs redirected output

#### Fixed

- Fixed LIMIT clause injection when queries end with semicolons

### v0.37.0 - 2025-11-20

#### Added

- Simplified programmatic interface to SQLSaber's capabilities, allowing you to run natural language queries against databases from Python code.

### v0.36.0 - 2025-10-23

#### Added

- `saber db exclude` command to manage schema exclusions without re-adding a connection, plus a `--exclude-schemas` flag when creating connections.
- Model name display in both interactive and non-interactive modes alongside database connection information.

#### Changed

- Schema exclusion configuration now applies to PostgreSQL, MySQL, DuckDB, and CSV connections, combining per-connection settings with environment variables such as `SQLSABER_MYSQL_EXCLUDE_SCHEMAS` and `SQLSABER_DUCKDB_EXCLUDE_SCHEMAS`.

#### Fixed

- Fixed ID column truncation in `saber threads list` on smaller terminal screens by ensuring full UUID visibility.

### v0.35.0 - 2025-10-22

#### Added

- Table and column comment support across all databases
  - Comments are now included in schema introspection to provide richer context to LLM
  - PostgreSQL: Uses `obj_description()` and `col_description()` functions
  - MySQL: Retrieves `table_comment` and `column_comment` from `information_schema`
  - DuckDB: Joins with `duckdb_tables()` and `duckdb_columns()` for comment data
  - SQLite: Returns `None` for comments (SQLite doesn't support native comments)
  - Comments are conditionally included in tool output only when present, avoiding clutter
  - Updated `ColumnInfo` and `SchemaInfo` TypedDicts with optional comment fields

#### Fixed

- Strip leading/trailing whitespace and new lines from submitted user inputs

### v0.34.0 - 2025-10-17

#### Changed

- Improved `auth setup` command: Reworked to select the provider first, offer to reset existing credentials per provider, and allow configuring multiple providers in one session without manual cleanup.
- PostgreSQL schema introspection: Exclude TimescaleDB internal schemas by default and add env-based filtering
  - Default exclusions now include `pg_catalog`, `information_schema`, `_timescaledb_internal`, `_timescaledb_cache`, `_timescaledb_config`, `_timescaledb_catalog`
  - New environment variable `SQLSABER_PG_EXCLUDE_SCHEMAS` allows excluding additional schemas (comma-separated)
  - Applies to both `list_tables` and `introspect_schema` tools

### v0.33.0 - 2025-10-16

#### Changed

- Improved system prompt for GPT-5 using OpenAI's prompt optimizer tool
- Improved system prompt for Sonnet 4.5 and others using Anthropic's prompt improvement tool
- Dedicated prompt for GPT-5

### v0.32.1 - 2025-10-15

#### Fixed

- Prevented `saber auth setup` from crashing when launching Anthropic OAuth by moving the browser authentication flow onto a background thread, avoiding nested asyncio event loops.

### v0.32.0 - 2025-10-14

#### Added

- Optional MLflow autologging
  - Set `MLFLOW_URI` and `MLFLOW_EXP` before running `sqlsaber` to forward telemetry via `mlflow.pydantic_ai.autolog()` with no runtime requirement when the MLflow package is absent.

### v0.31.0 - 2025-10-13

#### Added

- Structured logging across the project
  - Centralized setup with JSON logs to a rotating file by default
  - Daily rotation at midnight (configurable), with backups retained
  - Optional pretty console output in development via `SQLSABER_LOG_TO_STDERR=1` or `SQLSABER_DEBUG=1`
  - Captures stdlib logging and Python warnings into the same pipeline
  - Environment variables to control behavior: `SQLSABER_LOG_LEVEL`, `SQLSABER_LOG_FILE`, `SQLSABER_LOG_ROTATION`, `SQLSABER_LOG_WHEN`, `SQLSABER_LOG_INTERVAL`, `SQLSABER_LOG_BACKUP_COUNT`, `SQLSABER_LOG_MAX_BYTES`

### v0.30.2 - 2025-10-10

#### Changed

- Added `keyrings.cryptfile` as Linux-only dependency for better keyring support on Linux systems

### v0.30.1 - 2025-10-07

#### Fixed

- Minor styling fixes

### v0.30.0 - 2025-10-07

#### Removed

- MCP server support and related console scripts (`saber-mcp`, `sqlsaber-mcp`)

  > `sqlsaber` can still be used as a cli in coding agents like Claude Code, Codex, or Amp.
  >
  > Just _ask_ the coding agent to invoke `sqlsaber` cli with your question.

### v0.29.1 - 2025-10-05

#### Fixed

- Fixed `list_tables` tool displaying "0 total" when tables exist

### v0.29.0 - 2025-10-04

#### Added

- Theme management commands
  - `saber theme set` - Interactively select from all available Pygments themes with searchable list
  - `saber theme reset` - Reset to default theme (nord)
  - Theme configuration persists across sessions
  - Environment variable `SQLSABER_THEME` can override configured theme

#### Changed

- Theme manager now derives semantic colors directly from selected Pygments styles, enabling out-of-the-box support for any upstream theme while retaining user overrides and fallbacks.

### v0.28.0 - 2025-10-03

#### Added

- Unified theming system
- Easy theme switching via `SQLSABER_THEME` environment variable or config file

#### Changed

- Enhanced read-only query validation using `sqlglot` AST analysis
  - Improved security with comprehensive AST-based detection of write operations
  - Blocks dangerous operations in nested queries, CTEs, and subqueries
  - Detects dialect-specific dangerous functions (pg_read_file, LOAD_FILE, readfile, etc.)
  - Prevents SELECT INTO, SELECT FOR UPDATE/SHARE, and multi-statement queries
  - Dialect-aware LIMIT clause injection for Postgres, MySQL, SQLite, and DuckDB

### v0.27.0 - 2025-10-01

#### Added

- Added onboarding flow for new users
  - If users don't have a database set up or model provider API key setup, SQLsaber will guide them through the process interactively in a frictionless and delightful manner.

#### Changed

- Final thinking blocks now render Markdown

### v0.26.0 - 2025-09-30

#### Added

- Extended thinking/reasoning mode
  - Support for Anthropic, OpenAI, Google, and Groq
  - CLI flags: `--thinking` and `--no-thinking` for per-query control
  - Interactive commands: `/thinking on` and `/thinking off` for runtime toggling
  - Configurable default via `thinking_enabled` in model config
  - Thinking content displayed with dim styling to distinguish from final answers

> This is useful when you want to see the reasoning used to generate the answer or when dealing with complex queries and questions.

### v0.25.0 - 2025-09-26

#### Added

- DuckDB support for efficient CSV and data file analysis
  - Added `duckdb` provider to database options
  - Enhanced CSV processing capabilities using DuckDB's optimized engine

### v0.24.0 - 2025-09-24

#### Added

- Index information is now included in `introspect_schema` tool output alongside columns, primary keys, and foreign keys
- Cleaner markdown formatting for thread exports when redirected to files
  - Terminal display remains unchanged with rich styling and colors

### v0.23.0 - 2025-09-16

#### Added

- Smoother markdown streaming and prevent duplicate messages
- SQL execution errors now display in interactive sessions instead of being silently dropped

### v0.22.0 - 2025-09-15

#### Added

- Query timeout protection to prevent runaway queries
  - 30-second timeout applied to all database operations by default
  - Both client-side and server-side timeout enforcement where supported (PostgreSQL, MySQL)
  - Per-query timeout override parameter for edge cases
  - Automatic rollback of transactions when queries timeout

### v0.21.0 - 2025-09-15

#### Changed

- Use Responses API for OpenAI models
- Stream markdown while streaming response from models

---

### v0.20.0 - 2025-09-10

#### Fixed

- Subcommand help visibility for nested apps
- Removed mouse support for user prompt
  - Enabling mouse support disables terminal scrolling when user prompt is focused

### v0.19.0 - 2025-09-09

#### Added

- Notable improvements to user prompt interface
  - Ctrl+D to exit the application without having to use "/exit" or "/quit" slash command
  - Display multiline input submission info as bottom toolbar
  - Pressing up and down arrow keys now help navigate through prior user prompts
  - Visual improvements to user prompt area
  - Added mouse support - users can now click around text to edit

### v0.18.0 - 2025-09-08

#### Changed

- Improved CLI startup time

### v0.17.0 - 2025-09-08

#### Added

- Conversation threads system for storing, displaying, and resuming conversations
  - Automatic thread creation for both interactive and non-interactive sessions
  - `saber threads list` - List all conversation threads with filtering options
  - `saber threads show THREAD_ID` - Display full transcript of a conversation thread
  - `saber threads resume THREAD_ID` - Continue a previous conversation in interactive mode
  - `saber threads prune` - Clean up old threads based on age
  - Thread persistence with metadata (title, model, database, last activity)
  - Seamless resumption of conversation context and history

#### Removed

- Removed visualization tools and plotting capabilities
  - Removed PlotDataTool and uniplot dependency
  - Cleaned up visualization-related code from CLI, registry, and instructions

### v0.16.1 - 2025-09-04

#### Added

- Compile python byte code during installation
- Updated CLI help string

### v0.16.0 - 2025-09-04

#### Added

- Migrated to Pydantic-AI agent runtime with model-agnostic interfaces
- Added multi-provider model support: Anthropic, OpenAI, Google, Groq, Mistral, Cohere, Hugging Face
- Provider registry tests to ensure invariants and alias normalization

#### Changed

- Reworked agents to use new pydantic-ai-based agent implementation
- Updated CLI modules and settings to integrate provider selection and authentication
- `saber auth reset` now mirrors setup by prompting for a provider, then selectively removing stored credentials for that provider
  - Removes API keys from OS credential store for the selected provider
  - For Anthropic, also detects and removes OAuth tokens
  - Offers optional prompt to unset global auth method when Anthropic OAuth is removed
- Centralized provider definitions in `sqlsaber.config.providers` and refactored CLI, config, and agent code to use the registry (single source of truth)
- Normalized provider aliases (e.g., `google-gla` → `google`) for consistent behavior across modules

#### Removed

- Deprecated custom `clients` module and Anthropic-specific client code
- Removed legacy streaming and events modules and related tests

### v0.15.0 - 2025-08-18

#### Added

- Tool abstraction system with centralized registry (new `Tool` base class, `ToolRegistry`, decorators)
- Dynamic instruction generation system (`InstructionBuilder`)
- Comprehensive test suite for the tools module

#### Changed

- Refactored agents to use centralized tool registry instead of hardcoded tools
- Enhanced MCP server with dynamic tool registration
- Moved core SQL functionality to dedicated tool classes

### v0.14.0 - 2025-08-01

#### Added

- Local conversation storage between user and agent
  - Store conversation history persistently
  - Track messages with proper attribution
- Added automated test execution in CI
  - New GitHub Actions workflow for running tests
  - Updated code review workflow

#### Fixed

- Fixed CLI commands test suite (#11)

#### Changed

- Removed schema caching from SchemaManager
  - Simplified schema introspection by removing cache logic
  - Direct database queries for schema information

### v0.13.0 - 2025-07-26

#### Added

- Database resolver abstraction for unified connection handling
  - Extended `-d` flag to accept PostgreSQL and MySQL connection strings (e.g., `postgresql://user:pass@host:5432/db`)
  - Support for direct connection strings alongside existing file path and configured database support
  - Examples: `saber -d "postgresql://user:pass@host:5432/db" "show users"`

### v0.12.0 - 2025-07-23

#### Added

- Add support for ad-hoc SQLite files via `--database`/`-d` flag

### v0.11.0 - 2025-07-09

#### Changed

- Removed row counting from `list_tables` tool for all database types

### v0.10.0 - 2025-07-08

#### Added

- Support for reading queries from stdin via pipe operator
  - `echo 'show me all users' | saber` now works
  - `cat query.txt | saber` reads query from file via stdin
  - Allows integration with other command-line tools and scripts

### v0.9.0 - 2025-07-08

#### Changed

- Migrated from Typer to Cyclopts for CLI framework
  - Improved command structure and parameter handling
  - Better support for sub-commands and help documentation
- Made interactive mode more ergonomic
  - `saber` now directly starts interactive mode (previously `saber query`)
  - `saber "question"` executes a single query (previously `saber query "question"`)
  - Removed the `query` subcommand for a cleaner interface

### v0.8.2 - 2025-07-08

#### Changed

- Updated formatting for final answer display
- New ASCII art in interactive mode

### v0.8.1 - 2025-07-07

#### Fixed

- Fixed OAuth validation logic to not require API key when Claude Pro OAuth is configured

### v0.8.0 - 2025-07-07

#### Added

- OAuth support for Claude Pro/Max subscriptions
- Authentication management with `saber auth` command
  - Interactive setup for API key or Claude Pro/Max subscription
  - `saber auth setup`
  - `saber auth status`
  - `saber auth reset`
  - Persistent storage of user authentication preferences
- New `clients` module with custom Anthropic API client
  - `AnthropicClient` for direct API communication

#### Changed

- Enhanced authentication system to support both API keys and OAuth tokens
- Replaced Anthropic SDK with direct API implementation using httpx
- Modernized type annotations throughout the codebase
- Refactored query streaming into smaller, more maintainable functions

### v0.7.0 - 2025-07-01

#### Added

- Table name autocomplete with "@" prefix in interactive mode

  - Type "@" followed by table name to get fuzzy matching completions
  - Supports schema-aware completions (e.g., "@sample" matches "public.sample")

- Rich markdown display for assistant responses
  - After streaming completes, the final response is displayed as formatted markdown

### v0.6.0 - 2025-06-30

#### Added

- Slash command autocomplete in interactive mode
  - Commands now use slash prefix: `/clear`, `/exit`, `/quit`
  - Autocomplete shows when typing `/` at the start of a line
  - Press Tab to select suggestion
- Query interruption with Ctrl+C in interactive mode
  - Press Ctrl+C during query execution to gracefully cancel ongoing operations
  - Preserves conversation history up to the interruption point

#### Changed

- Updated table display for better readability: limit to first 15 columns on wide tables
  - Shows warning when columns are truncated
- Interactive commands now require slash prefix (breaking change)
  - `clear` → `/clear`
  - `exit` → `/exit`
  - `quit` → `/quit`
- Removed default limit of 100. Now model will decide it.

### v0.5.0 - 2025-06-27

#### Added

- Added support for plotting data from query results.
  - The agent can decide if plotting will useful and create a plot with query results.
- Small updates to system prompt

### v0.4.1 - 2025-06-26

#### Added

- Show connected database information at the start of a session
- Update welcome message for clarity

### v0.4.0 - 2025-06-25

#### Added

- MCP (Model Context Protocol) server support
- `saber-mcp` console script for running MCP server
- MCP tools: `get_databases()`, `list_tables()`, `introspect_schema()`, `execute_sql()`
- Instructions and documentation for configuring MCP clients (Claude Code, etc.)

### v0.3.0 - 2025-06-25

#### Added

- Support for CSV files as a database option: `saber query -d mydata.csv`

#### Changed

- Extracted tools to BaseSQLAgent for better inheritance across SQLAgents

#### Fixed

- Fixed getting row counts for SQLite

### v0.2.0 - 2025-06-24

#### Added

- SSL support for database connections during configuration
- Memory feature similar to Claude Code
- Support for SQLite and MySQL databases
- Model configuration (configure, select, set, reset) - Anthropic models only
- Comprehensive database command to securely store multiple database connection info
- API key storage using keyring for security
- Interactive questionary for all user interactions
- Test suite implementation

#### Changed

- Package renamed from original name to sqlsaber
- Better configuration handling
- Simplified CLI interface
- Refactored query stream function into smaller functions
- Interactive markup cleanup
- Extracted table display functionality
- Refactored and cleaned up codebase structure

#### Fixed

- Fixed list_tables tool functionality
- Fixed introspect schema tool
- Fixed minor type checking errors
- Check before adding new database to prevent duplicates

#### Removed

- Removed write support completely for security

### v0.1.0 - 2025-06-19

#### Added

- First working version of SQLSaber
- Streaming tool response and status messages
- Schema introspection with table listing
- Result row streaming as agent works
- Database connection and query capabilities
- Added publish workflow
- Created documentation and README
- Added CLAUDE.md for development instructions
