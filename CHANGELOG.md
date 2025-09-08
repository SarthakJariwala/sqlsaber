# Changelog

All notable changes to SQLsaber will be documented here.

## [Unreleased]

## [0.18.0] - 2025-09-08

### Changed

- Improved CLI startup time

## [0.17.0] - 2025-09-08

### Added

- Conversation threads system for storing, displaying, and resuming conversations
  - Automatic thread creation for both interactive and non-interactive sessions
  - `saber threads list` - List all conversation threads with filtering options
  - `saber threads show THREAD_ID` - Display full transcript of a conversation thread
  - `saber threads resume THREAD_ID` - Continue a previous conversation in interactive mode
  - `saber threads prune` - Clean up old threads based on age
  - Thread persistence with metadata (title, model, database, last activity)
  - Seamless resumption of conversation context and history

### Removed

- Removed visualization tools and plotting capabilities
  - Removed PlotDataTool and uniplot dependency
  - Cleaned up visualization-related code from CLI, registry, and instructions

## [0.16.1] - 2025-09-04

### Added

- Compile python byte code during installation
- Updated CLI help string

## [0.16.0] - 2025-09-04

### Added

- Migrated to Pydantic-AI agent runtime with model-agnostic interfaces
- Added multi-provider model support: Anthropic, OpenAI, Google, Groq, Mistral, Cohere, Hugging Face
- Provider registry tests to ensure invariants and alias normalization

### Changed

- Reworked agents to use new pydantic-ai-based agent implementation
- Updated CLI modules and settings to integrate provider selection and authentication
- `saber auth reset` now mirrors setup by prompting for a provider, then selectively removing stored credentials for that provider
  - Removes API keys from OS credential store for the selected provider
  - For Anthropic, also detects and removes OAuth tokens
  - Offers optional prompt to unset global auth method when Anthropic OAuth is removed
- Centralized provider definitions in `sqlsaber.config.providers` and refactored CLI, config, and agent code to use the registry (single source of truth)
- Normalized provider aliases (e.g., `google-gla` → `google`) for consistent behavior across modules

### Removed

- Deprecated custom `clients` module and Anthropic-specific client code
- Removed legacy streaming and events modules and related tests

## [0.15.0] - 2025-08-18

### Added

- Tool abstraction system with centralized registry (new `Tool` base class, `ToolRegistry`, decorators)
- Dynamic instruction generation system (`InstructionBuilder`)
- Comprehensive test suite for the tools module

### Changed

- Refactored agents to use centralized tool registry instead of hardcoded tools
- Enhanced MCP server with dynamic tool registration
- Moved core SQL functionality to dedicated tool classes

## [0.14.0] - 2025-08-01

### Added

- Local conversation storage between user and agent
  - Store conversation history persistently
  - Track messages with proper attribution
- Added automated test execution in CI
  - New GitHub Actions workflow for running tests
  - Updated code review workflow

### Fixed

- Fixed CLI commands test suite (#11)

### Changed

- Removed schema caching from SchemaManager
  - Simplified schema introspection by removing cache logic
  - Direct database queries for schema information

## [0.13.0] - 2025-07-26

### Added

- Database resolver abstraction for unified connection handling
  - Extended `-d` flag to accept PostgreSQL and MySQL connection strings (e.g., `postgresql://user:pass@host:5432/db`)
  - Support for direct connection strings alongside existing file path and configured database support
  - Examples: `saber -d "postgresql://user:pass@host:5432/db" "show users"`

## [0.12.0] - 2025-07-23

### Added

- Add support for ad-hoc SQLite files via `--database`/`-d` flag

## [0.11.0] - 2025-07-09

### Changed

- Removed row counting from `list_tables` tool for all database types

## [0.10.0] - 2025-07-08

### Added

- Support for reading queries from stdin via pipe operator
  - `echo 'show me all users' | saber` now works
  - `cat query.txt | saber` reads query from file via stdin
  - Allows integration with other command-line tools and scripts

## [0.9.0] - 2025-07-08

### Changed

- Migrated from Typer to Cyclopts for CLI framework
  - Improved command structure and parameter handling
  - Better support for sub-commands and help documentation
- Made interactive mode more ergonomic
  - `saber` now directly starts interactive mode (previously `saber query`)
  - `saber "question"` executes a single query (previously `saber query "question"`)
  - Removed the `query` subcommand for a cleaner interface

## [0.8.2] - 2025-07-08

## Changed

- Updated formatting for final answer display
- New ASCII art in interactive mode

## [0.8.1] - 2025-07-07

### Fixed

- Fixed OAuth validation logic to not require API key when Claude Pro OAuth is configured

## [0.8.0] - 2025-07-07

### Added

- OAuth support for Claude Pro/Max subscriptions
- Authentication management with `saber auth` command
  - Interactive setup for API key or Claude Pro/Max subscription
  - `saber auth setup`
  - `saber auth status`
  - `saber auth reset`
  - Persistent storage of user authentication preferences
- New `clients` module with custom Anthropic API client
  - `AnthropicClient` for direct API communication

### Changed

- Enhanced authentication system to support both API keys and OAuth tokens
- Replaced Anthropic SDK with direct API implementation using httpx
- Modernized type annotations throughout the codebase
- Refactored query streaming into smaller, more maintainable functions

## [0.7.0] - 2025-07-01

### Added

- Table name autocomplete with "@" prefix in interactive mode

  - Type "@" followed by table name to get fuzzy matching completions
  - Supports schema-aware completions (e.g., "@sample" matches "public.sample")

- Rich markdown display for assistant responses
  - After streaming completes, the final response is displayed as formatted markdown

## [0.6.0] - 2025-06-30

### Added

- Slash command autocomplete in interactive mode
  - Commands now use slash prefix: `/clear`, `/exit`, `/quit`
  - Autocomplete shows when typing `/` at the start of a line
  - Press Tab to select suggestion
- Query interruption with Ctrl+C in interactive mode
  - Press Ctrl+C during query execution to gracefully cancel ongoing operations
  - Preserves conversation history up to the interruption point

### Changed

- Updated table display for better readability: limit to first 15 columns on wide tables
  - Shows warning when columns are truncated
- Interactive commands now require slash prefix (breaking change)
  - `clear` → `/clear`
  - `exit` → `/exit`
  - `quit` → `/quit`
- Removed default limit of 100. Now model will decide it.

## [0.5.0] - 2025-06-27

### Added

- Added support for plotting data from query results.
  - The agent can decide if plotting will useful and create a plot with query results.
- Small updates to system prompt

## [0.4.1] - 2025-06-26

### Added

- Show connected database information at the start of a session
- Update welcome message for clarity

## [0.4.0] - 2025-06-25

### Added

- MCP (Model Context Protocol) server support
- `saber-mcp` console script for running MCP server
- MCP tools: `get_databases()`, `list_tables()`, `introspect_schema()`, `execute_sql()`
- Instructions and documentation for configuring MCP clients (Claude Code, etc.)

## [0.3.0] - 2025-06-25

### Added

- Support for CSV files as a database option: `saber query -d mydata.csv`

### Changed

- Extracted tools to BaseSQLAgent for better inheritance across SQLAgents

### Fixed

- Fixed getting row counts for SQLite

## [0.2.0] - 2025-06-24

### Added

- SSL support for database connections during configuration
- Memory feature similar to Claude Code
- Support for SQLite and MySQL databases
- Model configuration (configure, select, set, reset) - Anthropic models only
- Comprehensive database command to securely store multiple database connection info
- API key storage using keyring for security
- Interactive questionary for all user interactions
- Test suite implementation

### Changed

- Package renamed from original name to sqlsaber
- Better configuration handling
- Simplified CLI interface
- Refactored query stream function into smaller functions
- Interactive markup cleanup
- Extracted table display functionality
- Refactored and cleaned up codebase structure

### Fixed

- Fixed list_tables tool functionality
- Fixed introspect schema tool
- Fixed minor type checking errors
- Check before adding new database to prevent duplicates

### Removed

- Removed write support completely for security

## [0.1.0] - 2025-06-19

### Added

- First working version of SQLSaber
- Streaming tool response and status messages
- Schema introspection with table listing
- Result row streaming as agent works
- Database connection and query capabilities
- Added publish workflow
- Created documentation and README
- Added CLAUDE.md for development instructions
