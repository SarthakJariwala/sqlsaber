# Changelog

All notable changes to SQLSaber will be documented in this file.

## [Unreleased]

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
