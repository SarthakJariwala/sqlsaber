# Changelog

All notable changes to SQLSaber will be documented in this file.

## [Unreleased]

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
