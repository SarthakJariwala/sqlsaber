---
title: Command Reference
description: Complete reference for all SQLsaber commands
---

This is a comprehensive reference for all SQLsaber commands and their options.

### `saber`

The main SQLsaber command for running queries.

**Usage:**

```bash
# Interactive mode (default)
saber

# Single query
saber "How many users do we have?"

# With specific database
saber -d my-database "Show me recent orders"

# With connection string
saber -d "postgresql://user:pass@host:5432/db" "User statistics for 2024"
```

**Parameters:**

- `QUERY-TEXT` - SQL query in natural language (optional, starts interactive mode if not provided)
- `-d, --database` - Database connection name, file path (CSV/SQLite/DuckDB), or connection string (postgresql://, mysql://, duckdb://)
- `--thinking` / `--no-thinking` - Enable/disable extended thinking/reasoning mode

**Global Options:**

- `--help, -h` - Display help message
- `--version` - Show version information

---

### `saber auth`

Manage authentication configuration for AI providers.

#### `saber auth setup`

Configure authentication for SQLsaber (API keys).

**Usage:**

```bash
saber auth setup
```

#### `saber auth status`

Check current authentication configuration.

**Usage:**

```bash
saber auth status
```

**Output shows:**

- Configured providers

#### `saber auth reset`

Remove stored credentials for a provider.

**Usage:**

```bash
saber auth reset
```

---

### `saber db`

Manage database connections.

#### `saber db add`

Add a new database connection.

**Usage:**

```bash
saber db add my-database [OPTIONS]
```

**Parameters:**

- `NAME` - Name for the database connection (required)

**Options:**

- `-t, --type` - Database type: `postgresql`, `mysql`, `sqlite`, `duckdb` (default: postgresql)
- `-h, --host` - Database host
- `-p, --port` - Database port
- `--database, --db` - Database name
- `-u, --username` - Username
- `--exclude-schemas` - Comma-separated list of schemas to skip during introspection
- `--ssl-mode` - SSL mode (see SSL options below)
- `--ssl-ca` - SSL CA certificate file path
- `--ssl-cert` - SSL client certificate file path
- `--ssl-key` - SSL client private key file path
- `--interactive/--no-interactive` - Use interactive mode (default: true)

**SSL Modes:**

_PostgreSQL:_

- `disable` - No SSL
- `allow` - Try SSL, fallback to non-SSL
- `prefer` - Try SSL first (default)
- `require` - Require SSL
- `verify-ca` - Require SSL and verify certificate
- `verify-full` - Require SSL, verify certificate and hostname

_MySQL:_

- `DISABLED` - No SSL
- `PREFERRED` - Try SSL first (default)
- `REQUIRED` - Require SSL
- `VERIFY_CA` - Require SSL and verify certificate
- `VERIFY_IDENTITY` - Require SSL, verify certificate and hostname

#### `saber db list`

List all configured database connections.

**Usage:**

```bash
saber db list
```

**Output shows:**

- Database names
- Connection details (host, port, database)
- Any excluded schemas configured for the connection
- Default database indicator

#### `saber db exclude NAME`

Update or inspect schema exclusions for an existing database connection.

**Usage:**

```bash
saber db exclude my-database [--set SCHEMAS | --add SCHEMAS | --remove SCHEMAS | --clear]
```

**Options:**

- `--set` — Replace the exclusion list entirely with the provided comma-separated schemas
- `--add` — Append schemas to the current exclusion list (duplicates are ignored)
- `--remove` — Remove the provided schemas from the exclusion list
- `--clear` — Remove all exclusions

Run without flags to interactively edit the exclusion list.

#### `saber db set-default NAME`

Set a database as the default connection.

**Usage:**

```bash
saber db set-default my-database
```

#### `saber db test NAME`

Test a database connection.

**Usage:**

```bash
saber db test my-database
```

**Output:**

- Connection success/failure
- Error details if connection fails

#### `saber db remove`

Remove a database connection.

**Usage:**

```bash
saber db remove my-database
```

**Confirmation required** - Will prompt before deletion.

---

### `saber memory`

Manage database-specific memories and context.

#### `saber memory add`

Add a new memory entry.

**Usage:**

```bash
saber memory add "Memory content here" [OPTIONS]
```

**Parameters:**

- `CONTENT` - Memory content to add (required)

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)

**Examples:**

```bash
# Add memory to default database
saber memory add "Active customers are those who made a purchase in the last 90 days"

# Add memory to specific database
saber memory add -d prod-db "Revenue is recognized when orders are shipped"

# Business rules
saber memory add "VIP customers have lifetime_value > 10000"

# Formatting preferences
saber memory add "Always format dates as YYYY-MM-DD for reports"
```

#### `saber memory list`

List all memory entries for a database.

**Usage:**

```bash
saber memory list [OPTIONS]
```

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)

**Output shows:**

- Memory ID
- Memory content
- Creation timestamp

#### `saber memory remove`

Remove a specific memory entry.

**Usage:**

```bash
saber memory remove a1b2c3d4
```

**Parameters:**

- `ID` - Memory ID from `saber memory list` output

#### `saber memory clear`

Remove all memory entries for a database.

**Usage:**

```bash
saber memory clear [OPTIONS]
```

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)

**Confirmation required** - Will prompt before deletion.

---

### `saber models`

Manage LLM models from different providers.

#### `saber models list`

List all available models for configured providers.

**Usage:**

```bash
saber models list
```

#### `saber models set`

Set the default model and configure thinking level.

**Usage:**

```bash
saber models set
```

#### `saber models current`

Show the currently configured model and thinking settings.

**Usage:**

```bash
saber models current
```

#### `saber models reset`

Reset to the default model (Claude Sonnet 4).

**Usage:**

```bash
saber models reset
```

---

### `saber theme`

Manage syntax highlighting theme settings.

#### `saber theme set`

Interactively select a syntax highlighting theme from all available Pygments themes.

**Usage:**

```bash
saber theme set
```

You can also set themes via environment variable:

```bash
export SQLSABER_THEME=dracula
saber
```

#### `saber theme reset`

Reset to the default theme (nord).

**Usage:**

```bash
saber theme reset
```

---

### `saber threads`

Manage conversation threads.

#### `saber threads list`

List conversation threads.

**Usage:**

```bash
saber threads list [OPTIONS]
```

**Options:**

- `-d, --database` - Filter by database name
- `-n, --limit` - Maximum threads to return (default: 50)

#### `saber threads show`

Show complete thread transcript.

**Usage:**

```bash
saber threads show a1b2c3d4
```

**Parameters:**

- `THREAD_ID` - Thread ID from `saber threads list`

**Output shows:**

- Thread metadata (database, model, timestamps)
- Complete conversation history
- SQL queries and results
- Tool calls and responses

#### `saber threads resume`

Resume an existing conversation thread.

**Usage:**

```bash
saber threads resume a1b2c3d4 [OPTIONS]
```

**Parameters:**

- `THREAD_ID` - Thread ID to resume

**Options:**

- `-d, --database` - Use different database than original thread

**Features:**

- Loads full conversation context
- Uses same model as original thread
- Connects to original database
- Continues where conversation left off in interactive mode

#### `saber threads prune`

Clean up old conversation threads.

**Usage:**

```bash
saber threads prune
```

---

### Interactive Mode

When in interactive mode (`saber` with no arguments), you have access to a few additional features:

#### Slash Commands

- `/clear` - Clear conversation history
- `/exit` - Exit SQLsaber
- `/quit` - Exit SQLsaber (alias for `/exit`)
- `/thinking` - Show current thinking status and level
- `/thinking on` - Enable extended thinking with current level
- `/thinking off` - Disable extended thinking
- `/thinking <level>` - Set thinking level (implies enable)

**Thinking Levels:**

| Level | Description |
|-------|-------------|
| `off` | Disable extended thinking |
| `minimal` | Quick responses, minimal reasoning |
| `low` | Light reasoning |
| `medium` | Balanced cost/quality (default) |
| `high` | Deep reasoning |
| `maximum` | Complex problems, highest cost |

#### Autocomplete

- **Table names** - Type `@table_name[TAB]` for completions
- **Slash commands** - Type `/[TAB]` for command completions

---

### Environment Variables

These environment variables adjust runtime behavior:

- `SQLSABER_THEME` — Override the configured theme for the session.
- `SQLSABER_PG_EXCLUDE_SCHEMAS` — Comma-separated list of PostgreSQL schemas to exclude from schema discovery and introspection. Defaults already exclude `pg_catalog`, `information_schema`, `_timescaledb_internal`, `_timescaledb_cache`, `_timescaledb_config`, `_timescaledb_catalog`.
- `SQLSABER_MYSQL_EXCLUDE_SCHEMAS` — Comma-separated list of MySQL databases to omit from discovery. Defaults exclude `information_schema`, `performance_schema`, `mysql`, and `sys`.
- `SQLSABER_DUCKDB_EXCLUDE_SCHEMAS` — Comma-separated list of DuckDB schemas to skip during introspection. Defaults exclude `information_schema`, `pg_catalog`, and `duckdb_catalog`.
