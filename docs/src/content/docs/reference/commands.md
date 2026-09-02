---
title: Command Reference
description: "Complete CLI reference for SQLsaber commands. Database management, authentication, models, knowledge, threads, and query options."
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

# With multiple databases (repeat -d)
saber -d sales -d analytics "Compare revenue to web sessions"

# Continue a saved thread with one non-interactive follow-up
saber --thread a1b2c3d4 "Now compare that with last quarter"
```

**Parameters:**

- `QUERY-TEXT` - SQL query in natural language (optional, starts interactive mode if not provided)
- `-d, --database` - Database connection name, file path (CSV/SQLite/DuckDB), or connection string (postgresql://, mysql://, duckdb://). Repeat the flag to connect to [multiple databases](/guides/multi-database) at once (or to merge multiple CSV files into one session).
- `--thinking` / `--no-thinking` - Enable/disable extended thinking/reasoning mode
- `--allow-dangerous` - Allow INSERT/UPDATE/DELETE and restricted DDL (CREATE TABLE/VIEW/INDEX, ALTER TABLE). DROP/TRUNCATE and admin/security operations remain blocked; UPDATE/DELETE require WHERE.
- `--system-prompt` - Custom system prompt text or path to a file (overrides built-in prompt)
- `--thread` - Continue a saved thread non-interactively. Requires a query; uses the stored configured database unless `-d` overrides it.

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

# Non-interactive
saber auth reset openai --yes
```

Pass the provider directly for automation. `--yes` skips confirmation; without it,
the command prompts only when attached to an interactive terminal.

---

### `saber db`

Manage database connections.

#### `saber db add`

Add a new database connection.

**Usage:**

```bash
saber db add my-database [OPTIONS]

# Non-interactive SQLite setup
saber db add local --type sqlite --database ./local.db --no-interactive

# Read a server password from stdin instead of an argument or prompt
printf '%s' "$DB_PASSWORD" | saber db add analytics --no-interactive \
  --host db.example.com --database analytics --username agent --password-stdin
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
- `--description` - Short human-readable description of the connection. Shown to the agent in [multi-database sessions](/guides/multi-database) to help it pick the right database.
- `--ssl-mode` - SSL mode (see SSL options below)
- `--ssl-ca` - SSL CA certificate file path
- `--ssl-cert` - SSL client certificate file path
- `--ssl-key` - SSL client private key file path
- `--interactive/--no-interactive` - Use interactive mode (default: true)
- `--password-stdin` - Read the database password from stdin. Requires `--no-interactive`.

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
saber db remove my-database --yes
```

**Confirmation required** - Will prompt before deletion in a terminal. Use `--yes`
for a deliberate non-interactive removal.

---

### `saber knowledge`

Manage database-specific knowledge entries used by the `search_knowledge` tool.

Knowledge entries are scoped per database and support optional SQL snippets and source references.

#### `saber knowledge add`

Add a new knowledge entry.

**Usage:**

```bash
saber knowledge add "Name" "Description" [OPTIONS]
```

**Parameters:**

- `NAME` - Knowledge entry name (required)
- `DESCRIPTION` - Knowledge description (required)

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)
- `--sql` - Optional SQL query or pattern
- `--source` - Optional source reference (wiki, URL, etc.)

**Examples:**

```bash
# Add to default database
saber knowledge add "Revenue KPI" "Recognized revenue from shipped orders only"

# Include SQL pattern
saber knowledge add "Monthly revenue rollup" "Use shipped orders for monthly revenue" --sql "SELECT date_trunc('month', shipped_at), SUM(amount) FROM orders WHERE status = 'shipped' GROUP BY 1"

# Include a source reference
saber knowledge add "NRR definition" "Exclude new logo revenue from NRR" --source "finance-wiki"

# Use files for long content
saber knowledge add "Revenue definition" "$(cat ./knowledge/revenue_definition.md)"
saber knowledge add "Monthly revenue rollup" "$(cat ./knowledge/monthly_revenue_notes.md)" --sql "$(cat ./sql/monthly_revenue_rollup.sql)"
```

#### `saber knowledge list`

List all knowledge entries for a database.

**Usage:**

```bash
saber knowledge list [OPTIONS]
```

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)

**Output shows:**

- Knowledge ID
- Name
- Description preview
- Last updated timestamp

#### `saber knowledge show`

Show a full knowledge entry by ID.

**Usage:**

```bash
saber knowledge show ENTRY_ID [OPTIONS]
```

**Parameters:**

- `ENTRY_ID` - Knowledge ID from `saber knowledge list` output

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)

#### `saber knowledge search`

Search knowledge entries for a database.

**Usage:**

```bash
saber knowledge search "QUERY" [OPTIONS]
```

**Parameters:**

- `QUERY` - Keyword query to search for

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)
- `--limit` - Maximum number of entries to return (default: 10)

**Notes:**

- Results are ranked by full-text relevance.
- Search is database-scoped.

#### `saber knowledge remove`

Remove a specific knowledge entry.

**Usage:**

```bash
saber knowledge remove ENTRY_ID [OPTIONS]
```

**Parameters:**

- `ENTRY_ID` - Knowledge ID from `saber knowledge list` output

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)
- `--yes` - Skip confirmation prompt (required when no interactive terminal is available)

#### `saber knowledge clear`

Remove all knowledge entries for a database.

**Usage:**

```bash
saber knowledge clear [OPTIONS]
```

**Options:**

- `-d, --database` - Database connection name (uses default if not specified)
- `--yes` - Skip confirmation prompt

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
# Interactive selection
saber models set

# Direct, non-interactive selection
saber models set openai:gpt-5 --thinking-level medium
saber models set openai:gpt-5 --agent handoff
```

**Options:**

- `--agent` - Target agent to configure (`main`, `handoff`, `viz`, `notebook`). Defaults to `main`.
- `--thinking-level` - Main-model thinking mode: `off`, `minimal`, `low`, `medium`, `high`, or `maximum`.

#### `saber models current`

Show the currently configured model and thinking settings.

**Usage:**

```bash
saber models current
```

**Options:**

- `--agent` - Show model for a specific agent (`main`, `handoff`, `viz`, `notebook`).

#### `saber models reset`

Reset to the default model (`openai:gpt-5.6-sol`).

**Usage:**

```bash
saber models reset
saber models reset --agent handoff --yes
```

**Options:**

- `--agent` - Reset a specific agent (`main`, `handoff`, `viz`, `notebook`). Defaults to `main`.
- `--yes` - Skip confirmation prompt (required when no interactive terminal is available).

---

### `saber theme`

Manage syntax highlighting theme settings.

#### `saber theme set`

Select a syntax highlighting theme. Omit the theme name to browse interactively.

**Usage:**

```bash
saber theme set
saber theme set dracula
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
saber theme reset --yes
```

`--yes` skips confirmation and is required when no interactive terminal is available.

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
- Durable artifact names and links

#### `saber threads artifacts`

List durable artifacts referenced by a thread without replaying its full transcript.

**Usage:**

```bash
saber threads artifacts a1b2c3d4
```

The output includes publication ID and kind, artifact kind/name/size, local URI,
and an unavailable marker when integrity verification fails.

#### `saber threads resume`

Resume an existing conversation thread.

**Usage:**

```bash
saber threads resume a1b2c3d4 [OPTIONS]
```

**Parameters:**

- `THREAD_ID` - Thread ID to resume

**Options:**

- `-d, --database` - Use a different database than the original thread. Repeat the flag to resume against multiple databases.

**Features:**

- Loads full conversation context
- Uses the currently configured model
- Reconnects to the original database(s), including [multi-database](/guides/multi-database) threads
- Continues where conversation left off in interactive mode

:::note
Automatic resume requires every database in the thread to be a saved connection. If a thread used an ad-hoc connection string or file path, resume it with explicit `-d` flags.
:::

For one automated follow-up rather than an interactive session, use the root
command:

```bash
saber --thread a1b2c3d4 "Now compare that with last quarter"
```

#### `saber threads prune`

Clean up old conversation threads.

**Usage:**

```bash
saber threads prune
saber threads prune --days 30 --dry-run
saber threads prune --days 30 --yes
```

**Options:**

- `-n, --days` - Delete threads older than this many days (default: 30)
- `--dry-run` - Report how many threads would be deleted without deleting them
- `--yes` - Skip confirmation prompt (required when no interactive terminal is available)

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
