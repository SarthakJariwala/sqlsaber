---
title: Commands
description: CLI reference for SQLsaber. Database connections, authentication, models, knowledge, threads, and query options.
---

Options, flags, and slash commands for the `saber` CLI.

## `saber`

Runs a query. With no question, starts interactive mode.

```bash
# Interactive mode (default)
saber

# Single query
saber "How many users do we have?"

# Named connection
saber -d my-database "Show me recent orders"

# Connection string
saber -d "postgresql://user:pass@host:5432/db" "User statistics for 2024"

# Several databases (repeat -d)
saber -d sales -d analytics "Compare revenue to web sessions"

# One follow-up on a saved thread
saber --thread a1b2c3d4 "Now compare that with last quarter"
```

**Parameters:**

- `QUERY-TEXT` - Natural-language question. Optional. Omit it to start interactive mode.
- `-d, --database` - Saved connection name, file path (CSV, Parquet, SQLite, or DuckDB), or connection string (`postgresql://`, `mysql://`, `duckdb://`, `csv:///`, `parquet:///`). Repeat the flag to connect [several databases](/guides/multi-database/) at once, or to load several CSV or Parquet files into one DuckDB session with one table per file.
- `--thinking` / `--no-thinking` - Enable or disable extended thinking for this run.
- `--csv-tool-results` / `--no-csv-tool-results` - Opt in to experimental CSV tables in model-facing SQL tool results. JSON is the default. Applies to single queries and interactive mode.
- `--allow-dangerous` - Allow `INSERT`, `UPDATE`, `DELETE`, and restricted DDL (`CREATE TABLE`, `CREATE VIEW`, `CREATE INDEX`, `ALTER TABLE`). `DROP`, `TRUNCATE`, and admin or security operations stay blocked. `UPDATE` and `DELETE` require `WHERE`.
- `--system-prompt` - Custom system prompt text, or a path to a file. Overrides the built-in prompt.
- `--thread` - Continue a saved thread without interactive mode. Requires a query. Uses the stored saved connection unless `-d` overrides it.

**Global options:**

- `--help, -h` - Show help
- `--version` - Show the version

## Experimental CSV tool results

```bash
# One query (also works with a question piped through stdin)
saber --csv-tool-results -d analytics "Show recent orders"

# Interactive session
saber --csv-tool-results -d analytics

# Resume interactively, opting in for new tool calls
saber threads resume THREAD_ID --csv-tool-results

# Resume for a single follow-up
saber --thread THREAD_ID --csv-tool-results "Compare with last month"
```

JSON remains the default. CSV applies only to tabular results sent to the model by `list_tables`, `introspect_schema`, `execute_sql`, and `list_dbs`. Metadata, errors, and empty results stay JSON. Displayed terminal tables and complete saved JSON results do not change. SQL previews keep a 12 KiB limit.

This is a session option, not a saved preference. Interactive thread switches keep the current session's choice. A new CLI process defaults to JSON unless the flag is passed again. Existing thread messages are not converted. There is no in-session toggle. Launch with the flag you want.

CSV often reduces token usage for multi-row data. Small results can be larger. The effect on answer quality is not established, so the flag stays opt-in. For Python, see [Configuration](/sdk/configuration/#experimental-csv-tool-results).

---

## `saber auth`

Authentication for AI providers.

### `saber auth setup`

Save an API key.

```bash
saber auth setup
```

### `saber auth status`

Show which providers are configured, and whether each key came from an environment variable or the keychain.

```bash
saber auth status
```

### `saber auth reset`

Remove stored credentials for a provider.

```bash
saber auth reset

# Non-interactive
saber auth reset openai --yes
```

Pass the provider for automation. `--yes` skips confirmation. Without `--yes`, the command prompts only when attached to an interactive terminal.

---

## `saber db`

Saved database connections.

### `saber db add`

Add a connection.

```bash
saber db add my-database [OPTIONS]

# Non-interactive SQLite setup
saber db add local --type sqlite --database ./local.db --no-interactive

# Read a server password from stdin
printf '%s' "$DB_PASSWORD" | saber db add analytics --no-interactive \
  --host db.example.com --database analytics --username agent --password-stdin
```

**Parameters:**

- `NAME` - Connection name (required)

**Options:**

- `-t, --type` - Database type: `postgresql`, `mysql`, `sqlite`, `duckdb` (default: postgresql)
- `-h, --host` - Host
- `-p, --port` - Port
- `--database, --db` - Database name
- `-u, --username` - Username
- `--exclude-schemas` - Comma-separated schemas to skip during introspection
- `--description` - Short description shown to the agent in [multi-database sessions](/guides/multi-database/)
- `--ssl-mode` - SSL mode (see SSL modes below)
- `--ssl-ca` - SSL CA certificate file path
- `--ssl-cert` - SSL client certificate file path
- `--ssl-key` - SSL client private key file path
- `--interactive/--no-interactive` - Interactive prompts (default: true)
- `--password-stdin` - Read the database password from stdin. Requires `--no-interactive`.

**SSL modes:**

_PostgreSQL:_

- `disable` - No SSL
- `allow` - Try SSL, fall back to non-SSL
- `prefer` - Try SSL first (default)
- `require` - Require SSL
- `verify-ca` - Require SSL and verify the certificate
- `verify-full` - Require SSL, verify the certificate and hostname

_MySQL:_

- `DISABLED` - No SSL
- `PREFERRED` - Try SSL first (default)
- `REQUIRED` - Require SSL
- `VERIFY_CA` - Require SSL and verify the certificate
- `VERIFY_IDENTITY` - Require SSL, verify the certificate and hostname

### `saber db list`

List saved connections: names, host, port, database, excluded schemas, and the default marker.

```bash
saber db list
```

### `saber db exclude NAME`

Update or inspect schema exclusions for a saved connection.

```bash
saber db exclude my-database [--set SCHEMAS | --add SCHEMAS | --remove SCHEMAS | --clear]
```

**Options:**

- `--set` - Replace the exclusion list with the given comma-separated schemas
- `--add` - Append schemas (duplicates are ignored)
- `--remove` - Remove the given schemas
- `--clear` - Remove all exclusions

With no flags, the command edits the list interactively.

### `saber db set-default NAME`

Set the default connection.

```bash
saber db set-default my-database
```

### `saber db test NAME`

Test a connection. Prints success or the error details.

```bash
saber db test my-database
```

### `saber db remove`

Remove a connection. Prompts for confirmation in a terminal. `--yes` skips the prompt.

```bash
saber db remove my-database
saber db remove my-database --yes
```

---

## `saber knowledge`

Database-scoped knowledge entries used by the `search_knowledge` tool. Entries may include SQL snippets and source references.

### `saber knowledge add`

```bash
saber knowledge add "Name" "Description" [OPTIONS]
```

**Parameters:**

- `NAME` - Entry name (required)
- `DESCRIPTION` - Description (required)

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)
- `--sql` - SQL query or pattern
- `--source` - Source reference, such as a wiki page or URL

**Examples:**

```bash
saber knowledge add "Revenue KPI" "Recognized revenue from shipped orders only"

saber knowledge add "Monthly revenue rollup" "Use shipped orders for monthly revenue" --sql "SELECT date_trunc('month', shipped_at), SUM(amount) FROM orders WHERE status = 'shipped' GROUP BY 1"

saber knowledge add "NRR definition" "Exclude new logo revenue from NRR" --source "finance-wiki"

saber knowledge add "Revenue definition" "$(cat ./knowledge/revenue_definition.md)"
saber knowledge add "Monthly revenue rollup" "$(cat ./knowledge/monthly_revenue_notes.md)" --sql "$(cat ./sql/monthly_revenue_rollup.sql)"
```

### `saber knowledge list`

Lists ID, name, description preview, and last updated time.

```bash
saber knowledge list [OPTIONS]
```

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)

### `saber knowledge show`

```bash
saber knowledge show ENTRY_ID [OPTIONS]
```

**Parameters:**

- `ENTRY_ID` - ID from `saber knowledge list`

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)

### `saber knowledge search`

```bash
saber knowledge search "QUERY" [OPTIONS]
```

**Parameters:**

- `QUERY` - Keyword query

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)
- `--limit` - Maximum entries to return (default: 10)

Results are ranked by full-text relevance and scoped to one database.

### `saber knowledge remove`

```bash
saber knowledge remove ENTRY_ID [OPTIONS]
```

**Parameters:**

- `ENTRY_ID` - ID from `saber knowledge list`

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)
- `--yes` - Skip confirmation (required when no interactive terminal is available)

### `saber knowledge clear`

Remove every knowledge entry for a database.

```bash
saber knowledge clear [OPTIONS]
```

**Options:**

- `-d, --database` - Saved connection name (uses the default if omitted)
- `--yes` - Skip confirmation

## `saber models`

LLM models from configured providers.

### `saber models list`

```bash
saber models list
```

### `saber models set`

Set the default model and thinking level.

```bash
# Interactive selection
saber models set

# Direct, non-interactive selection
saber models set openai:gpt-5 --thinking-level medium
saber models set openai:gpt-5 --agent handoff
```

**Options:**

- `--agent` - Agent to configure (`main`, `handoff`, `viz`, `notebook`). Defaults to `main`.
- `--thinking-level` - Main-model thinking mode: `off`, `minimal`, `low`, `medium`, `high`, or `maximum`.

### `saber models current`

```bash
saber models current
```

**Options:**

- `--agent` - Show the model for one agent (`main`, `handoff`, `viz`, `notebook`).

### `saber models reset`

Reset to `openai:gpt-5.6-sol`.

```bash
saber models reset
saber models reset --agent handoff --yes
```

**Options:**

- `--agent` - Agent to reset (`main`, `handoff`, `viz`, `notebook`). Defaults to `main`.
- `--yes` - Skip confirmation (required when no interactive terminal is available).

---

## `saber theme`

Syntax highlighting theme.

### `saber theme set`

Omit the theme name to browse interactively.

```bash
saber theme set
saber theme set dracula
```

Override the theme for one process:

```bash
export SQLSABER_THEME=dracula
saber
```

### `saber theme reset`

Reset to the default theme (`nord`).

```bash
saber theme reset
saber theme reset --yes
```

`--yes` skips confirmation and is required when no interactive terminal is available.

---

## `saber threads`

Saved conversation threads.

### `saber threads list`

```bash
saber threads list [OPTIONS]
```

**Options:**

- `-d, --database` - Filter by database name
- `-n, --limit` - Maximum threads to return (default: 50)

### `saber threads show`

Prints thread metadata (database, model, timestamps), the full transcript, SQL and results, tool calls, and durable artifact names and links.

```bash
saber threads show a1b2c3d4
```

**Parameters:**

- `THREAD_ID` - ID from `saber threads list`

### `saber threads artifacts`

Lists durable artifacts for a thread without replaying the transcript. Output includes publication ID and kind, artifact kind, name, size, local URI, and an unavailable marker when integrity verification fails.

```bash
saber threads artifacts a1b2c3d4
```

### `saber threads export`

Writes a standalone HTML transcript. Default path is `./thread-<id>.html`.

```bash
saber threads export a1b2c3d4
saber threads export a1b2c3d4 --output analysis.html
```

**Parameters:**

- `THREAD_ID` - ID from `saber threads list`

**Options:**

- `-o, --output` - Output HTML file path

### `saber threads resume`

Resume a thread in interactive mode.

```bash
saber threads resume a1b2c3d4 [OPTIONS]
```

**Parameters:**

- `THREAD_ID` - Thread to resume

**Options:**

- `-d, --database` - Use a different database than the original thread. Repeat the flag to resume against several databases.

Loads the saved messages, uses the currently configured model, and reconnects to the original database or databases, including [multi-database](/guides/multi-database/) threads.

:::note
Automatic resume requires every database on the thread to be a saved connection. If a thread used a connection string or file path, resume it with explicit `-d` flags.
:::

For one follow-up without interactive mode, use the root command:

```bash
saber --thread a1b2c3d4 "Now compare that with last quarter"
```

### `saber threads prune`

Delete threads older than a given number of days.

```bash
saber threads prune
saber threads prune --days 30 --dry-run
saber threads prune --days 30 --yes
```

**Options:**

- `-n, --days` - Delete threads older than this many days (default: 30)
- `--dry-run` - Report how many threads would be deleted, without deleting them
- `--yes` - Skip confirmation (required when no interactive terminal is available)

---

## Interactive mode

`saber` with no question starts interactive mode. Session commands:

- `/help [GROUP [COMMAND]]` - Show slash-command help (`/?` is an alias)
- `/clear` - Clear conversation history
- `/exit` - End the session (`/quit` is an alias)
- `/thinking` - Show current thinking status and level
- `/thinking on` - Enable extended thinking at the current level
- `/thinking off` - Disable extended thinking
- `/thinking <level>` - Set thinking level (also enables thinking)
- `/handoff GOAL` - Draft a prompt for a new thread from the current context

Management commands from the CLI also work with a leading `/`. Examples: `/db list`, `/auth status`, `/threads resume ID`. Type `/help` for the full list.

**Thinking levels:**

| Level | Description |
|-------|-------------|
| `off` | No extended thinking |
| `minimal` | Least reasoning |
| `low` | Light reasoning |
| `medium` | Default balance of cost and quality |
| `high` | Deeper reasoning |
| `maximum` | Highest reasoning depth and cost |

**Autocomplete:**

- Table names: type `@table_name` and press Tab
- Slash commands: type `/` and press Tab

---

## Environment variables

- `SQLSABER_THEME` - Override the configured theme for this process
- `SQLSABER_PG_EXCLUDE_SCHEMAS` - Extra PostgreSQL schemas to skip during discovery and introspection. Defaults already skip `pg_catalog`, `information_schema`, `_timescaledb_internal`, `_timescaledb_cache`, `_timescaledb_config`, `_timescaledb_catalog`.
- `SQLSABER_MYSQL_EXCLUDE_SCHEMAS` - Extra MySQL databases to skip. Defaults skip `information_schema`, `performance_schema`, `mysql`, and `sys`.
- `SQLSABER_DUCKDB_EXCLUDE_SCHEMAS` - Extra DuckDB schemas to skip. Defaults skip `information_schema`, `pg_catalog`, and `duckdb_catalog`.
