---
title: MCP server
description: Connect coding agents and other MCP clients to SQLsaber's read-only database tools.
---

`saber mcp` exposes database tools through [FastMCP](https://gofastmcp.com/).
Your MCP client supplies the agent and model. SQLsaber discovers schemas and
executes SQL without model credentials, an internal LLM, or conversation storage.
FastMCP is included in the normal SQLsaber installation.

## Stdio

Configure your client to launch SQLsaber as a subprocess:

```json
{
  "mcpServers": {
    "sqlsaber": {
      "command": "saber",
      "args": ["mcp", "-d", "analytics"]
    }
  }
}
```

Use the absolute path to `saber` if your client does not inherit your shell's PATH.
The process runs as your local user and reads that user's saved database settings.
Stdin/stdout carry only MCP messages; diagnostics go to stderr. The client owns
the subprocess lifecycle. No interactive setup or confirmation prompts run.

## Local HTTP

```bash
saber mcp -d analytics --transport http --port 8000
```

Connect a Streamable HTTP client to `http://127.0.0.1:8000/mcp` on the **same machine**.
For clients using the common `mcpServers` configuration format:

```json
{
  "mcpServers": {
    "sqlsaber": { "url": "http://127.0.0.1:8000/mcp" }
  }
}
```

Client configuration formats vary; select Streamable HTTP, not legacy SSE.
The command binds only to loopback and validates HTTP Host/Origin headers.
**There is no authentication.** Other local processes can access the selected
databases. Do not expose this endpoint through a public proxy or tunnel.
Stop the server with Ctrl+C when finished.

## Database selection

```bash
saber mcp                                # saved default database
saber mcp -d analytics                   # saved connection
saber mcp -d ./data.db                    # SQLite file (prefer absolute paths in client config)
saber mcp -d sales -d analytics           # two selected databases
saber mcp -d ./orders.csv -d ./users.csv   # one DuckDB database with two file tables
```

Database selection follows the regular CLI: saved names, connection strings,
SQLite/DuckDB files, and CSV/Parquet files are supported. Only selected databases
are exposed. Tool calls cannot add connections or supply arbitrary connection
strings. Prefer saved names over credentials in command arguments.

| Tool | Arguments | Response |
| --- | --- | --- |
| `list_dbs` | None | `databases`: names, dialects, display names, descriptions |
| `list_tables` | Optional `db_name` | `db_name`, tables and discovery metadata |
| `introspect_schema` | Optional `db_name`, `table_pattern` | `db_name`, `tables`: columns, keys, indexes |
| `execute_sql` | Required `query`, optional `db_name` | `db_name`, `results`, `row_count`, `row_limit`, `truncated` |

Use `list_dbs` to obtain aliases. Omit `db_name` only when one database is selected.
An explicit unknown name is always an error. `table_pattern` uses SQL LIKE syntax,
such as `main.user%` or `%orders`.

## Query behavior and limits

- Only one read-only SELECT-like statement is allowed. Existing SQL guards and
  database-level read-only controls apply. There is no MCP dangerous/write mode.
- At most **1,000 rows** are returned. SQLsaber caps the top-level SQL limit before
  fetching; nested or oversized limits do not bypass the bound. Smaller limits,
  ordering, and offsets remain intact. Use nonnegative integer limits; limit
  expressions and options such as `WITH TIES` are not supported.
- `row_count` is the number of returned rows, not the total matching count.
  `truncated` means additional rows were omitted. Use `COUNT(*)` for totals.
- Structured payloads are limited to **1 MB**. An oversized result produces an
  MCP tool error asking you to narrow the query; cell values are not silently cut.
- Decimals are exact strings, dates/times ISO strings, UUIDs strings, and binary
  values base64 strings. Non-finite floats and unsupported values produce errors;
  cast them in SQL when necessary.
- Database query timeouts apply (normally 30 seconds). Database operations are
  serialized per server. A disconnected client's active operation is allowed to
  finish cleanup before the next operation starts.
- Errors are MCP tool errors, not successful responses containing an error field.
  Driver details and connection strings are not included in tool error messages.

Row limits do not cap all database CPU or memory use: aggregates and large values
can still be expensive. Use least-privilege database credentials. Excluded schemas
filter discovery; they are **not** an authorization boundary for SQL queries.

MCP mode does not expose plugins, knowledge mutations, retained query artifacts,
or conversation tools. To embed SQLsaber's own natural-language agent instead,
use [RPC mode](/reference/rpc/).
