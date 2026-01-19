# SQLsaber

> SQLsaber is an open-source agentic SQL assistant. Think Claude Code but for SQL.

**Ask questions about your data in plain English. SQLsaber writes SQL queries, executes them, and explains the results.**

![demo](./sqlsaber.gif)

SQLsaber reads your schema, writes SQL, executes it, and explains the results—all from a single natural language prompt.

```bash
$ saber "How many orders were placed last month?"

# SQLsaber will:
# 1. Discover relevant tables (orders, order_items, etc.)
# 2. Analyze their schema
# 3. Generate and run SQL
# 4. Return results with explanation
```

## Why SQLsaber?

- **No context switching** — Stay in your terminal, ask questions, get answers
- **Schema-aware** — Automatically introspects your database; no manual setup
- **Safe by default** — Read-only queries; won't modify your data
- **Works with your stack** — PostgreSQL, MySQL, SQLite, DuckDB, and CSV files

## Install

```bash
# Recommended
uv tool install sqlsaber

# Or with pipx
pipx install sqlsaber
```

Then run:

```bash
saber
```

On first launch, SQLsaber walks you through connecting a database and setting up authentication.

## Quick Examples

```bash
# Interactive mode
saber

# Single query
saber "show me users who signed up this week"

# Pipe from stdin
echo "top 10 customers by revenue" | saber

# Use a specific database
saber -d mydb "count active subscriptions"
```

## Features

| Feature | Description |
|---------|-------------|
| **Schema introspection** | Discovers tables, columns, and relationships automatically |
| **Conversation memory** | Add context like "dates are stored as Unix timestamps" |
| **Thread history** | Resume past conversations with `saber threads resume` |
| **Extended thinking** | Enable `--thinking` for complex analytical queries |
| **Multiple AI providers** | Anthropic, OpenAI, Google, Groq (Claude Opus 4.5 by default) |

## Configuration

### Add a database

```bash
saber db add mydb
# Prompts for connection details
```

Or pass a connection string directly:

```bash
saber -d "postgresql://user:pass@localhost:5432/mydb" "count users"
```

### Change AI model

```bash
saber models set
```

### Add memory (context for your database)

```bash
saber memory add "customer_id in orders table references users.id"
saber memory add "all timestamps are UTC"
```

## How It Works

1. **Discovery** — Lists tables and identifies relevant ones based on your question
2. **Schema analysis** — Introspects only the tables needed
3. **Query generation** — Writes SQL tailored to your database dialect
4. **Execution** — Runs read-only queries with safety checks
5. **Results** — Formats output with explanations

## Documentation

Full docs at [sqlsaber.com](https://sqlsaber.com):

- [Installation](https://sqlsaber.com/installation/)
- [Getting Started](https://sqlsaber.com/guides/getting-started/)
- [Database Setup](https://sqlsaber.com/guides/database-setup/)
- [Command Reference](https://sqlsaber.com/reference/commands/)

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

If you find SQLsaber useful, a ⭐ on GitHub helps others discover it.

## License

Apache-2.0 — see [LICENSE](./LICENSE)
