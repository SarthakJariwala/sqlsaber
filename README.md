# SQLsaber

[![PyPI](https://img.shields.io/pypi/v/sqlsaber.svg)](https://pypi.org/project/sqlsaber/)
[![Docs](https://img.shields.io/badge/docs-sqlsaber.com-blue)](https://sqlsaber.com)

> SQLsaber is an open-source agentic SQL assistant. Think Claude Code but for SQL.

Ask questions about databases, SQLite/DuckDB files, and CSVs in plain English from your terminal or Python code. SQLsaber reads your schema, writes SQL, executes read-only queries by default, and explains the results.

![SQLsaber demo showing a natural language database query in the terminal](./sqlsaber.gif)

Featured in research: SQLsaber appears in an ACM Conference on AI and Agentic Systems '26 paper. [Read the paper](https://dl.acm.org/doi/10.1145/3786335.3813217).

## Quickstart

```bash
# Recommended
uv tool install sqlsaber
```

Try SQLsaber with the sample SQLite database:

```bash
curl -L -o legislators.db https://github.com/SarthakJariwala/sqlsaber/raw/refs/heads/main/legislators.db

saber -d ./legislators.db "How many VPs became president by election in the 20th century?"
```

Or connect your own database:

```bash
saber db add analytics
saber "Show me revenue by month"
```

On first launch, SQLsaber walks you through connecting a database and setting up authentication.

## Use it with your data

```bash
# Interactive mode
saber

# Single question
saber "show me users who signed up this week"

# Pipe from stdin
echo "top 10 customers by revenue" | saber

# Use a saved database connection
saber -d analytics "count active subscriptions"

# Use a connection string directly
saber -d "postgresql://user:pass@localhost:5432/mydb" "count users"

# Query local files
saber -d ./customers.csv "How many customers are from each state?"
saber -d ./warehouse.duckdb "Show me the latest partition"

# Connect multiple databases in one session
saber -d sales -d analytics "Compare last month's revenue to web sessions"
```

## Why SQLsaber?

- **No context switching** — Stay in your terminal, ask questions, get answers.
- **Schema-aware** — Automatically discovers tables, columns, indexes, comments, and relationships.
- **Safe by default** — Runs read-only queries unless you explicitly enable dangerous mode.
- **Works with your stack** — PostgreSQL, MySQL, SQLite, DuckDB, and CSV files.
- **Remembers your work** — Resume previous analysis with conversation threads.
- **Learns your business context** — Store KPI definitions, SQL patterns, and domain notes in a searchable knowledge base.
- **Flexible model support** — Use Anthropic, OpenAI, Google, Groq, Mistral, Cohere, Hugging Face, and other supported providers.

## Common workflows

| Workflow | Command |
| --- | --- |
| Explore data interactively | `saber` |
| Ask a one-off question | `saber "monthly active users"` |
| Analyze a CSV | `saber -d ./customers.csv "customers by state"` |
| Compare multiple databases | `saber -d sales -d analytics "compare revenue to traffic"` |
| Save a KPI definition | `saber knowledge add "Revenue KPI" "Recognized revenue from shipped orders only"` |
| Resume previous analysis | `saber threads list` then `saber threads resume <id>` |
| Use deeper reasoning | `saber --thinking "analyze retention by cohort"` |

## Knowledge base

Save reusable business context so SQLsaber can answer consistently:

```bash
saber knowledge add \
  "Revenue KPI" \
  "Recognized revenue from shipped orders only" \
  --sql "SELECT SUM(amount) FROM orders WHERE status = 'shipped'" \
  --source "finance-wiki"

saber knowledge search "revenue shipped orders"
```

Knowledge entries are scoped per database and are discovered automatically when relevant.

## Optional plugins

Install official plugins alongside SQLsaber:

```bash
# Render charts in your terminal
uv tool install --with sqlsaber-viz sqlsaber

# Run Python in a secure sandbox
uv tool install --with sqlsaber-sandbox sqlsaber

# Install both
uv tool install --with sqlsaber-viz,sqlsaber-sandbox sqlsaber
```

## Python SDK

Use the same SQLsaber agent from Python scripts, notebooks, web apps, or pipelines:

```python
import asyncio

from sqlsaber import SQLSaber, SQLSaberOptions


async def main() -> None:
    async with SQLSaber(options=SQLSaberOptions(database="sqlite:///my.db")) as saber:
        result = await saber.query("Top 5 customers by revenue")
        print(result)
        print(result.usage)


asyncio.run(main())
```

## How it works

1. **Discovery** — Lists tables and identifies relevant ones based on your question.
2. **Schema analysis** — Introspects only the tables needed.
3. **Knowledge retrieval** — Searches saved KPI definitions and SQL patterns when useful.
4. **Query generation** — Writes SQL tailored to your database dialect.
5. **Execution** — Runs the query with safety checks.
6. **Results** — Formats the output with an explanation.

## Documentation

Full docs at [sqlsaber.com](https://sqlsaber.com):

- [Installation](https://sqlsaber.com/installation/)
- [Getting Started](https://sqlsaber.com/guides/getting-started/)
- [Database Setup](https://sqlsaber.com/guides/database-setup/)
- [Running Queries](https://sqlsaber.com/guides/queries/)
- [Multiple Databases](https://sqlsaber.com/guides/multi-database/)
- [Knowledge Base](https://sqlsaber.com/guides/knowledge/)
- [Python SDK](https://sqlsaber.com/sdk/overview/)
- [Command Reference](https://sqlsaber.com/reference/commands/)

## Contributing

Contributions welcome! Please open an issue first to discuss changes.

If you find SQLsaber useful, a ⭐ on GitHub helps others discover it.

## License

Apache-2.0 — see [LICENSE](./LICENSE)
