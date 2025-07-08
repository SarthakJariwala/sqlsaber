# SQLSaber

> Use the agent Luke!

SQLSaber is an agentic SQL assistant. Think Claude Code but for SQL.

Ask your questions in natural language and it will gather the right context automatically and answer your query by writing SQL and analyzing the results.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Database Connection](#database-connection)
  - [AI Model Configuration](#ai-model-configuration)
  - [Memory Management](#memory-management)
- [Usage](#usage)
  - [Interactive Mode](#interactive-mode)
  - [Single Query](#single-query)
  - [Database Selection](#database-selection)
- [Examples](#examples)
- [MCP Server Integration](#mcp-server-integration)
  - [Starting the MCP Server](#starting-the-mcp-server)
  - [Configuring MCP Clients](#configuring-mcp-clients)
  - [Available MCP Tools](#available-mcp-tools)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Natural language to SQL conversion
- 🔍 Automatic database schema introspection
- 🛡️ Safe query execution (read-only by default)
- 🧠 Memory management
- 💬 Interactive REPL mode
- 🎨 Beautiful formatted output with syntax highlighting
- 🗄️ Support for PostgreSQL, SQLite, and MySQL
- 🔌 MCP (Model Context Protocol) server support

## Installation

### `uv`

```bash
uv tool install sqlsaber
```

### `pipx`

```bash
pipx install sqlsaber
```

### `brew`

```bash
brew install uv
uv tool install sqlsaber
```

## Configuration

### Database Connection

Set your database connection URL:

```bash
saber db add DB_NAME
```

This will ask you some questions about your database connection

### AI Model Configuration

SQLSaber uses Sonnet-4 by default. You can change it using:

```bash
saber models set

# for more model settings run:
saber models --help
```

### Memory Management

You can add specific context about your database to the model using the memory feature. This is similar to how you add memory/context in Claude Code.

```bash
saber memory add 'always convert dates to string for easier formating'
```

View all memories

```bash
saber memory list
```

> You can also add memories in an interactive query session by starting with the `#` sign

## Usage

### Interactive Mode

Start an interactive session:

```bash
saber
```

> You can also add memories in an interactive session by starting your message with the `#` sign

### Single Query

Execute a single natural language query:

```bash
saber "show me all users created this month"
```

You can also pipe queries from stdin:

```bash
echo "show me all users created this month" | saber
cat query.txt | saber
```

### Database Selection

Use a specific database connection:

```bash
# Interactive mode with specific database
saber -d mydb

# Single query with specific database
saber -d mydb "count all orders"
```

## Examples

```bash
# Show database schema
saber "what tables are in my database?"

# Count records
saber "how many active users do we have?"

# Complex queries with joins
saber "show me orders with customer details for this week"

# Aggregations
saber "what's the total revenue by product category?"

# Date filtering
saber "list users who haven't logged in for 30 days"

# Data exploration
saber "show me the distribution of customer ages"

# Business analytics
saber "which products had the highest sales growth last quarter?"

# Start interactive mode
saber
```

## MCP Server Integration

SQLSaber includes an MCP (Model Context Protocol) server that allows AI agents like Claude Code to directly leverage tools available in SQLSaber.

### Starting the MCP Server

Run the MCP server using uvx:

```bash
uvx --from sqlsaber saber-mcp
```

### Configuring MCP Clients

#### Claude Code

Add SQLSaber as an MCP server in Claude Code:

```bash
claude mcp add -- uvx --from sqlsaber saber-mcp
```

#### Other MCP Clients

For other MCP clients, configure them to run the command: `uvx --from sqlsaber saber-mcp`

### Available MCP Tools

Once connected, the MCP client will have access to these tools:

- `get_databases()` - Lists all configured databases
- `list_tables(database)` - Get all tables in a database with row counts
- `introspect_schema(database, table_pattern?)` - Get detailed schema information
- `execute_sql(database, query, limit?)` - Execute SQL queries (read-only)

The MCP server uses your existing SQLSaber database configurations, so make sure to set up your databases using `saber db add` first.

## How It Works

SQLSaber uses a multi-step process to gather the right context, provide it to the model, and execute SQL queries to get the right answers:

![](./sqlsaber.svg)

### 🔍 Discovery Phase

1. **List Tables Tool**: Quickly discovers available tables with row counts
2. **Pattern Matching**: Identifies relevant tables based on your query

### 📋 Schema Analysis

3. **Smart Schema Introspection**: Analyzes only the specific table structures needed for your query

### ⚡ Execution Phase

4. **SQL Generation**: Creates optimized SQL queries based on natural language input
5. **Safe Execution**: Runs read-only queries with built-in protections against destructive operations
6. **Result Formatting**: Presents results with explanations in tables and optionally, visualizes using plots

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss your ideas or report bugs.

## License

This project is licensed under Apache-2.0 License - see the LICENSE file for details.
