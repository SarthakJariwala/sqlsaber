# SQLsaber

> SQLsaber is an open-source agentic SQL assistant. Think Claude Code but for SQL.

![demo](./sqlsaber.gif)

Stop fighting your database.

Ask your questions in natural language and `sqlsaber` will gather the right context automatically and answer your query by writing SQL and analyzing the results.

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
  - [Resume Past Conversation](#resume-past-conversation)
  - [Database Selection](#database-selection)
- [Examples](#examples)

- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Features

- Automatic database schema introspection
- Safe query execution (read-only by default)
- Memory management
- Interactive REPL mode
- Conversation threads (store, display, and resume conversations)
- Support for PostgreSQL, MySQL, SQLite, DuckDB, and CSVs

- Extended thinking mode for select models (Anthropic, OpenAI, Google, Groq)
- Beautiful formatted output

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

### Extended Thinking Mode

For complex queries that require deeper reasoning, `sqlsaber` supports extended thinking mode. When enabled, you will see the model's reasoning process as it generates SQL queries and arrives at conclusions.

**Enable/disable via CLI flags:**

```bash
# Enable thinking for a single query
saber --thinking "analyze sales trends across regions"

# Disable thinking for a single query
saber --no-thinking "show me all users"
```

**Toggle in interactive mode:**

```bash
# In interactive mode, use slash commands
/thinking on   # Enable thinking
/thinking off  # Disable thinking
```

**Configure default setting:**

Thinking is disabled by default. To change the default, edit your config file at `~/.config/sqlsaber/model_config.json`:

```json
{
  "model": "anthropic:claude-sonnet-4-20250514",
  "thinking_enabled": true
}
```

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

### Resume Past Conversation

Continue a previous conversation thread:

```bash
saber threads resume THREAD_ID
```

### Database Selection

Use a specific database connection:

```bash
# Interactive mode with specific database
saber -d mydb

# Single query with specific database
saber -d mydb "count all orders"

# You can also pass a connection string
saber -d "postgresql://user:password@localhost:5432/mydb" "count all orders"
saber -d "duckdb:///path/to/data.duckdb" "top customers"
```

## Examples

```bash
# Start interactive mode
saber

# Non-interactive mode
saber "show me orders with customer details for this week"

saber "which products had the highest sales growth last quarter?"
```

## How It Works

SQLsaber uses a multi-step agentic process to gather the right context and execute SQL queries to answer your questions:

![](./sqlsaber.svg)

### 🔍 Discovery Phase

1. **List Tables Tool**: Quickly discovers available tables with row counts
2. **Pattern Matching**: Identifies relevant tables based on your query

### 📋 Schema Analysis

3. **Smart Schema Introspection**: Analyzes only the specific table structures needed for your query

### ⚡ Execution Phase

4. **SQL Generation**: Creates optimized SQL queries based on natural language input
5. **Safe Execution**: Runs read-only queries with built-in protections against destructive operations
6. **Result Formatting**: Presents results with explanations in tables

## Contributing

If you like the project, starring the repo is a great way to show your support!

Other contributions are welcome! Please feel free to open an issue to discuss your ideas or report bugs.

## License

This project is licensed under Apache-2.0 License - see the LICENSE file for details.
