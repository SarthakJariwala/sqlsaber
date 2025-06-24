# SQLSaber

> Use the agent Luke!

SQLSaber is an agentic SQL assistant. Think Claude Code but for SQL.

Ask your questions in natural language and it will gather the right context and answer your query by writing SQL and analyzing the results.

## Features

- Natural language to SQL conversion
- 🔍 Automatic database schema introspection
- 🛡️ Safe query execution (read-only by default)
- 🧠 Memory management
- 💬 Interactive REPL mode
- 🎨 Beautiful formatted output with syntax highlighting
- 🗄️ Support for PostgreSQL, SQLite, and MySQL

## Installation

```bash
uv tool install sqlsaber
```

or

```bash
pipx install sqlsaber
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
saber query
```

> You can also add memories in an interactive session by starting your message with the `#` sign

### Single Query

Execute a single natural language query:

```bash
saber query "show me all users created this month"
```

### Database Selection

Use a specific database connection:

```bash
# Use named database from config
saber query -d mydb "count all orders"
```

## Examples

```bash
# Show database schema
saber query "what tables are in my database?"

# Count records
saber query "how many active users do we have?"

# Complex queries with joins
saber query "show me orders with customer details for this week"

# Aggregations
saber query "what's the total revenue by product category?"

# Date filtering
saber query "list users who haven't logged in for 30 days"

# Data exploration
saber query "show me the distribution of customer ages"

# Business analytics
saber query "which products had the highest sales growth last quarter?"
```

## How It Works

SQLSaber uses an intelligent three-step process optimized for minimal token usage:

### 🔍 Discovery Phase

1. **List Tables Tool**: Quickly discovers available tables with row counts
2. **Pattern Matching**: Identifies relevant tables based on your query using SQL LIKE patterns

### 📋 Schema Analysis

3. **Smart Introspection**: Analyzes only the specific table structures needed for your query
4. **Selective Loading**: Fetches schema information only for relevant tables

### ⚡ Execution Phase

5. **SQL Generation**: Creates optimized SQL queries based on natural language input
6. **Safe Execution**: Runs queries with built-in protections against destructive operations
7. **Result Formatting**: Presents results with syntax highlighting and explanations

## Contributing

Contributions are welcome! Please feel free to open an issue to discuss your ideas or report bugs.

## License

This project is licensed under Apache-2.0 License - see the LICENSE file for details.
