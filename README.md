# JoinObi - SQL Assistant CLI

JoinObi is a natural language SQL assistant powered by pydantic-ai. It helps you query your PostgreSQL database using plain English instead of SQL syntax.

## Features

- > Natural language to SQL conversion
- = Automatic database schema introspection
- =� Safe query execution (read-only by default)
- =� Interactive REPL mode
- =� Beautiful formatted output with syntax highlighting

## Installation

```bash
pip install -e .
```

## Configuration

### Database Connection

Set your database connection URL:

```bash
export DATABASE_URL="postgresql://username:password@localhost:5432/dbname"
```

### AI Model Configuration

JoinObi uses OpenAI's GPT-4 by default. Set your API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

You can also use other models:

```bash
# Use Claude (Anthropic)
export JOINOBI_MODEL="anthropic:claude-3-opus-20240229"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Use other OpenAI models
export JOINOBI_MODEL="openai:gpt-3.5-turbo"
```

## Usage

### Interactive Mode

Start an interactive SQL session:

```bash
jb sql -i
```

### Single Query

Execute a single natural language query:

```bash
jb sql "show me all users created this month"
```

### With Write Permissions

Enable write operations (INSERT, UPDATE, DELETE):

```bash
jb sql --write "update the user with id 123 to set status as active"
```

### Specify Database URL

Override the DATABASE_URL environment variable:

```bash
jb sql --db postgresql://localhost/mydb "count all orders"
```

## Examples

```bash
# Show database schema
jb sql "what tables are in my database?"

# Count records
jb sql "how many active users do we have?"

# Complex queries with joins
jb sql "show me orders with customer details for this week"

# Aggregations
jb sql "what's the total revenue by product category?"

# Date filtering
jb sql "list users who haven't logged in for 30 days"
```

## How It Works

JoinObi uses three main tools optimized for minimal token usage:

1. **List Tables Tool**: Quickly discovers available tables with row counts (minimal data transfer).

2. **Schema Introspection Tool**: Analyzes specific table structures using pattern matching to fetch only relevant schema information.

3. **SQL Execution Tool**: Safely executes the generated SQL queries with built-in protections against destructive operations.

The AI agent:
- Lists available tables first (minimal tokens)
- Identifies relevant tables based on your query
- Introspects only the necessary schema (80-95% token reduction)
- Generates appropriate SQL
- Executes the query safely
- Formats and explains the results

### Performance Optimizations

JoinObi includes several optimizations to handle large databases efficiently:

- **Lazy Schema Loading**: Only fetches schema for tables that match your query pattern
- **Schema Caching**: Caches schema information for 15 minutes to reduce repeated database queries
- **Incremental Discovery**: First lists tables, then fetches details only for relevant ones
- **Pattern Matching**: Supports SQL LIKE patterns (e.g., 'user%', '%order%') for efficient filtering

These optimizations reduce token usage by 80-95% compared to fetching the entire schema, helping you stay within API rate limits even with large databases.

## Programmatic Usage

You can also use JoinObi programmatically:

```python
import asyncio
from joinobi.agent import query_database
from joinobi.database import DatabaseConnection

async def main():
    db = DatabaseConnection("postgresql://localhost/mydb")
    
    response = await query_database(
        db, 
        "show me top 10 customers by total order value"
    )
    
    print(f"SQL: {response.query}")
    print(f"Explanation: {response.explanation}")
    print(f"Results: {response.results}")
    
    await db.close()

asyncio.run(main())
```

## Security

- Read-only by default - write operations require explicit `--write` flag
- SQL injection protection through proper query construction
- Connection pooling for efficient resource usage

## Requirements

- Python 3.12+
- PostgreSQL database
- OpenAI API key (for pydantic-ai)