"""CLI command definitions and handlers."""

import asyncio
import sys
from typing import Annotated

import cyclopts
from rich.console import Console

from sqlsaber.cli.auth import create_auth_app
from sqlsaber.cli.database import create_db_app
from sqlsaber.cli.memory import create_memory_app
from sqlsaber.cli.models import create_models_app
from sqlsaber.cli.threads import create_threads_app

# Lazy imports - only import what's needed for CLI parsing
from sqlsaber.config.database import DatabaseConfigManager


class CLIError(Exception):
    """Exception raised for CLI errors that should result in exit."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


app = cyclopts.App(
    name="sqlsaber",
    help="SQLsaber - Open-source agentic SQL assistant for your database",
)

app.command(create_auth_app(), name="auth")
app.command(create_db_app(), name="db")
app.command(create_memory_app(), name="memory")
app.command(create_models_app(), name="models")
app.command(create_threads_app(), name="threads")

console = Console()
config_manager = DatabaseConfigManager()


@app.meta.default
def meta_handler(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name, file path (CSV/SQLite/DuckDB), or connection string (postgresql://, mysql://, duckdb://) (uses default if not specified)",
        ),
    ] = None,
):
    """
    Query your database using natural language.

    Examples:
        saber                                  # Start interactive mode
        saber "show me all users"              # Run a single query with default database
        saber -d mydb "show me users"          # Run a query with specific database
        saber -d data.csv "show me users"      # Run a query with ad-hoc CSV file
        saber -d data.db "show me users"       # Run a query with ad-hoc SQLite file
        saber -d data.duckdb "show me users"   # Run a query with ad-hoc DuckDB file
        saber -d "postgresql://user:pass@host:5432/db" "show users"  # PostgreSQL connection string
        saber -d "mysql://user:pass@host:3306/db" "show users"       # MySQL connection string
        saber -d "duckdb:///data.duckdb" "show users"                 # DuckDB connection string
        echo "show me all users" | saber       # Read query from stdin
        cat query.txt | saber                  # Read query from file via stdin
    """
    # Store database in app context for commands to access
    app.meta["database"] = database


@app.default
def query(
    query_text: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Question in natural language (if not provided, reads from stdin or starts interactive mode)",
        ),
    ] = None,
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name, file path (CSV/SQLite/DuckDB), or connection string (postgresql://, mysql://, duckdb://) (uses default if not specified)",
        ),
    ] = None,
    thinking: bool = False,
):
    """Run a query against the database or start interactive mode.

    When called without arguments:
    - If stdin has data, reads query from stdin
    - Otherwise, starts interactive mode

    When called with a query string, executes that query and exits.

    Examples:
        saber                             # Start interactive mode
        saber "show me all users"         # Run a single query
        saber -d data.csv "show users"    # Run a query with ad-hoc CSV file
        saber -d data.db "show users"     # Run a query with ad-hoc SQLite file
        saber -d data.duckdb "show users" # Run a query with ad-hoc DuckDB file
        saber -d "postgresql://user:pass@host:5432/db" "show users"  # PostgreSQL connection string
        saber -d "mysql://user:pass@host:3306/db" "show users"       # MySQL connection string
        saber -d "duckdb:///data.duckdb" "show users"                 # DuckDB connection string
        echo "show me all users" | saber  # Read query from stdin
    """

    async def run_session():
        # Import heavy dependencies only when actually running a query
        # This is only done to speed up startup time
        from sqlsaber.agents import SQLSaberAgent
        from sqlsaber.cli.interactive import InteractiveSession
        from sqlsaber.cli.streaming import StreamingQueryHandler
        from sqlsaber.database import (
            DatabaseConnection,
        )
        from sqlsaber.database.resolver import DatabaseResolutionError, resolve_database
        from sqlsaber.threads import ThreadStorage

        # Check if query_text is None and stdin has data
        actual_query = query_text
        if query_text is None and not sys.stdin.isatty():
            # Read from stdin
            actual_query = sys.stdin.read().strip()
            if not actual_query:
                # If stdin was empty, fall back to interactive mode
                actual_query = None

        # Resolve database from CLI input
        try:
            resolved = resolve_database(database, config_manager)
            connection_string = resolved.connection_string
            db_name = resolved.name
        except DatabaseResolutionError as e:
            raise CLIError(str(e))

        # Create database connection
        try:
            db_conn = DatabaseConnection(connection_string)
        except Exception as e:
            raise CLIError(f"Error creating database connection: {e}")

        # Create pydantic-ai agent instance with database name for memory context
        sqlsaber_agent = SQLSaberAgent(db_conn, db_name, thinking_enabled=thinking)

        try:
            if actual_query:
                # Single query mode with streaming
                streaming_handler = StreamingQueryHandler(console)
                db_type = sqlsaber_agent.db_type
                console.print(
                    f"[bold blue]Connected to:[/bold blue] {db_name} ({db_type})\n"
                )
                run = await streaming_handler.execute_streaming_query(
                    actual_query, sqlsaber_agent
                )
                # Persist non-interactive run as a thread snapshot so it can be resumed later
                try:
                    if run is not None:
                        threads = ThreadStorage()

                        thread_id = await threads.save_snapshot(
                            messages_json=run.all_messages_json(),
                            database_name=db_name,
                        )
                        await threads.save_metadata(
                            thread_id=thread_id,
                            title=actual_query,
                            model_name=sqlsaber_agent.agent.model.model_name,
                        )
                        await threads.end_thread(thread_id)
                        console.print(
                            f"[dim]You can continue this thread using:[/dim] saber threads resume {thread_id}"
                        )
                except Exception:
                    # best-effort persistence; don't fail the CLI on storage errors
                    pass
                finally:
                    await threads.prune_threads()
            else:
                # Interactive mode
                session = InteractiveSession(console, sqlsaber_agent, db_conn, db_name)
                await session.run()

        finally:
            # Clean up
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    # Run the async function with proper error handling
    try:
        asyncio.run(run_session())
    except CLIError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(e.exit_code)


def main():
    """Entry point for the CLI application."""
    app()
