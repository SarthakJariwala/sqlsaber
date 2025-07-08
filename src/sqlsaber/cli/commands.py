"""CLI command definitions and handlers."""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import cyclopts
from rich.console import Console

from sqlsaber.agents.anthropic import AnthropicSQLAgent
from sqlsaber.cli.auth import create_auth_app
from sqlsaber.cli.database import create_db_app
from sqlsaber.cli.interactive import InteractiveSession
from sqlsaber.cli.memory import create_memory_app
from sqlsaber.cli.models import create_models_app
from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.database.connection import DatabaseConnection


class CLIError(Exception):
    """Exception raised for CLI errors that should result in exit."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


app = cyclopts.App(
    name="sqlsaber",
    help="SQLSaber - Use the agent Luke!\n\nSQL assistant for your database",
)


console = Console()
config_manager = DatabaseConfigManager()


@app.meta.default
def meta_handler(
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """
    Query your database using natural language.

    Examples:
        saber                                  # Start interactive mode
        saber "show me all users"              # Run a single query with default database
        saber -d mydb "show me users"          # Run a query with specific database
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
            help="SQL query in natural language (if not provided, reads from stdin or starts interactive mode)",
        ),
    ] = None,
    database: Annotated[
        str | None,
        cyclopts.Parameter(
            ["--database", "-d"],
            help="Database connection name (uses default if not specified)",
        ),
    ] = None,
):
    """Run a query against the database or start interactive mode.

    When called without arguments:
    - If stdin has data, reads query from stdin
    - Otherwise, starts interactive mode

    When called with a query string, executes that query and exits.

    Examples:
        saber                             # Start interactive mode
        saber "show me all users"         # Run a single query
        echo "show me all users" | saber  # Read query from stdin
    """

    async def run_session():
        # Check if query_text is None and stdin has data
        actual_query = query_text
        if query_text is None and not sys.stdin.isatty():
            # Read from stdin
            actual_query = sys.stdin.read().strip()
            if not actual_query:
                # If stdin was empty, fall back to interactive mode
                actual_query = None

        # Get database configuration or handle direct CSV file
        if database:
            # Check if this is a direct CSV file path
            if database.endswith(".csv"):
                csv_path = Path(database).expanduser().resolve()
                if not csv_path.exists():
                    raise CLIError(f"CSV file '{database}' not found.")
                connection_string = f"csv:///{csv_path}"
                db_name = csv_path.stem
            else:
                # Look up configured database connection
                db_config = config_manager.get_database(database)
                if not db_config:
                    raise CLIError(
                        f"Database connection '{database}' not found. Use 'sqlsaber db list' to see available connections."
                    )
                connection_string = db_config.to_connection_string()
                db_name = db_config.name
        else:
            db_config = config_manager.get_default_database()
            if not db_config:
                raise CLIError(
                    "No database connections configured. Use 'sqlsaber db add <name>' to add a database connection."
                )
            connection_string = db_config.to_connection_string()
            db_name = db_config.name

        # Create database connection
        try:
            db_conn = DatabaseConnection(connection_string)
        except Exception as e:
            raise CLIError(f"Error creating database connection: {e}")

        # Create agent instance with database name for memory context
        agent = AnthropicSQLAgent(db_conn, db_name)

        try:
            if actual_query:
                # Single query mode with streaming
                streaming_handler = StreamingQueryHandler(console)
                console.print(
                    f"[bold blue]Connected to:[/bold blue] {db_name} {agent._get_database_type_name()}\n"
                )
                await streaming_handler.execute_streaming_query(actual_query, agent)
            else:
                # Interactive mode
                session = InteractiveSession(console, agent)
                await session.run()

        finally:
            # Clean up
            await agent.close()  # Close the agent's HTTP client
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    # Run the async function with proper error handling
    try:
        asyncio.run(run_session())
    except CLIError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(e.exit_code)


# Add authentication management commands
auth_app = create_auth_app()
app.command(auth_app, name="auth")

# Add database management commands after main callback is defined
db_app = create_db_app()
app.command(db_app, name="db")

# Add memory management commands
memory_app = create_memory_app()
app.command(memory_app, name="memory")

# Add model management commands
models_app = create_models_app()
app.command(models_app, name="models")


def main():
    """Entry point for the CLI application."""
    app()
