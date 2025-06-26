"""CLI command definitions and handlers."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from sqlsaber.agents.anthropic import AnthropicSQLAgent
from sqlsaber.cli.database import create_db_app
from sqlsaber.cli.interactive import InteractiveSession
from sqlsaber.cli.memory import create_memory_app
from sqlsaber.cli.models import create_models_app
from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.database.connection import DatabaseConnection

app = typer.Typer(
    name="sqlsaber",
    help="SQLSaber - Use the agent Luke!\n\nSQL assistant for your database",
    add_completion=True,
)


console = Console()
config_manager = DatabaseConfigManager()


@app.callback()
def main_callback(
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """
    Query your database using natural language.

    Examples:
        sb query                             # Start interactive mode
        sb query "show me all users"         # Run a single query with default database
        sb query -d mydb "show me users"     # Run a query with specific database
    """
    pass


@app.command()
def query(
    query_text: Optional[str] = typer.Argument(
        None,
        help="SQL query in natural language (if not provided, starts interactive mode)",
    ),
    database: Optional[str] = typer.Option(
        None,
        "--database",
        "-d",
        help="Database connection name (uses default if not specified)",
    ),
):
    """Run a query against the database or start interactive mode."""

    async def run_session():
        # Get database configuration or handle direct CSV file
        if database:
            # Check if this is a direct CSV file path
            if database.endswith(".csv"):
                csv_path = Path(database).expanduser().resolve()
                if not csv_path.exists():
                    console.print(
                        f"[bold red]Error:[/bold red] CSV file '{database}' not found."
                    )
                    raise typer.Exit(1)
                connection_string = f"csv:///{csv_path}"
                db_name = csv_path.stem
            else:
                # Look up configured database connection
                db_config = config_manager.get_database(database)
                if not db_config:
                    console.print(
                        f"[bold red]Error:[/bold red] Database connection '{database}' not found."
                    )
                    console.print(
                        "Use 'sqlsaber db list' to see available connections."
                    )
                    raise typer.Exit(1)
                connection_string = db_config.to_connection_string()
                db_name = db_config.name
        else:
            db_config = config_manager.get_default_database()
            if not db_config:
                console.print(
                    "[bold red]Error:[/bold red] No database connections configured."
                )
                console.print(
                    "Use 'sqlsaber db add <name>' to add a database connection."
                )
                raise typer.Exit(1)
            connection_string = db_config.to_connection_string()
            db_name = db_config.name

        # Create database connection
        try:
            db_conn = DatabaseConnection(connection_string)
        except Exception as e:
            console.print(
                f"[bold red]Error creating database connection:[/bold red] {e}"
            )
            raise typer.Exit(1)

        # Create agent instance with database name for memory context
        agent = AnthropicSQLAgent(db_conn, db_name)

        try:
            if query_text:
                # Single query mode with streaming
                streaming_handler = StreamingQueryHandler(console)
                console.print(
                    f"[bold blue]Connected to:[/bold blue] {db_name} {agent._get_database_type_name()}\n"
                )
                await streaming_handler.execute_streaming_query(query_text, agent)
            else:
                # Interactive mode
                session = InteractiveSession(console, agent)
                await session.run()

        finally:
            # Clean up
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    # Run the async function
    asyncio.run(run_session())


# Add database management commands after main callback is defined
db_app = create_db_app()
app.add_typer(db_app, name="db")

# Add memory management commands
memory_app = create_memory_app()
app.add_typer(memory_app, name="memory")

# Add model management commands
models_app = create_models_app()
app.add_typer(models_app, name="models")


def main():
    """Entry point for the CLI application."""
    app()
