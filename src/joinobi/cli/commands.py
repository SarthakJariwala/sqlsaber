"""CLI command definitions and handlers."""

import asyncio
import os
from typing import Optional

import typer
from rich.console import Console

from joinobi.agents.anthropic import AnthropicSQLAgent
from joinobi.cli.interactive import InteractiveSession
from joinobi.cli.streaming import StreamingQueryHandler
from joinobi.database.connection import DatabaseConnection

app = typer.Typer(
    name="joinobi",
    help="JoinObi CLI - SQL like Claude Code",
    add_completion=True,
)

console = Console()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    query: Optional[str] = typer.Argument(None, help="SQL query in natural language"),
    database_url: Optional[str] = typer.Option(
        None,
        "--db",
        "-d",
        help="Database URL (defaults to DATABASE_URL env var)",
        envvar="DATABASE_URL",
    ),
    allow_write: bool = typer.Option(
        False, "--write", "-w", help="Allow write operations (INSERT, UPDATE, DELETE)"
    ),
):
    """
    Query your database using natural language.

    Examples:
        jb "show me all users"  # Run a single query
        jb                       # Start interactive mode
    """
    # If a subcommand was invoked, don't run the main logic
    if ctx.invoked_subcommand is not None:
        return

    async def main_async():
        # Check if database URL is provided
        if not database_url:
            if not os.getenv("DATABASE_URL"):
                console.print("[bold red]Error:[/bold red] No database URL provided.")
                console.print(
                    "Please set DATABASE_URL environment variable or use --db option."
                )
                raise typer.Exit(1)

        # Create database connection
        db_conn = DatabaseConnection(database_url)

        # Create agent instance
        agent = AnthropicSQLAgent(db_conn, allow_write)

        try:
            if query is None:
                # Interactive mode
                session = InteractiveSession(console, agent, allow_write)
                await session.run()
            else:
                # Single query mode with streaming
                streaming_handler = StreamingQueryHandler(console)
                await streaming_handler.execute_streaming_query(query, agent)

        finally:
            # Clean up
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    # Run the async function
    asyncio.run(main_async())


def main():
    """Entry point for the CLI application."""
    app()
