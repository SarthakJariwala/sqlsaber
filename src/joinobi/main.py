import asyncio
import os
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .agent_anthropic import AnthropicSQLAgent, query_database
from .database import DatabaseConnection

app = typer.Typer(
    name="joinobi",
    help="JoinObi CLI - SQL like Claude Code",
    add_completion=True,
)

console = Console()


@app.callback()
def callback():
    """
    JoinObi CLI tool

    A powerful command-line interface for JoinObi operations.
    """
    pass


@app.command()
def hello(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Name to greet"),
):
    """
    Say hello to someone.
    """
    if name:
        typer.echo(f"Hello, {name}! Welcome to JoinObi CLI.")
    else:
        typer.echo("Hello! Welcome to JoinObi CLI.")


@app.command()
def version():
    """
    Show the CLI version.
    """
    typer.echo("JoinObi CLI v0.1.0")


@app.command()
def sql(
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
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Start interactive SQL session"
    ),
):
    """
    Query your database using natural language.

    Examples:
        jb sql "show me all users"
        jb sql "count orders by status"
        jb sql -i  # Interactive mode
    """

    async def run_streaming_query(user_query: str, agent: AnthropicSQLAgent):
        """Execute a query with streaming display."""
        console.print(f"\n[bold blue]Query:[/bold blue] {user_query}")

        displayed_results = False
        has_content = False
        explanation_started = False
        status = console.status("[yellow]🧠 Crunching data...[/yellow]")
        status.start()

        async for event in agent.query_stream(user_query):
            if event.type == "tool_use":
                # Stop any ongoing status, but don't mark has_content yet
                try:
                    status.stop()
                except Exception:
                    pass

                if event.data["status"] == "started":
                    # If explanation was streaming, add newline before tool use
                    if explanation_started:
                        console.print()
                    console.print(
                        f"\n[yellow]🔧 Using tool: {event.data['name']}[/yellow]"
                    )
                elif event.data["status"] == "executing":
                    if event.data["name"] == "list_tables":
                        console.print("[dim]  → Discovering available tables[/dim]")
                    elif event.data["name"] == "introspect_schema":
                        pattern = event.data["input"].get("table_pattern", "all tables")
                        console.print(f"[dim]  → Examining schema for: {pattern}[/dim]")
                    elif event.data["name"] == "execute_sql":
                        query = event.data["input"].get("query", "")
                        console.print("\n[bold green]Executing SQL:[/bold green]")
                        syntax = Syntax(
                            query, "sql", theme="monokai", line_numbers=True
                        )
                        console.print(syntax)

            elif event.type == "text":
                # Always stop status when text streaming starts
                try:
                    status.stop()
                except Exception:
                    pass

                if not explanation_started:
                    console.print("\n[bold cyan]Explanation:[/bold cyan] ", end="")
                    explanation_started = True
                    has_content = True

                # Print text as it streams
                console.print(event.data, end="", markup=False)

            elif event.type == "query_result":
                if not displayed_results and event.data["results"]:
                    results = event.data["results"]
                    console.print(
                        f"\n[bold magenta]Results ({len(results)} rows):[/bold magenta]"
                    )

                    # Create a rich table
                    table = Table(show_header=True, header_style="bold blue")

                    # Add columns
                    for key in results[0].keys():
                        table.add_column(key)

                    # Add rows
                    for row in results[:20]:  # Show first 20 rows
                        table.add_row(*[str(row[key]) for key in results[0].keys()])

                    console.print(table)

                    if len(results) > 20:
                        console.print(
                            f"[yellow]... and {len(results) - 20} more rows[/yellow]"
                        )

                    displayed_results = True

            elif event.type == "processing":
                # Show status when processing tool results
                if explanation_started:
                    console.print()  # Add newline after explanation text
                try:
                    status.stop()
                except Exception:
                    pass  # Status might already be stopped
                status = console.status(f"[yellow]🧠 {event.data}[/yellow]")
                status.start()
                has_content = True

            elif event.type == "error":
                if not has_content:
                    status.stop()
                    has_content = True
                console.print(f"\n[bold red]Error:[/bold red] {event.data}")

        # Make sure status is stopped
        try:
            status.stop()
        except Exception:
            pass  # Status might already be stopped

        # Add a newline after streaming completes if explanation was shown
        if explanation_started:
            console.print()  # Empty line for better readability

    async def run_query(user_query: str, db_conn: DatabaseConnection):
        """Execute a single query and display results (legacy non-streaming)."""
        console.print(f"\n[bold blue]Query:[/bold blue] {user_query}")

        with console.status("[yellow]Processing query...[/yellow]"):
            response = await query_database(db_conn, user_query, allow_write)

        # Display SQL query if generated
        if response.query:
            syntax = Syntax(response.query, "sql", theme="monokai", line_numbers=True)
            console.print("\n[bold green]Generated SQL:[/bold green]")
            console.print(syntax)

        # Display explanation
        console.print(f"\n[bold cyan]Explanation:[/bold cyan] {response.explanation}")

        # Display results if any
        if response.results:
            console.print(
                f"\n[bold magenta]Results ({len(response.results)} rows):[/bold magenta]"
            )

            if response.results:
                # Create a rich table
                table = Table(show_header=True, header_style="bold blue")

                # Add columns
                for key in response.results[0].keys():
                    table.add_column(key)

                # Add rows
                for row in response.results[:20]:  # Show first 20 rows
                    table.add_row(
                        *[str(row[key]) for key in response.results[0].keys()]
                    )

                console.print(table)

                if len(response.results) > 20:
                    console.print(
                        f"[yellow]... and {len(response.results) - 20} more rows[/yellow]"
                    )

        # Display error if any
        if response.error:
            console.print(f"\n[bold red]Error:[/bold red] {response.error}")

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
            if interactive or query is None:
                # Interactive mode with conversation history
                console.print(
                    Panel.fit(
                        "[bold green]JoinObi SQL Assistant[/bold green]\n"
                        "Type your queries in natural language. Type 'exit' or 'quit' to leave.\n"
                        "[dim]Now with conversation memory and real-time visibility![/dim]",
                        border_style="green",
                    )
                )

                if allow_write:
                    console.print(
                        "[yellow]Warning: Write operations are enabled![/yellow]\n"
                    )

                console.print(
                    "[dim]Commands: 'clear' to reset conversation, 'exit'/'quit' to leave[/dim]\n"
                )

                while True:
                    try:
                        user_query = console.input("[bold blue]sql> [/bold blue]")

                        if user_query.lower() in ["exit", "quit", "q"]:
                            break

                        if user_query.lower() == "clear":
                            agent.clear_history()
                            console.print(
                                "[green]Conversation history cleared.[/green]\n"
                            )
                            continue

                        if user_query.strip():
                            await run_streaming_query(user_query, agent)
                            console.print()  # Empty line for readability

                    except KeyboardInterrupt:
                        console.print(
                            "\n[yellow]Use 'exit' or 'quit' to leave.[/yellow]"
                        )
                    except Exception as e:
                        console.print(f"[bold red]Error:[/bold red] {str(e)}")

            else:
                # Single query mode (non-streaming for backward compatibility)
                await run_query(query, db_conn)

        finally:
            # Clean up
            await db_conn.close()
            console.print("\n[green]Goodbye![/green]")

    # Run the async function
    asyncio.run(main_async())


def main():
    app()


if __name__ == "__main__":
    main()
