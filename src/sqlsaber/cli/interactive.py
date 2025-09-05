"""Interactive mode handling for the CLI."""

import asyncio

import questionary
from pydantic_ai import Agent
from rich.console import Console
from rich.panel import Panel

from sqlsaber.cli.completers import (
    CompositeCompleter,
    SlashCommandCompleter,
    TableNameCompleter,
)
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.database.schema import SchemaManager


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(self, console: Console, agent: Agent, db_conn, database_name: str):
        self.console = console
        self.agent = agent
        self.db_conn = db_conn
        self.database_name = database_name
        self.display = DisplayManager(console)
        self.streaming_handler = StreamingQueryHandler(console)
        self.current_task: asyncio.Task | None = None
        self.cancellation_token: asyncio.Event | None = None
        self.table_completer = TableNameCompleter()
        self.message_history: list | None = []

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        # Show database information
        db_name = self.database_name or "Unknown"
        from sqlsaber.database.connection import (
            CSVConnection,
            MySQLConnection,
            PostgreSQLConnection,
            SQLiteConnection,
        )

        db_type = (
            "PostgreSQL"
            if isinstance(self.db_conn, PostgreSQLConnection)
            else "MySQL"
            if isinstance(self.db_conn, MySQLConnection)
            else "SQLite"
            if isinstance(self.db_conn, (SQLiteConnection, CSVConnection))
            else "database"
        )

        self.console.print(
            Panel.fit(
                """
███████  ██████  ██      ███████  █████  ██████  ███████ ██████
██      ██    ██ ██      ██      ██   ██ ██   ██ ██      ██   ██
███████ ██    ██ ██      ███████ ███████ ██████  █████   ██████
     ██ ██ ▄▄ ██ ██           ██ ██   ██ ██   ██ ██      ██   ██
███████  ██████  ███████ ███████ ██   ██ ██████  ███████ ██   ██
            ▀▀
"""
            )
        )
        self.console.print(
            "\n",
            "[dim] ≥ Use '/clear' to reset conversation",
            "[dim] ≥ Use '/exit' or '/quit' to leave[/dim]",
            "[dim] ≥ Use 'Ctrl+C' to interrupt and return to prompt\n\n",
            "[dim] ≥ Start message with '#' to add something to agent's memory for this database",
            "[dim] ≥ Type '@' to get table name completions",
            "[dim] ≥ Press 'Esc-Enter' or 'Meta-Enter' to submit your question",
            sep="\n",
        )

        self.console.print(
            f"[bold blue]\n\nConnected to:[/bold blue] {db_name} ({db_type})\n"
        )

    async def _update_table_cache(self):
        """Update the table completer cache with fresh data."""
        try:
            tables_data = await SchemaManager(self.db_conn).list_tables()

            # Parse the table information
            table_list = []
            if isinstance(tables_data, dict) and "tables" in tables_data:
                for table in tables_data["tables"]:
                    if isinstance(table, dict):
                        name = table.get("name", "")
                        schema = table.get("schema", "")
                        full_name = table.get("full_name", "")

                        # Use full_name if available, otherwise construct it
                        if full_name:
                            table_name = full_name
                        elif schema and schema != "main":
                            table_name = f"{schema}.{name}"
                        else:
                            table_name = name

                        # No description needed - cleaner completions
                        table_list.append((table_name, ""))

            # Update the completer cache
            self.table_completer.update_cache(table_list)

        except Exception:
            # If there's an error, just use empty cache
            self.table_completer.update_cache([])

    async def _execute_query_with_cancellation(self, user_query: str):
        """Execute a query with cancellation support."""
        # Create cancellation token
        self.cancellation_token = asyncio.Event()

        # Create the query task
        query_task = asyncio.create_task(
            self.streaming_handler.execute_streaming_query(
                user_query, self.agent, self.cancellation_token, self.message_history
            )
        )
        self.current_task = query_task

        try:
            run_result = await query_task
            # Persist message history from this run using pydantic-ai API
            if run_result is not None:
                try:
                    # Use all_messages() so the system prompt and all prior turns are preserved
                    self.message_history = run_result.all_messages()
                except Exception:
                    pass
        finally:
            self.current_task = None
            self.cancellation_token = None

    async def run(self):
        """Run the interactive session loop."""
        self.show_welcome_message()

        # Initialize table cache
        await self._update_table_cache()

        while True:
            try:
                user_query = await questionary.text(
                    ">",
                    qmark="",
                    multiline=True,
                    instruction="",
                    completer=CompositeCompleter(
                        SlashCommandCompleter(), self.table_completer
                    ),
                ).ask_async()

                if not user_query:
                    continue

                if (
                    user_query in ["/exit", "/quit"]
                    or user_query.startswith("/exit")
                    or user_query.startswith("/quit")
                ):
                    break

                if user_query == "/clear":
                    # Reset local history (pydantic-ai call will receive empty history on next run)
                    self.message_history = []
                    self.console.print("[green]Conversation history cleared.[/green]\n")
                    continue

                if memory_text := user_query.strip():
                    # Check if query starts with # for memory addition
                    if memory_text.startswith("#"):
                        memory_content = memory_text[1:].strip()  # Remove # and trim
                        if memory_content:
                            # Add memory via the agent's memory manager
                            try:
                                mm = getattr(
                                    self.agent, "_sqlsaber_memory_manager", None
                                )
                                if mm and self.database_name:
                                    memory = mm.add_memory(
                                        self.database_name, memory_content
                                    )
                                    self.console.print(
                                        f"[green]✓ Memory added:[/green] {memory_content}"
                                    )
                                    self.console.print(
                                        f"[dim]Memory ID: {memory.id}[/dim]\n"
                                    )
                                else:
                                    self.console.print(
                                        "[yellow]Could not add memory (no database context)[/yellow]\n"
                                    )
                            except Exception:
                                self.console.print(
                                    "[yellow]Could not add memory[/yellow]\n"
                                )
                        else:
                            self.console.print(
                                "[yellow]Empty memory content after '#'[/yellow]\n"
                            )
                        continue

                    # Execute query with cancellation support
                    await self._execute_query_with_cancellation(user_query)
                    self.display.show_newline()  # Empty line for readability

            except KeyboardInterrupt:
                # Handle Ctrl+C - cancel current task if running
                if self.current_task and not self.current_task.done():
                    if self.cancellation_token is not None:
                        self.cancellation_token.set()
                    self.current_task.cancel()
                    try:
                        await self.current_task
                    except asyncio.CancelledError:
                        pass
                    self.console.print("\n[yellow]Query interrupted[/yellow]")
                else:
                    self.console.print(
                        "\n[yellow]Use '/exit' or '/quit' to leave.[/yellow]"
                    )
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
