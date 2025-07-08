"""Interactive mode handling for the CLI."""

import asyncio

import questionary
from rich.console import Console
from rich.panel import Panel

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.cli.completers import (
    CompositeCompleter,
    SlashCommandCompleter,
    TableNameCompleter,
)
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.streaming import StreamingQueryHandler


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(self, console: Console, agent: BaseSQLAgent):
        self.console = console
        self.agent = agent
        self.display = DisplayManager(console)
        self.streaming_handler = StreamingQueryHandler(console)
        self.current_task: asyncio.Task | None = None
        self.cancellation_token: asyncio.Event | None = None
        self.table_completer = TableNameCompleter()

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        # Show database information
        db_name = getattr(self.agent, "database_name", None) or "Unknown"
        db_type = self.agent._get_database_type_name()

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
                "\n\n"
                "[dim]Use '/clear' to reset conversation, '/exit' or '/quit' to leave.[/dim]\n\n"
                "[dim]Start a message with '#' to add something to agent's memory for this database.[/dim]\n\n"
                "[dim]Type '@' to get table name completions.[/dim]",
                border_style="green",
            )
        )
        self.console.print(
            f"[bold blue]Connected to:[/bold blue] {db_name} ({db_type})\n"
        )
        self.console.print(
            "[dim]Press Esc-Enter or Meta-Enter to submit your query.[/dim]\n"
            "[dim]Press Ctrl+C during query execution to interrupt and return to prompt.[/dim]\n"
        )

    async def _update_table_cache(self):
        """Update the table completer cache with fresh data."""
        try:
            # Use the schema manager directly which has built-in caching
            tables_data = await self.agent.schema_manager.list_tables()

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
                user_query, self.agent, self.cancellation_token
            )
        )
        self.current_task = query_task

        try:
            # Simply await the query task
            # Ctrl+C will be handled by the KeyboardInterrupt exception in run()
            await query_task

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

                if user_query in ["/exit", "/quit"]:
                    break

                if user_query == "/clear":
                    self.agent.clear_history()
                    self.console.print("[green]Conversation history cleared.[/green]\n")
                    continue

                if memory_text := user_query.strip():
                    # Check if query starts with # for memory addition
                    if memory_text.startswith("#"):
                        memory_content = memory_text[1:].strip()  # Remove # and trim
                        if memory_content:
                            # Add memory
                            memory_id = self.agent.add_memory(memory_content)
                            if memory_id:
                                self.console.print(
                                    f"[green]✓ Memory added:[/green] {memory_content}"
                                )
                                self.console.print(
                                    f"[dim]Memory ID: {memory_id}[/dim]\n"
                                )
                            else:
                                self.console.print(
                                    "[yellow]Could not add memory (no database context)[/yellow]\n"
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
