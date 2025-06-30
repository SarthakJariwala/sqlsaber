"""Interactive mode handling for the CLI."""

import asyncio
from typing import Optional

import questionary
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.panel import Panel

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.streaming import StreamingQueryHandler


class SlashCommandCompleter(Completer):
    """Custom completer for slash commands."""

    def get_completions(self, document, complete_event):
        """Get completions for slash commands."""
        # Only provide completions if the line starts with "/"
        text = document.text
        if text.startswith("/"):
            # Get the partial command after the slash
            partial_cmd = text[1:]

            # Define available commands with descriptions
            commands = [
                ("clear", "Clear conversation history"),
                ("exit", "Exit the interactive session"),
                ("quit", "Exit the interactive session"),
            ]

            # Yield completions that match the partial command
            for cmd, description in commands:
                if cmd.startswith(partial_cmd):
                    yield Completion(
                        cmd,
                        start_position=-len(partial_cmd),
                        display_meta=description,
                    )


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(self, console: Console, agent: BaseSQLAgent):
        self.console = console
        self.agent = agent
        self.display = DisplayManager(console)
        self.streaming_handler = StreamingQueryHandler(console)
        self.current_task: Optional[asyncio.Task] = None
        self.cancellation_token: Optional[asyncio.Event] = None

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        # Show database information
        db_name = getattr(self.agent, "database_name", None) or "Unknown"
        db_type = self.agent._get_database_type_name()

        self.console.print(
            Panel.fit(
                "[bold green]SQLSaber - Use the agent Luke![/bold green]\n\n"
                "[bold]Your agentic SQL assistant.[/bold]\n\n\n"
                "[dim]Use '/clear' to reset conversation, '/exit' or '/quit' to leave.[/dim]\n\n"
                "[dim]Start a message with '#' to add something to agent's memory for this database.[/dim]",
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

        while True:
            try:
                user_query = await questionary.text(
                    ">",
                    qmark="",
                    multiline=True,
                    instruction="",
                    completer=SlashCommandCompleter(),
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
