"""Interactive mode handling for the CLI."""

import questionary
from rich.console import Console
from rich.panel import Panel

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.streaming import StreamingQueryHandler


class InteractiveSession:
    """Manages interactive CLI sessions."""

    def __init__(self, console: Console, agent: BaseSQLAgent):
        self.console = console
        self.agent = agent
        self.display = DisplayManager(console)
        self.streaming_handler = StreamingQueryHandler(console)

    def show_welcome_message(self):
        """Display welcome message for interactive mode."""
        self.console.print(
            Panel.fit(
                "[bold green]SQLSaber - Use the agent Luke![/bold green]\n\n"
                "Type your queries in natural language.\n\n"
                "Press Esc-Enter or Meta-Enter to submit your query.\n\n"
                "Type 'exit' or 'quit' to leave.",
                border_style="green",
            )
        )

        self.console.print(
            "[dim]Commands: 'clear' to reset conversation, 'exit' or 'quit' to leave[/dim]"
        )
        self.console.print(
            "[dim]Memory: Start a message with '#' to add it as a memory for this database[/dim]\n"
        )

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
                ).ask_async()

                if user_query.lower() in ["exit", "quit", "q"]:
                    break

                if user_query.lower() == "clear":
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

                    await self.streaming_handler.execute_streaming_query(
                        user_query, self.agent
                    )
                    self.display.show_newline()  # Empty line for readability

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' or 'quit' to leave.[/yellow]")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
