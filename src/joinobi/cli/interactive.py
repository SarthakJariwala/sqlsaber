"""Interactive mode handling for the CLI."""

from rich.console import Console
from rich.panel import Panel

from joinobi.agents.base import BaseSQLAgent
from joinobi.cli.display import DisplayManager
from joinobi.cli.streaming import StreamingQueryHandler


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
                "[bold green]JoinObi - Use the agent Luke![/bold green]\n\n"
                "Type your queries in natural language. Type 'exit' or 'quit' to leave.\n"
                "[dim]Now with conversation memory and real-time visibility![/dim]",
                border_style="green",
            )
        )

        self.console.print(
            "[dim]Commands: 'clear' to reset conversation, 'exit' or 'quit' to leave[/dim]\n"
        )

    async def run(self):
        """Run the interactive session loop."""
        self.show_welcome_message()

        while True:
            try:
                user_query = self.console.input("[bold blue]sql> [/bold blue]")

                if user_query.lower() in ["exit", "quit", "q"]:
                    break

                if user_query.lower() == "clear":
                    self.agent.clear_history()
                    self.console.print("[green]Conversation history cleared.[/green]\n")
                    continue

                if user_query.strip():
                    await self.streaming_handler.execute_streaming_query(
                        user_query, self.agent
                    )
                    self.display.show_newline()  # Empty line for readability

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use 'exit' or 'quit' to leave.[/yellow]")
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {str(e)}")
