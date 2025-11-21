from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rich.console import Console

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
    from sqlsaber.threads.manager import ThreadManager


@dataclass
class CommandContext:
    """Context passed to slash command handlers."""

    console: Console
    agent: "SQLSaberAgent"
    thread_manager: "ThreadManager"
    on_clear_history: Callable[[], None]


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool
    should_exit: bool = False


class SlashCommandProcessor:
    """Processes slash commands and special inputs."""

    EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}

    async def process(self, user_query: str, context: CommandContext) -> CommandResult:
        """
        Process a user query to see if it's a command.
        Returns CommandResult indicating if it was handled and if we should exit.
        """
        query = user_query.strip().lower()

        # Handle exit commands
        if query in self.EXIT_COMMANDS or any(
            query.startswith(cmd) for cmd in self.EXIT_COMMANDS
        ):
            return await self._handle_exit(context)

        if query == "/clear":
            return await self._handle_clear(context)

        if query == "/thinking on":
            return await self._handle_thinking(context, enabled=True)

        if query == "/thinking off":
            return await self._handle_thinking(context, enabled=False)

        return CommandResult(handled=False)

    async def _handle_exit(self, context: CommandContext) -> CommandResult:
        """Handle exit commands."""
        ended_thread_id = await context.thread_manager.end_current_thread()
        if ended_thread_id:
            hint = f"saber threads resume {ended_thread_id}"
            context.console.print(
                f"[muted]You can continue this thread using:[/muted] {hint}"
            )
        return CommandResult(handled=True, should_exit=True)

    async def _handle_clear(self, context: CommandContext) -> CommandResult:
        """Handle /clear command."""
        context.on_clear_history()
        await context.thread_manager.clear_current_thread()
        context.console.print("[success]Conversation history cleared.[/success]\n")
        return CommandResult(handled=True)

    async def _handle_thinking(
        self, context: CommandContext, enabled: bool
    ) -> CommandResult:
        """Handle /thinking on/off commands."""
        context.agent.set_thinking(enabled=enabled)
        state = "enabled" if enabled else "disabled"
        context.console.print(f"[success]✓ Thinking {state}[/success]\n")
        return CommandResult(handled=True)
