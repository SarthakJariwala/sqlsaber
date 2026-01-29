from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from rich.console import Console

from sqlsaber.cli.display import DisplayManager
from sqlsaber.config.settings import ThinkingLevel

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
    from sqlsaber.cli.usage import SessionUsage
    from sqlsaber.threads.manager import ThreadManager

# Valid thinking level strings for slash command
THINKING_LEVELS = {"minimal", "low", "medium", "high", "maximum"}


@dataclass
class CommandContext:
    """Context passed to slash command handlers."""

    console: Console
    agent: "SQLSaberAgent"
    thread_manager: "ThreadManager"
    on_clear_history: Callable[[], None]
    session_usage: "SessionUsage | None" = None


@dataclass
class CommandResult:
    """Result of command processing."""

    handled: bool
    should_exit: bool = False
    handoff_goal: str | None = None


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

        # Handle /thinking command with various arguments
        if query.startswith("/thinking"):
            return await self._handle_thinking_command(context, query)

        # Handle /handoff command
        if query.startswith("/handoff"):
            return await self._handle_handoff(context, user_query)

        return CommandResult(handled=False)

    async def _handle_exit(self, context: CommandContext) -> CommandResult:
        """Handle exit commands."""
        if context.session_usage is not None:
            display = DisplayManager(context.console)
            display.show_session_summary(context.session_usage)
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

    async def _handle_thinking_command(
        self, context: CommandContext, query: str
    ) -> CommandResult:
        """Handle /thinking commands with various arguments.

        Supported formats:
            /thinking           - Show current status and level
            /thinking on        - Enable thinking with current level
            /thinking off       - Disable thinking
            /thinking <level>   - Set level (implies enable)
        """
        parts = query.split(maxsplit=1)
        arg = parts[1].strip() if len(parts) > 1 else ""

        # No argument: show current status
        if not arg:
            return await self._show_thinking_status(context)

        # Handle "on" - enable with current level
        if arg == "on":
            context.agent.set_thinking(enabled=True)
            level = context.agent.thinking_level
            context.console.print(
                f"[success]✓ Thinking: enabled ({level.value})[/success]\n"
            )
            return CommandResult(handled=True)

        # Handle "off" - disable thinking
        if arg == "off":
            context.agent.set_thinking(enabled=False)
            context.console.print("[success]✓ Thinking: disabled[/success]\n")
            return CommandResult(handled=True)

        # Handle thinking levels
        if arg in THINKING_LEVELS:
            level = ThinkingLevel(arg)
            context.agent.set_thinking(enabled=True, level=level)
            context.console.print(
                f"[success]✓ Thinking: enabled ({level.value})[/success]\n"
            )
            return CommandResult(handled=True)

        # Invalid argument
        valid_args = ", ".join(sorted(THINKING_LEVELS | {"on", "off"}))
        context.console.print(
            f"[warning]Invalid argument. Use: /thinking [{valid_args}][/warning]\n"
        )
        return CommandResult(handled=True)

    async def _show_thinking_status(self, context: CommandContext) -> CommandResult:
        """Show current thinking status and level."""
        enabled = context.agent.thinking_enabled
        level = context.agent.thinking_level

        if enabled:
            context.console.print(f"[info]Thinking: enabled ({level.value})[/info]\n")
        else:
            context.console.print("[info]Thinking: disabled[/info]\n")

        return CommandResult(handled=True)

    async def _handle_handoff(
        self, context: CommandContext, raw_query: str
    ) -> CommandResult:
        """Handle /handoff command.

        Usage: /handoff <goal>
        Returns a CommandResult with the handoff goal for InteractiveSession to process.
        """
        parts = raw_query.split(maxsplit=1)
        goal = parts[1].strip() if len(parts) > 1 else ""

        if not goal:
            context.console.print(
                "[warning]Usage: /handoff <goal>[/warning]\n"
                "[muted]Example: /handoff now optimize this query for performance[/muted]\n"
            )
            return CommandResult(handled=True)

        return CommandResult(handled=True, handoff_goal=goal)
