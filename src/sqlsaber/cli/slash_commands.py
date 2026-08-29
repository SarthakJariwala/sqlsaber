from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.render import blocks as b
from sqlsaber.render.surface import Surface

if TYPE_CHECKING:
    from sqlsaber import SQLSaber
    from sqlsaber.cli.usage import SessionUsage

THINKING_LEVELS = {"minimal", "low", "medium", "high", "maximum"}


@dataclass
class CommandContext:
    """Context passed to slash command handlers."""

    surface: Surface
    saber: "SQLSaber"
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

        if query in self.EXIT_COMMANDS or any(
            query.startswith(cmd) for cmd in self.EXIT_COMMANDS
        ):
            return await self._handle_exit(context)

        if query == "/clear":
            return await self._handle_clear(context)

        if query.startswith("/thinking"):
            return await self._handle_thinking_command(context, query)

        if query.startswith("/handoff"):
            return await self._handle_handoff(context, user_query)

        return CommandResult(handled=False)

    async def _handle_exit(self, context: CommandContext) -> CommandResult:
        """Handle exit commands."""
        ended_thread_id = await context.saber.end_thread()
        if ended_thread_id:
            hint = f"saber threads resume {ended_thread_id}"
            context.surface.emit(
                b.md(f"You can continue this thread using: `{hint}`", role="muted")
            )
        return CommandResult(handled=True, should_exit=True)

    async def _handle_clear(self, context: CommandContext) -> CommandResult:
        """Handle /clear command."""
        await context.saber.new_thread()
        context.surface.emit(b.success("Conversation history cleared."))
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

        if not arg:
            return await self._show_thinking_status(context)

        if arg == "on":
            state = context.saber.set_thinking(enabled=True)
            context.surface.emit(b.success(f"Thinking: enabled ({state.level.value})"))
            return CommandResult(handled=True)

        if arg == "off":
            context.saber.set_thinking(enabled=False)
            context.surface.emit(b.success("Thinking: disabled"))
            return CommandResult(handled=True)

        if arg in THINKING_LEVELS:
            level = ThinkingLevel(arg)
            context.saber.set_thinking(enabled=True, level=level)
            context.surface.emit(b.success(f"Thinking: enabled ({level.value})"))
            return CommandResult(handled=True)

        valid_args = ", ".join(sorted(THINKING_LEVELS | {"on", "off"}))
        context.surface.emit(b.warn(f"Invalid argument. Use: /thinking [{valid_args}]"))
        return CommandResult(handled=True)

    async def _show_thinking_status(self, context: CommandContext) -> CommandResult:
        """Show current thinking status and level."""
        thinking = context.saber.info.thinking

        if thinking.enabled:
            context.surface.emit(
                b.md(f"Thinking: enabled ({thinking.level.value})", role="info")
            )
        else:
            context.surface.emit(b.md("Thinking: disabled", role="info"))

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
            context.surface.emit(
                b.warn("Usage: /handoff <goal>"),
                b.md(
                    "Example: `/handoff now optimize this query for performance`",
                    role="muted",
                ),
            )
            return CommandResult(handled=True)

        return CommandResult(handled=True, handoff_goal=goal)
