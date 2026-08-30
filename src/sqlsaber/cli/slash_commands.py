from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlsaber.cli.command_catalog import (
    COMMAND_SPECS,
    CommandKind,
    CommandSpec,
    PaletteCommand,
    palette_commands,
)
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.render import bind_cli_surfaces, blocks as b
from sqlsaber.render.surface import Surface

if TYPE_CHECKING:
    from sqlsaber import SQLSaber
    from sqlsaber.cli.usage import SessionUsage

THINKING_LEVELS = {"minimal", "low", "medium", "high", "maximum"}


@dataclass(frozen=True, slots=True)
class ThreadResumeRequest:
    thread_id: str
    databases: tuple[str, ...] = ()


@dataclass
class CommandContext:
    surface: Surface
    saber: SQLSaber
    session_usage: SessionUsage | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    handled: bool
    should_exit: bool = False
    handoff_goal: str | None = None
    resume_request: ThreadResumeRequest | None = None


@dataclass(frozen=True, slots=True)
class _Invocation:
    spec: CommandSpec
    arguments: tuple[str, ...]


class _TokenError(ValueError):
    pass


def _split_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    started = False
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == quote:
                quote = None
            elif (
                char == "\\"
                and index + 1 < len(text)
                and text[index + 1]
                in {
                    quote,
                    "\\",
                }
            ):
                index += 1
                current.append(text[index])
            else:
                current.append(char)
            started = True
        elif char in {"'", '"'}:
            quote = char
            started = True
        elif char.isspace():
            if started:
                tokens.append("".join(current))
                current = []
                started = False
        else:
            current.append(char)
            started = True
        index += 1
    if quote is not None:
        raise _TokenError("Unterminated quoted value.")
    if started:
        tokens.append("".join(current))
    return tuple(tokens)


def _path_variants(spec: CommandSpec) -> tuple[tuple[str, ...], ...]:
    return (spec.path, *spec.aliases)


def _resolve(tokens: tuple[str, ...]) -> _Invocation | None:
    folded = tuple(token.casefold() for token in tokens)
    matches: list[tuple[int, CommandSpec]] = []
    for spec in COMMAND_SPECS:
        for path in _path_variants(spec):
            if folded[: len(path)] == path:
                matches.append((len(path), spec))
    if not matches:
        return None
    path_length, spec = max(matches, key=lambda item: item[0])
    return _Invocation(spec, tokens[path_length:])


def _group_specs(path: tuple[str, ...]) -> tuple[CommandSpec, ...]:
    folded = tuple(part.casefold() for part in path)
    matches: list[CommandSpec] = []
    for spec in COMMAND_SPECS:
        if len(spec.path) < len(folded):
            continue
        if any(variant[: len(folded)] == folded for variant in _path_variants(spec)):
            matches.append(spec)
    return tuple(matches)


def _exact_spec(path: tuple[str, ...]) -> CommandSpec | None:
    folded = tuple(part.casefold() for part in path)
    for spec in COMMAND_SPECS:
        if folded in _path_variants(spec):
            return spec
    return None


class SlashCommandProcessor:
    async def process(self, user_query: str, context: CommandContext) -> CommandResult:
        stripped = user_query.strip()
        folded = stripped.casefold()
        if folded in {"exit", "quit"}:
            return await self._handle_exit(context)
        if not stripped.startswith("/"):
            return CommandResult(handled=False)

        try:
            tokens = _split_tokens(stripped[1:])
        except _TokenError as exc:
            context.surface.emit(
                b.error(str(exc)), b.md("Use `/help` for command syntax.")
            )
            return CommandResult(handled=True)

        if not tokens:
            self._show_help(context, ())
            return CommandResult(handled=True)

        invocation = _resolve(tokens)
        if invocation is None:
            group_path = tokens[:1] if tokens[1:] == ("--help",) else tokens
            groups = _group_specs(group_path)
            if groups and (len(tokens) == 1 or tokens[1:] == ("--help",)):
                self._show_help(context, group_path)
            else:
                context.surface.emit(
                    b.error(f"Unknown slash command: /{' '.join(tokens)}"),
                    b.md("Use `/help` to list available commands.", role="muted"),
                )
            return CommandResult(handled=True)

        spec = invocation.spec
        arguments = invocation.arguments
        if spec.path == ("help",):
            self._show_help(context, arguments)
            return CommandResult(handled=True)
        if "--help" in arguments:
            self._show_help(context, spec.path)
            return CommandResult(handled=True)
        if spec.kind is CommandKind.MANAGEMENT:
            return await self._handle_management(spec, arguments, context)
        if spec.path == ("exit",):
            if arguments:
                return self._argument_error(spec, context)
            return await self._handle_exit(context)
        if spec.path == ("clear",):
            if arguments:
                return self._argument_error(spec, context)
            return await self._handle_clear(context)
        if spec.path == ("thinking",):
            return await self._handle_thinking(context, spec, arguments)
        if spec.path == ("handoff",):
            return self._handle_handoff(context, spec, arguments)
        return CommandResult(handled=True)

    def palette_commands(self) -> tuple[PaletteCommand, ...]:
        return palette_commands()

    def _show_help(self, context: CommandContext, path: tuple[str, ...]) -> None:
        if not path:
            session = [
                spec for spec in COMMAND_SPECS if spec.kind is CommandKind.SESSION
            ]
            management = [
                spec for spec in COMMAND_SPECS if spec.kind is CommandKind.MANAGEMENT
            ]
            lines = ["## Session commands", *self._help_lines(session)]
            lines.extend(("", "## Management commands", *self._help_lines(management)))
            context.surface.emit(b.md("\n".join(lines)))
            return

        spec = _exact_spec(path)
        if spec is not None:
            aliases = [f"/{' '.join(alias)}" for alias in spec.aliases]
            lines = [f"## `{spec.command}`", spec.summary, "", f"Usage: `{spec.usage}`"]
            if aliases:
                lines.extend(
                    ("", f"Aliases: {', '.join(f'`{alias}`' for alias in aliases)}")
                )
            if spec.options:
                labels = ["/".join(option.names) for option in spec.options]
                lines.extend(
                    ("", f"Options: {', '.join(f'`{name}`' for name in labels)}")
                )
            context.surface.emit(b.md("\n".join(lines)))
            return

        group = _group_specs(path)
        if group:
            canonical_group = group[0].path[0]
            context.surface.emit(
                b.md(
                    "\n".join(
                        [f"## `/{canonical_group}` commands", *self._help_lines(group)]
                    )
                )
            )
            return

        context.surface.emit(
            b.error(f"Unknown help topic: {' '.join(path)}"),
            b.md("Use `/help` to list available commands.", role="muted"),
        )

    @staticmethod
    def _help_lines(specs: list[CommandSpec] | tuple[CommandSpec, ...]) -> list[str]:
        return [f"- `{spec.usage}`. {spec.summary}" for spec in specs]

    async def _handle_management(
        self,
        spec: CommandSpec,
        arguments: tuple[str, ...],
        context: CommandContext,
    ) -> CommandResult:
        from cyclopts.exceptions import CycloptsError
        from sqlsaber.cli.commands import app

        try:
            command, bound, _ = app.parse_args(
                [*spec.path, *arguments],
                exit_on_error=False,
                print_error=False,
            )
        except CycloptsError as exc:
            context.surface.emit(
                b.error(str(exc)), b.md(f"Usage: `{spec.usage}`", role="muted")
            )
            return CommandResult(handled=True)
        except SystemExit:
            return CommandResult(handled=True)

        if spec.path == ("threads", "resume"):
            values = bound.arguments
            database = values.get("database")
            databases = tuple(database) if isinstance(database, list) else ()
            return CommandResult(
                handled=True,
                resume_request=ThreadResumeRequest(
                    thread_id=str(values["thread_id"]), databases=databases
                ),
            )

        def invoke() -> None:
            command(*bound.args, **bound.kwargs)

        try:
            with bind_cli_surfaces(context.surface):
                await asyncio.to_thread(invoke)
        except SystemExit:
            pass
        return CommandResult(handled=True)

    @staticmethod
    def _argument_error(spec: CommandSpec, context: CommandContext) -> CommandResult:
        context.surface.emit(
            b.error(f"Unexpected arguments for {spec.command}."),
            b.md(f"Usage: `{spec.usage}`", role="muted"),
        )
        return CommandResult(handled=True)

    async def _handle_exit(self, context: CommandContext) -> CommandResult:
        ended_thread_id = await context.saber.end_thread()
        if ended_thread_id:
            hint = f"saber threads resume {ended_thread_id}"
            context.surface.emit(
                b.md(f"You can continue this thread using: `{hint}`", role="muted")
            )
        return CommandResult(handled=True, should_exit=True)

    async def _handle_clear(self, context: CommandContext) -> CommandResult:
        await context.saber.new_thread()
        context.surface.emit(b.success("Conversation history cleared."))
        return CommandResult(handled=True)

    async def _handle_thinking(
        self,
        context: CommandContext,
        spec: CommandSpec,
        arguments: tuple[str, ...],
    ) -> CommandResult:
        if len(arguments) > 1:
            return self._argument_error(spec, context)
        arg = arguments[0].casefold() if arguments else ""
        if not arg:
            thinking = context.saber.info.thinking
            if thinking.enabled:
                context.surface.emit(
                    b.md(f"Thinking: enabled ({thinking.level.value})", role="info")
                )
            else:
                context.surface.emit(b.md("Thinking: disabled", role="info"))
            return CommandResult(handled=True)
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

    @staticmethod
    def _handle_handoff(
        context: CommandContext,
        spec: CommandSpec,
        arguments: tuple[str, ...],
    ) -> CommandResult:
        goal = " ".join(arguments).strip()
        if not goal:
            context.surface.emit(
                b.warn(f"Usage: {spec.usage}"),
                b.md(
                    "Example: `/handoff now optimize this query for performance`",
                    role="muted",
                ),
            )
            return CommandResult(handled=True)
        return CommandResult(handled=True, handoff_goal=goal)
