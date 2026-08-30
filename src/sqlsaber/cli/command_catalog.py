from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CommandKind(StrEnum):
    SESSION = "session"
    MANAGEMENT = "management"


class PaletteMode(StrEnum):
    SUBMIT = "submit"
    FILL = "fill"
    THINKING = "thinking"


@dataclass(frozen=True, slots=True)
class OptionSpec:
    names: tuple[str, ...]
    takes_value: bool = True
    repeatable: bool = False


@dataclass(frozen=True, slots=True)
class CommandSpec:
    path: tuple[str, ...]
    summary: str
    usage: str
    kind: CommandKind
    aliases: tuple[tuple[str, ...], ...] = ()
    options: tuple[OptionSpec, ...] = ()
    palette_mode: PaletteMode = PaletteMode.SUBMIT
    palette_label: str | None = None

    @property
    def command(self) -> str:
        return f"/{' '.join(self.path)}"


@dataclass(frozen=True, slots=True)
class PaletteCommand:
    command: str
    label: str
    description: str
    mode: PaletteMode


def _option(
    *names: str,
    takes_value: bool = True,
    repeatable: bool = False,
) -> OptionSpec:
    return OptionSpec(names, takes_value=takes_value, repeatable=repeatable)


def _management(
    path: tuple[str, str],
    summary: str,
    usage: str,
    *,
    group_alias: str | None = None,
    options: tuple[OptionSpec, ...] = (),
    takes_arguments: bool = False,
) -> CommandSpec:
    aliases = ((group_alias, path[1]),) if group_alias is not None else ()
    return CommandSpec(
        path=path,
        aliases=aliases,
        summary=summary,
        usage=usage,
        kind=CommandKind.MANAGEMENT,
        options=options,
        palette_mode=PaletteMode.FILL if takes_arguments else PaletteMode.SUBMIT,
    )


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        path=("thinking",),
        summary="Show or change reasoning for this session.",
        usage="/thinking [on|off|minimal|low|medium|high|maximum]",
        kind=CommandKind.SESSION,
        palette_mode=PaletteMode.THINKING,
        palette_label="Thinking mode",
    ),
    CommandSpec(
        path=("handoff",),
        summary="Draft a prompt for a new thread with the current context.",
        usage="/handoff GOAL",
        kind=CommandKind.SESSION,
        palette_mode=PaletteMode.FILL,
        palette_label="Handoff thread",
    ),
    CommandSpec(
        path=("clear",),
        summary="Clear the current conversation history.",
        usage="/clear",
        kind=CommandKind.SESSION,
        palette_label="Clear conversation",
    ),
    CommandSpec(
        path=("exit",),
        aliases=(("quit",),),
        summary="End this interactive session.",
        usage="/exit",
        kind=CommandKind.SESSION,
        palette_label="Exit",
    ),
    CommandSpec(
        path=("help",),
        aliases=(("?",),),
        summary="Show slash-command help.",
        usage="/help [GROUP [COMMAND]]",
        kind=CommandKind.SESSION,
        palette_mode=PaletteMode.FILL,
        palette_label="Command help",
    ),
    _management(("auth", "setup"), "Configure API-key authentication.", "/auth setup"),
    _management(("auth", "status"), "Show authentication status.", "/auth status"),
    _management(
        ("auth", "reset"),
        "Reset stored credentials for a provider.",
        "/auth reset [PROVIDER] [--yes]",
        options=(_option("--yes", takes_value=False),),
        takes_arguments=True,
    ),
    _management(
        ("db", "add"),
        "Add a saved database connection.",
        "/db add NAME [OPTIONS]",
        group_alias="database",
        options=(
            _option("--type", "-t"),
            _option("--host", "-h"),
            _option("--port", "-p"),
            _option("--database", "--db"),
            _option("--username", "-u"),
            _option("--ssl-mode"),
            _option("--ssl-ca"),
            _option("--ssl-cert"),
            _option("--ssl-key"),
            _option("--exclude-schemas"),
            _option("--description"),
            _option("--interactive", "--no-interactive", takes_value=False),
        ),
        takes_arguments=True,
    ),
    _management(
        ("db", "list"),
        "List saved database connections.",
        "/db list",
        group_alias="database",
    ),
    _management(
        ("db", "exclude"),
        "Change excluded schemas for a connection.",
        "/db exclude NAME [--set CSV|--add CSV|--remove CSV|--clear]",
        group_alias="database",
        options=(
            _option("--set"),
            _option("--add"),
            _option("--remove"),
            _option("--clear", takes_value=False),
        ),
        takes_arguments=True,
    ),
    _management(
        ("db", "remove"),
        "Remove a saved database connection.",
        "/db remove NAME [--yes]",
        group_alias="database",
        options=(_option("--yes", takes_value=False),),
        takes_arguments=True,
    ),
    _management(
        ("db", "set-default"),
        "Set the default database connection.",
        "/db set-default NAME",
        group_alias="database",
        takes_arguments=True,
    ),
    _management(
        ("db", "test"),
        "Test a database connection.",
        "/db test [NAME]",
        group_alias="database",
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "add"),
        "Add database-specific knowledge.",
        "/knowledge add NAME DESCRIPTION [-d DATABASE] [--sql SQL] [--source SOURCE]",
        group_alias="k",
        options=(
            _option("--database", "-d"),
            _option("--sql"),
            _option("--source"),
        ),
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "list"),
        "List database-specific knowledge.",
        "/knowledge list [-d DATABASE]",
        group_alias="k",
        options=(_option("--database", "-d"),),
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "show"),
        "Show one knowledge entry.",
        "/knowledge show ID [-d DATABASE]",
        group_alias="k",
        options=(_option("--database", "-d"),),
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "search"),
        "Search database-specific knowledge.",
        "/knowledge search QUERY [-d DATABASE] [--limit N]",
        group_alias="k",
        options=(_option("--database", "-d"), _option("--limit")),
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "remove"),
        "Remove one knowledge entry.",
        "/knowledge remove ID [-d DATABASE] [--yes]",
        group_alias="k",
        options=(
            _option("--database", "-d"),
            _option("--yes", takes_value=False),
        ),
        takes_arguments=True,
    ),
    _management(
        ("knowledge", "clear"),
        "Clear knowledge for a database.",
        "/knowledge clear [-d DATABASE] [--yes]",
        group_alias="k",
        options=(
            _option("--database", "-d"),
            _option("--yes", takes_value=False),
        ),
        takes_arguments=True,
    ),
    _management(
        ("models", "list"),
        "List available models.",
        "/models list",
        group_alias="model",
    ),
    _management(
        ("models", "set"),
        "Set a saved model configuration.",
        "/models set [MODEL] [--agent AGENT] [--thinking-level LEVEL]",
        group_alias="model",
        options=(_option("--agent"), _option("--thinking-level")),
        takes_arguments=True,
    ),
    _management(
        ("models", "current"),
        "Show saved model configuration.",
        "/models current [--agent AGENT]",
        group_alias="model",
        options=(_option("--agent"),),
        takes_arguments=True,
    ),
    _management(
        ("models", "reset"),
        "Reset a saved model configuration.",
        "/models reset [--agent AGENT] [--yes]",
        group_alias="model",
        options=(_option("--agent"), _option("--yes", takes_value=False)),
        takes_arguments=True,
    ),
    _management(
        ("theme", "set"),
        "Set the theme for later sessions.",
        "/theme set [THEME]",
        takes_arguments=True,
    ),
    _management(
        ("theme", "reset"),
        "Reset the saved theme.",
        "/theme reset [--yes]",
        options=(_option("--yes", takes_value=False),),
        takes_arguments=True,
    ),
    _management(
        ("threads", "list"),
        "List retained threads.",
        "/threads list [-d DATABASE] [--limit N]",
        group_alias="thread",
        options=(
            _option("--database", "-d"),
            _option("--limit", "-n"),
        ),
        takes_arguments=True,
    ),
    _management(
        ("threads", "show"),
        "Show a retained thread and transcript.",
        "/threads show ID",
        group_alias="thread",
        takes_arguments=True,
    ),
    _management(
        ("threads", "artifacts"),
        "List durable artifacts for a thread.",
        "/threads artifacts ID",
        group_alias="thread",
        takes_arguments=True,
    ),
    _management(
        ("threads", "resume"),
        "Resume a retained thread in this session.",
        "/threads resume ID [-d DATABASE ...]",
        group_alias="thread",
        options=(_option("--database", "-d", repeatable=True),),
        takes_arguments=True,
    ),
    _management(
        ("threads", "prune"),
        "Prune old retained threads.",
        "/threads prune [--days N] [--dry-run] [--yes]",
        group_alias="thread",
        options=(
            _option("--days", "-n"),
            _option("--dry-run", takes_value=False),
            _option("--yes", takes_value=False),
        ),
        takes_arguments=True,
    ),
    _management(
        ("threads", "export"),
        "Export a retained thread as HTML.",
        "/threads export ID [-o PATH]",
        group_alias="thread",
        options=(_option("--output", "-o"),),
        takes_arguments=True,
    ),
)


def _validate_registry() -> None:
    paths: set[tuple[str, ...]] = set()
    aliases: set[tuple[str, ...]] = set()
    for spec in COMMAND_SPECS:
        if spec.path in paths or spec.path in aliases:
            raise RuntimeError(f"duplicate slash-command path: {spec.path}")
        paths.add(spec.path)
        option_names = [name for option in spec.options for name in option.names]
        if len(option_names) != len(set(option_names)):
            raise RuntimeError(f"duplicate options for slash command: {spec.path}")
        for alias in spec.aliases:
            if alias in paths or alias in aliases:
                raise RuntimeError(f"duplicate slash-command alias: {alias}")
            aliases.add(alias)


_validate_registry()


def management_paths() -> frozenset[tuple[str, ...]]:
    return frozenset(
        spec.path for spec in COMMAND_SPECS if spec.kind is CommandKind.MANAGEMENT
    )


def palette_commands() -> tuple[PaletteCommand, ...]:
    return tuple(
        PaletteCommand(
            command=spec.command,
            label=spec.palette_label or spec.command,
            description=spec.summary,
            mode=spec.palette_mode,
        )
        for spec in COMMAND_SPECS
    )
