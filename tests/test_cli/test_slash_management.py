from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sqlsaber.cli.command_catalog import (
    PaletteMode,
    management_paths,
    palette_commands,
)
from sqlsaber.cli.interactive import InteractiveSession
from sqlsaber.cli.output import out
from sqlsaber.cli.slash_commands import (
    CommandContext,
    CommandResult,
    SlashCommandProcessor,
    ThreadResumeRequest,
)
from sqlsaber.cli.threads import (
    PreparedThreadResume,
    ThreadResumePreparationError,
)
from sqlsaber.cli.tui_chat import ChatApp
from sqlsaber.cli.usage import SessionUsage, UsageMeter
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.render import bind_cli_surfaces, blocks as b
from sqlsaber.render.markdown_text import md_of

EXPECTED_MANAGEMENT_PATHS = {
    ("auth", "setup"),
    ("auth", "status"),
    ("auth", "reset"),
    ("db", "add"),
    ("db", "list"),
    ("db", "exclude"),
    ("db", "remove"),
    ("db", "set-default"),
    ("db", "test"),
    ("knowledge", "add"),
    ("knowledge", "list"),
    ("knowledge", "show"),
    ("knowledge", "search"),
    ("knowledge", "remove"),
    ("knowledge", "clear"),
    ("models", "list"),
    ("models", "set"),
    ("models", "current"),
    ("models", "reset"),
    ("theme", "set"),
    ("theme", "reset"),
    ("threads", "list"),
    ("threads", "show"),
    ("threads", "artifacts"),
    ("threads", "resume"),
    ("threads", "prune"),
    ("threads", "export"),
}


def _context() -> CommandContext:
    saber = MagicMock()
    saber.end_thread = AsyncMock(return_value=None)
    saber.new_thread = AsyncMock(return_value=None)
    saber.info = SimpleNamespace(
        thinking=SimpleNamespace(enabled=False, level=ThinkingLevel.MEDIUM)
    )
    return CommandContext(surface=MagicMock(), saber=saber)


def _output(context: CommandContext) -> str:
    return "\n\n".join(md_of(call.args) for call in context.surface.emit.call_args_list)


def test_registry_has_exact_management_parity() -> None:
    assert management_paths() == EXPECTED_MANAGEMENT_PATHS
    assert len(management_paths()) == 27


def test_palette_projects_every_management_command_from_registry() -> None:
    rows = {row.command: row for row in palette_commands()}

    assert {f"/{' '.join(path)}" for path in EXPECTED_MANAGEMENT_PATHS} <= rows.keys()
    assert rows["/db list"].mode is PaletteMode.SUBMIT
    assert rows["/db remove"].mode is PaletteMode.FILL
    assert rows["/thinking"].mode is PaletteMode.THINKING


def test_palette_submits_zero_arg_commands_and_fills_argument_commands() -> None:
    rows = {row.command: row for row in palette_commands()}
    app = MagicMock()

    ChatApp._run_command_palette_action(app, rows["/db list"])
    app.submit.assert_called_once_with("/db list")

    app.reset_mock()
    ChatApp._run_command_palette_action(app, rows["/db remove"])
    app.editor.set_text.assert_called_once_with("/db remove ")
    app.submit.assert_not_called()


@pytest.mark.asyncio
async def test_read_only_management_dispatch_invokes_existing_handler() -> None:
    context = _context()
    database = SimpleNamespace(
        name="CaseDB",
        type="sqlite",
        host="localhost",
        port=0,
        database="Case.DB",
        username="sqlite",
        exclude_schemas=[],
        ssl_mode=None,
        ssl_ca=None,
        ssl_cert=None,
    )
    manager = MagicMock()
    manager.list_databases.return_value = [database]
    manager.get_default_name.return_value = "CaseDB"

    with patch("sqlsaber.cli.database.config_manager", manager):
        result = await SlashCommandProcessor().process("/db list", context)

    assert result.handled is True
    manager.list_databases.assert_called_once_with()
    assert "CaseDB" in _output(context)


@pytest.mark.asyncio
async def test_cyclopts_validation_is_rendered_in_chat() -> None:
    context = _context()

    result = await SlashCommandProcessor().process("/db remove", context)

    assert result.handled is True
    assert "requires an argument" in _output(context)
    assert "Usage: `/db remove NAME [--yes]`" in _output(context)


@pytest.mark.asyncio
async def test_dispatch_preserves_quoted_values_and_case() -> None:
    context = _context()
    config = MagicMock()
    config.get_database.return_value = SimpleNamespace(name="Analytics")
    manager = MagicMock()
    manager.add_knowledge = AsyncMock(
        return_value=SimpleNamespace(id="entry-1", name="Revenue KPI")
    )

    with (
        patch("sqlsaber.cli.knowledge.config_manager", config),
        patch("sqlsaber.cli.knowledge._knowledge_manager", manager),
    ):
        result = await SlashCommandProcessor().process(
            '/knowledge add "Revenue KPI" "Case Sensitive Value" -d Analytics',
            context,
        )

    assert result.handled is True
    manager.add_knowledge.assert_awaited_once_with(
        database_name="Analytics",
        name="Revenue KPI",
        description="Case Sensitive Value",
        sql=None,
        source=None,
    )


@pytest.mark.asyncio
async def test_malformed_quotes_are_rejected_before_dispatch() -> None:
    context = _context()

    result = await SlashCommandProcessor().process('/knowledge search "open', context)

    assert result.handled is True
    assert "Unterminated quoted value" in _output(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", ["/database", "/model", "/thread", "/k"])
async def test_group_aliases_render_registry_help(alias: str) -> None:
    context = _context()

    result = await SlashCommandProcessor().process(f"{alias} --help", context)

    assert result.handled is True
    assert "commands" in _output(context)


@pytest.mark.asyncio
async def test_help_alias_group_leaf_and_flag_use_registry() -> None:
    processor = SlashCommandProcessor()
    context = _context()

    await processor.process("/?", context)
    assert "Session commands" in _output(context)
    assert "Management commands" in _output(context)

    context.surface.reset_mock()
    await processor.process("/help db", context)
    assert "`/db` commands" in _output(context)
    assert "/db add" in _output(context)

    context.surface.reset_mock()
    await processor.process("/database remove --help", context)
    assert "Usage: `/db remove NAME [--yes]`" in _output(context)


@pytest.mark.asyncio
async def test_exact_aliases_and_prefix_regressions() -> None:
    processor = SlashCommandProcessor()
    context = _context()

    result = await processor.process("/quit", context)
    assert result.should_exit is True

    for text in ("/exit-now", "/thinkingxyz", "/unknown"):
        context.surface.reset_mock()
        result = await processor.process(text, context)
        assert result.handled is True
        assert "Unknown slash command" in _output(context)

    for text in ("quitter", "exit report"):
        result = await processor.process(text, context)
        assert result.handled is False


@pytest.mark.asyncio
async def test_thread_resume_uses_cyclopts_parsing_without_nested_dispatch() -> None:
    context = _context()

    result = await SlashCommandProcessor().process(
        "/threads resume Thread-ABC -d Analytics -d Reporting", context
    )

    assert result.resume_request == ThreadResumeRequest(
        thread_id="Thread-ABC", databases=("Analytics", "Reporting")
    )


@pytest.mark.asyncio
async def test_context_bound_rendering_isolated_across_worker_threads() -> None:
    first = MagicMock()
    second = MagicMock()

    async def emit(surface: MagicMock, text: str) -> None:
        with bind_cli_surfaces(surface):
            await asyncio.to_thread(lambda: out(b.success(text)))

    await asyncio.gather(emit(first, "first"), emit(second, "second"))

    assert "first" in md_of(first.emit.call_args.args)
    assert "second" in md_of(second.emit.call_args.args)
    assert "second" not in md_of(first.emit.call_args.args)


@pytest.mark.asyncio
async def test_slash_commands_are_not_written_to_disk_history() -> None:
    session = InteractiveSession.__new__(InteractiveSession)
    session.current_task = None
    session._handoff_mode = False
    session._append_history = MagicMock()
    session.command_processor = MagicMock()
    session.command_processor.process = AsyncMock(
        return_value=CommandResult(handled=True)
    )
    session.saber = MagicMock()
    session.usage = UsageMeter(model_id=session._model_id)
    session.log = MagicMock()
    app = MagicMock()

    await session._handle_submit(
        app,
        MagicMock(),
        "/threads resume id -d postgresql://user:secret@host/db",
    )

    session._append_history.assert_not_called()


def _prepared(saber: MagicMock) -> PreparedThreadResume:
    return PreparedThreadResume(
        saber=saber,
        thread_id="thread-new",
        history=[],
        hydrated_results={},
        unavailable_results=set(),
        unavailable_artifacts=set(),
        resolved_artifacts={},
        storage=object(),
        artifact_store=object(),
        query_result_store=object(),
    )


@pytest.mark.asyncio
async def test_resume_of_active_thread_does_not_prepare_duplicate() -> None:
    old = MagicMock()
    old.info = SimpleNamespace(thread_id="thread-new")
    session = InteractiveSession.__new__(InteractiveSession)
    session.saber = old
    prepare = AsyncMock()
    surface = MagicMock()

    with patch("sqlsaber.cli.threads.prepare_thread_resume", prepare):
        await session._resume_thread(
            MagicMock(), surface, ThreadResumeRequest("thread-new")
        )

    prepare.assert_not_awaited()
    assert "already active" in _output(CommandContext(surface=surface, saber=old))


@pytest.mark.asyncio
async def test_resume_swaps_after_preparation_and_refreshes_session() -> None:
    old = MagicMock()
    old.info = SimpleNamespace(thread_id="thread-old")
    old.end_thread = AsyncMock(return_value="thread-old")
    old.close = AsyncMock()
    new = MagicMock()
    new.info = SimpleNamespace(
        thread_id="thread-new",
        database_names=("Analytics",),
        primary_database_name="Analytics",
        primary_database_type="SQLite",
        model_name="test-model",
        dangerous_mode=False,
    )
    new.close = AsyncMock()
    new.display_registry = {}
    new.query_result_store = MagicMock()
    new.list_tables = AsyncMock(return_value=[])
    session = InteractiveSession.__new__(InteractiveSession)
    session.saber = old
    session.usage = UsageMeter(
        model_id=session._model_id, on_change=session._refresh_footer
    )
    session.usage._session = SessionUsage(total_input_tokens=10)
    session.autocomplete_provider = MagicMock()
    session.streaming_handler = MagicMock()
    app = MagicMock()
    surface = MagicMock()

    with (
        patch(
            "sqlsaber.cli.threads.prepare_thread_resume",
            AsyncMock(return_value=_prepared(new)),
        ),
        patch("sqlsaber.cli.threads.render_prepared_thread") as render,
    ):
        await session._resume_thread(app, surface, ThreadResumeRequest("thread-new"))

    old.end_thread.assert_awaited_once_with()
    old.close.assert_awaited_once_with()
    assert session.saber is new
    app.clear_chat.assert_called_once_with()
    render.assert_called_once()
    new.list_tables.assert_awaited_once_with()
    assert session.usage.session.total_input_tokens == 0
    await session.saber.close()
    new.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_resume_keeps_new_session_when_old_cleanup_fails() -> None:
    old = MagicMock()
    old.info = SimpleNamespace(thread_id="thread-old")
    old.end_thread = AsyncMock(return_value="thread-old")
    old.close = AsyncMock(side_effect=RuntimeError("cleanup broke"))
    new = MagicMock()
    new.info = SimpleNamespace(
        thread_id="thread-new",
        database_names=("Analytics",),
        primary_database_name="Analytics",
        primary_database_type="SQLite",
        model_name="test-model",
        dangerous_mode=False,
    )
    new.display_registry = {}
    new.query_result_store = MagicMock()
    new.list_tables = AsyncMock(return_value=[])
    session = InteractiveSession.__new__(InteractiveSession)
    session.saber = old
    session.usage = UsageMeter(
        model_id=session._model_id, on_change=session._refresh_footer
    )
    session.autocomplete_provider = MagicMock()
    session.streaming_handler = MagicMock()
    session.log = MagicMock()
    surface = MagicMock()

    with (
        patch(
            "sqlsaber.cli.threads.prepare_thread_resume",
            AsyncMock(return_value=_prepared(new)),
        ),
        patch("sqlsaber.cli.threads.render_prepared_thread"),
    ):
        await session._resume_thread(
            MagicMock(), surface, ThreadResumeRequest("thread-new")
        )

    assert session.saber is new
    assert "Previous session cleanup failed" in _output(
        CommandContext(surface=surface, saber=new)
    )


@pytest.mark.asyncio
async def test_resume_preparation_failure_keeps_old_session() -> None:
    old = MagicMock()
    old.info = SimpleNamespace(thread_id="thread-old")
    old.end_thread = AsyncMock()
    old.close = AsyncMock()
    session = InteractiveSession.__new__(InteractiveSession)
    session.saber = old

    with patch(
        "sqlsaber.cli.threads.prepare_thread_resume",
        AsyncMock(side_effect=ThreadResumePreparationError("cannot resume")),
    ):
        with pytest.raises(ThreadResumePreparationError, match="cannot resume"):
            await session._resume_thread(
                MagicMock(), MagicMock(), ThreadResumeRequest("thread-new")
            )

    assert session.saber is old
    old.end_thread.assert_not_awaited()
    old.close.assert_not_awaited()
