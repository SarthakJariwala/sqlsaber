"""Coverage for non-interactive, agent-friendly CLI behavior."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from sqlsaber.cli import database as database_cli
from sqlsaber.cli import models as models_cli
from sqlsaber.cli import theme as theme_cli
from sqlsaber.cli import threads as threads_cli
from sqlsaber.cli.commands import app, query
from sqlsaber.config.settings import ThinkingLevel


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["auth", "reset"],
        ["db", "add"],
        ["db", "remove"],
        ["knowledge", "remove"],
        ["knowledge", "clear"],
        ["models", "set"],
        ["models", "reset"],
        ["theme", "set"],
        ["theme", "reset"],
        ["threads", "prune"],
    ],
)
def test_affected_help_includes_examples(command, capsys):
    with pytest.raises(SystemExit) as exc_info:
        app([*command, "--help"])

    assert exc_info.value.code == 0
    assert "Example" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command",
    [
        ["auth", "reset"],
        ["db", "remove"],
        ["knowledge", "remove"],
        ["knowledge", "clear"],
        ["models", "reset"],
        ["theme", "reset"],
        ["threads", "prune"],
    ],
)
def test_destructive_help_has_only_the_yes_confirmation_option(command, capsys):
    with pytest.raises(SystemExit) as exc_info:
        app([*command, "--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--yes" in output
    assert "--force" not in output
    assert " -y" not in output


def test_models_set_directly_without_fetching_or_prompting():
    config = MagicMock()
    with (
        patch.object(
            models_cli.model_manager, "set_model", return_value=True
        ) as set_model,
        patch.object(
            models_cli.model_manager,
            "fetch_available_models",
            side_effect=AssertionError("direct model selection must not fetch"),
        ),
        patch.object(models_cli, "Config", return_value=config),
    ):
        models_cli.set_model_command(
            "openai:gpt-5", thinking_level=ThinkingLevel.HIGH.value
        )

    set_model.assert_called_once_with("openai:gpt-5")
    config.model.set_thinking.assert_called_once_with(True, ThinkingLevel.HIGH)


def test_models_set_accepts_xai_grok_4_6():
    with patch.object(
        models_cli.model_manager, "set_model", return_value=True
    ) as set_model:
        models_cli.set_model_command("xai:grok-4.6")

    set_model.assert_called_once_with("xai:grok-4.6")


def test_models_set_accepts_groq_without_confusing_xai():
    with patch.object(
        models_cli.model_manager, "set_model", return_value=True
    ) as set_model:
        models_cli.set_model_command("groq:llama-3-3-70b-versatile")

    set_model.assert_called_once_with("groq:llama-3-3-70b-versatile")


def test_models_set_rejects_deprecated_grok_prefix(capsys):
    with pytest.raises(SystemExit) as exc_info:
        models_cli.set_model_command("grok:grok-4.6")

    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "supported PROVIDER:MODEL" in err
    assert "xai" in err


def test_models_set_rejects_unsupported_provider(capsys):
    with pytest.raises(SystemExit) as exc_info:
        models_cli.set_model_command("unknown:model")

    assert exc_info.value.code == 2
    assert "supported PROVIDER:MODEL" in capsys.readouterr().err


def test_theme_set_directly_without_prompting():
    with (
        patch.object(
            theme_cli.theme_manager,
            "get_available_themes",
            return_value=["dracula", "nord"],
        ),
        patch.object(
            theme_cli.theme_manager, "set_theme", return_value=True
        ) as set_theme,
    ):
        theme_cli.set("dracula")

    set_theme.assert_called_once_with("dracula")


def test_db_add_reads_password_from_stdin_without_prompting():
    manager = MagicMock()
    manager.list_databases.return_value = [object()]
    stdin = StringIO("not-a-real-secret\n")

    with (
        patch.object(database_cli, "config_manager", manager),
        patch.object(database_cli.sys, "stdin", stdin),
        pytest.raises(SystemExit) as exc_info,
    ):
        database_cli.db_app(
            [
                "add",
                "analytics",
                "--host",
                "db.example.com",
                "--database",
                "analytics",
                "--username",
                "agent",
                "--no-interactive",
                "--password-stdin",
            ]
        )

    assert exc_info.value.code == 0
    saved_config, saved_password = manager.add_database.call_args.args
    assert saved_config.name == "analytics"
    assert saved_password == "not-a-real-secret"


def test_destructive_command_requires_yes_without_a_terminal(capsys):
    manager = MagicMock()
    manager.get_database.return_value = object()

    with (
        patch.object(database_cli, "config_manager", manager),
        patch.object(database_cli.sys, "stdin", StringIO()),
        pytest.raises(SystemExit) as exc_info,
    ):
        database_cli.remove("analytics")

    assert exc_info.value.code == 2
    assert "saber db remove analytics --yes" in capsys.readouterr().err
    manager.remove_database.assert_not_called()


def test_destructive_command_accepts_yes_without_a_terminal():
    manager = MagicMock()
    manager.get_database.return_value = object()
    manager.remove_database.return_value = True

    with (
        patch.object(database_cli, "config_manager", manager),
        patch.object(database_cli.sys, "stdin", StringIO()),
    ):
        database_cli.remove("analytics", yes=True)

    manager.remove_database.assert_called_once_with("analytics")


def test_invalid_input_uses_stderr_and_usage_exit_code(capsys):
    with (
        patch.object(
            theme_cli.theme_manager,
            "get_available_themes",
            return_value=["dracula", "nord"],
        ),
        pytest.raises(SystemExit) as exc_info,
    ):
        theme_cli.set("not-a-theme")

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "unknown theme 'not-a-theme'" in captured.err
    assert "unknown theme" not in captured.out


def test_threads_prune_dry_run_does_not_delete(capsys):
    store = MagicMock()
    store.count_prunable_threads = AsyncMock(return_value=3)
    store.prune_threads = AsyncMock()

    with patch("sqlsaber.threads.ThreadStorage", return_value=store):
        threads_cli.prune(days=30, dry_run=True)

    assert "3 thread(s)" in capsys.readouterr().out
    store.prune_threads.assert_not_awaited()


def test_root_thread_option_resumes_through_public_sdk():
    store = MagicMock()
    artifact_store = object()
    query_result_store = object()
    captured: dict[str, object] = {}
    lifecycle_events: list[str] = []

    class FakeSQLSaber:
        def __init__(self, *, options):
            captured["options"] = options
            captured["saber"] = self
            self.info = SimpleNamespace(
                primary_database_name="analytics",
                primary_database_type="SQLite",
                model_name="openai:gpt-5",
                model_id=None,
                thread_id="thread-1",
            )
            self.display_registry = {}
            self.query_result_store = query_result_store
            self.query = AsyncMock(
                return_value=SimpleNamespace(usage=None, request_usages=[])
            )

        @classmethod
        async def resume(cls, thread_id, *, options, storage):
            captured["resume"] = (thread_id, options, storage)
            return cls(options=options)

        async def close(self):
            captured["closed"] = True
            lifecycle_events.append("closed")

    class FakeStreamingQueryHandler:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def execute_streaming_query(self, user_query, *, run_query):
            captured["query"] = user_query
            return await run_query(user_query, event_stream_handler=None)

    retention = AsyncMock(
        side_effect=lambda *args: lifecycle_events.append("retention")
    )
    with (
        patch("sqlsaber.threads.ThreadStorage", return_value=store),
        patch("sqlsaber.SQLSaber", FakeSQLSaber),
        patch(
            "sqlsaber.cli.stream_presenter.AgentStreamPresenter",
            FakeStreamingQueryHandler,
        ),
        patch(
            "sqlsaber.cli.artifacts.cli_artifact_store",
            return_value=artifact_store,
        ),
        patch(
            "sqlsaber.cli.query_results.cli_query_result_store",
            return_value=query_result_store,
        ),
        patch("sqlsaber.cli.retention.run_cli_retention", retention),
        patch("sqlsaber.cli.commands.schedule_update_check"),
    ):
        query("Compare this year", thread="thread-1")

    thread_id, options, storage = captured["resume"]
    assert thread_id == "thread-1"
    assert storage is store
    assert options.database is None
    assert options.thread_manager is None
    captured["saber"].query.assert_awaited_once_with(
        captured["query"], event_stream_handler=ANY
    )
    assert captured["query"] == "Compare this year"
    assert captured["closed"] is True
    retention.assert_awaited_once_with(store, artifact_store, query_result_store)
    assert lifecycle_events == ["closed", "retention"]


@pytest.mark.parametrize(
    ("query_text", "stdin_text", "expected_query"),
    [
        ("show revenue", "", "show revenue"),
        (None, "show customers\n", "show customers"),
    ],
)
def test_root_one_shot_modes_construct_public_sdk(
    query_text, stdin_text, expected_query
):
    store = MagicMock()
    artifact_store = object()
    query_result_store = object()
    captured: dict[str, object] = {}

    class FakeSQLSaber:
        def __init__(self, *, options):
            captured["options"] = options
            captured["saber"] = self
            self.info = SimpleNamespace(
                primary_database_name="analytics",
                primary_database_type="SQLite",
                model_name="openai:gpt-5",
                model_id=None,
                thread_id=None,
            )
            self.display_registry = {}
            self.query_result_store = query_result_store
            self.query = AsyncMock(
                return_value=SimpleNamespace(usage=None, request_usages=[])
            )

        async def close(self):
            captured["closed"] = True

    class FakeStreamingQueryHandler:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def execute_streaming_query(self, user_query, *, run_query):
            captured["query"] = user_query
            return await run_query(user_query, event_stream_handler=None)

    retention = AsyncMock()
    with (
        patch("sqlsaber.cli.commands.sys.stdin", StringIO(stdin_text)),
        patch("sqlsaber.threads.ThreadStorage", return_value=store),
        patch("sqlsaber.SQLSaber", FakeSQLSaber),
        patch(
            "sqlsaber.cli.stream_presenter.AgentStreamPresenter",
            FakeStreamingQueryHandler,
        ),
        patch(
            "sqlsaber.cli.artifacts.cli_artifact_store",
            return_value=artifact_store,
        ),
        patch(
            "sqlsaber.cli.query_results.cli_query_result_store",
            return_value=query_result_store,
        ),
        patch("sqlsaber.cli.retention.run_cli_retention", retention),
        patch("sqlsaber.cli.commands.schedule_update_check"),
    ):
        query(query_text, database=["analytics"])

    options = captured["options"]
    assert options.thread_manager.storage is store
    assert captured["query"] == expected_query
    captured["saber"].query.assert_awaited_once_with(
        captured["query"], event_stream_handler=ANY
    )
    assert captured["closed"] is True
    retention.assert_awaited_once_with(store, artifact_store, query_result_store)


def test_root_bare_mode_passes_public_sdk_to_tui():
    store = MagicMock()
    artifact_store = object()
    query_result_store = object()
    captured: dict[str, object] = {}
    stdin = MagicMock()
    stdin.isatty.return_value = True

    class FakeSQLSaber:
        def __init__(self, *, options):
            captured["options"] = options
            captured["saber"] = self
            self.info = SimpleNamespace(
                primary_database_name="analytics",
                primary_database_type="SQLite",
            )

        async def close(self):
            captured["closed"] = True

    class FakeInteractiveSession:
        @classmethod
        def start_unbound_shell(cls, **kwargs):
            captured["shell_kwargs"] = kwargs

            class FakeShell:
                def stop(self):
                    captured["shell_stopped"] = True

            captured["shell"] = FakeShell()
            return captured["shell"]

        def __init__(self, saber):
            captured["interactive_saber"] = saber

        async def run(self, shell=None):
            captured["ran"] = True
            captured["run_shell"] = shell

    retention = AsyncMock()
    with (
        patch("sqlsaber.cli.commands.sys.stdin", stdin),
        patch("sqlsaber.threads.ThreadStorage", return_value=store),
        patch("sqlsaber.SQLSaber", FakeSQLSaber),
        patch("sqlsaber.cli.interactive.InteractiveSession", FakeInteractiveSession),
        patch(
            "sqlsaber.cli.artifacts.cli_artifact_store",
            return_value=artifact_store,
        ),
        patch(
            "sqlsaber.cli.query_results.cli_query_result_store",
            return_value=query_result_store,
        ),
        patch("sqlsaber.cli.retention.run_cli_retention", retention),
        patch("sqlsaber.cli.commands.schedule_update_check"),
    ):
        query(database=["analytics"])

    assert captured["options"].thread_manager.storage is store
    assert captured["interactive_saber"] is captured["saber"]
    assert captured["ran"] is True
    assert captured["closed"] is True
    retention.assert_awaited_once_with(store, artifact_store, query_result_store)


def test_root_thread_option_maps_sdk_resume_errors(capsys):
    from sqlsaber import (
        ThreadDatabaseRequiredError,
        ThreadDatabaseUnavailableError,
        ThreadNotFoundError,
        ThreadResumeHistoryError,
        ThreadResumeMetadataError,
    )

    cases = [
        (
            ThreadNotFoundError("thread-1"),
            "Thread not found: thread-1. List threads with: saber threads list",
        ),
        (
            ThreadDatabaseRequiredError(
                "thread-1",
                "No configured database selector is stored for this thread.",
            ),
            "No database is stored with this thread.",
        ),
        (
            ThreadResumeHistoryError("thread-1", "corrupt snapshot"),
            "Thread history cannot be resumed: corrupt snapshot",
        ),
        (
            ThreadResumeMetadataError("thread-1", "broken metadata"),
            "Invalid thread metadata: broken metadata.",
        ),
        (
            ThreadDatabaseUnavailableError("thread-1", ("analytics",)),
            "The thread database is not configured for automatic continuation.",
        ),
    ]

    for error, expected in cases:

        class FailingSQLSaber:
            @classmethod
            async def resume(cls, thread_id, *, options, storage):
                del cls, thread_id, options, storage
                raise error

        with (
            patch("sqlsaber.threads.ThreadStorage", return_value=MagicMock()),
            patch("sqlsaber.SQLSaber", FailingSQLSaber),
            patch("sqlsaber.cli.artifacts.cli_artifact_store", return_value=object()),
            patch(
                "sqlsaber.cli.query_results.cli_query_result_store",
                return_value=object(),
            ),
            patch("sqlsaber.cli.commands.schedule_update_check"),
            pytest.raises(SystemExit) as exc_info,
        ):
            query("continue", thread="thread-1")

        assert exc_info.value.code == 1
        assert expected in capsys.readouterr().err
