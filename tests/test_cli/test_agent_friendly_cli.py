"""Coverage for non-interactive, agent-friendly CLI behavior."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

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
        patch.object(
            theme_cli.questionary,
            "select",
            side_effect=AssertionError("direct theme selection must not prompt"),
        ),
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


def test_root_thread_option_continues_with_saved_history():
    history = [object(), object()]
    stored_thread = SimpleNamespace(
        id="thread-1",
        database_name="analytics",
        extra_metadata=None,
    )
    store = MagicMock()
    store.get_thread = AsyncMock(return_value=stored_thread)
    store.get_thread_messages = AsyncMock(return_value=history)
    captured: dict[str, object] = {}

    class FakeThreadManager:
        def __init__(self, initial_thread_id=None, storage=None):
            captured["initial_thread_id"] = initial_thread_id
            captured["storage"] = storage
            self.current_thread_id = initial_thread_id

    class FakeDatabaseConfigManager:
        def get_database(self, name):
            return object() if name == "analytics" else None

    class FakeSession:
        def __init__(self, options):
            captured["options"] = options
            self.db_name = "analytics"
            self.connection = object()
            self.query_result_store = object()
            self.query = AsyncMock()
            self.agent = SimpleNamespace(
                display_registry={},
                db_type="sqlite",
                config=SimpleNamespace(model=SimpleNamespace(name="openai:gpt-5")),
            )

        async def close(self):
            captured["closed"] = True

    class FakeStreamingQueryHandler:
        def __init__(self, *args):
            pass

        async def execute_streaming_query(
            self, user_query, *, run_query, message_history
        ):
            captured["query"] = user_query
            captured["run_query"] = run_query
            captured["history"] = message_history
            return None

    with (
        patch("sqlsaber.threads.ThreadStorage", return_value=store),
        patch("sqlsaber.threads.manager.ThreadManager", FakeThreadManager),
        patch(
            "sqlsaber.config.database.DatabaseConfigManager",
            FakeDatabaseConfigManager,
        ),
        patch("sqlsaber.session.SQLSaberSession", FakeSession),
        patch(
            "sqlsaber.cli.streaming.StreamingQueryHandler",
            FakeStreamingQueryHandler,
        ),
        patch("sqlsaber.cli.artifacts.cli_artifact_store", return_value=object()),
        patch(
            "sqlsaber.cli.query_results.cli_query_result_store", return_value=object()
        ),
        patch("sqlsaber.cli.commands.needs_onboarding", return_value=False),
        patch("sqlsaber.cli.commands.schedule_update_check"),
    ):
        query("Compare this year", thread="thread-1")

    options = captured["options"]
    assert options.database == "analytics"
    assert options.thread_manager.current_thread_id == "thread-1"
    assert captured["initial_thread_id"] == "thread-1"
    assert captured["storage"] is store
    assert captured["history"] is history
    assert captured["query"] == "Compare this year"
    assert captured["closed"] is True
