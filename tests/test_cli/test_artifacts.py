from __future__ import annotations

import asyncio
from io import StringIO
from unittest.mock import patch

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    InMemoryArtifactStore,
)
from sqlsaber.cli.artifacts import cli_artifact_store
from rich.console import Console

from sqlsaber.cli.display import DisplayManager
from sqlsaber.cli.html_export import render_thread_html
from sqlsaber.cli.threads import create_threads_app
from sqlsaber.threads.storage import Thread, ThreadStorage


def test_live_tool_result_lists_durable_artifact_uri() -> None:
    publication = asyncio.run(
        InMemoryArtifactStore().publish(
            ArtifactBundle(
                kind="report",
                artifacts=(Artifact("report.csv", b"data", "text/csv"),),
            ),
            context=ArtifactContext(),
        )
    )
    output = StringIO()
    display = DisplayManager(Console(file=output, force_terminal=False, width=200))

    display.show_tool_result(
        "create_report",
        "done",
        metadata=publication.to_metadata(),
    )

    rendered = output.getvalue()
    assert "report.csv" in rendered
    assert publication.artifacts[0].uri in rendered


def test_html_export_links_artifacts_without_embedding_bytes() -> None:
    publication = asyncio.run(
        InMemoryArtifactStore().publish(
            ArtifactBundle(
                kind="report",
                artifacts=(Artifact("report.csv", b"secret-data", "text/csv"),),
            ),
            context=ArtifactContext(),
        )
    )
    messages = [
        ModelRequest(parts=[UserPromptPart("Create a report")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "create_report",
                    "done",
                    metadata=publication.to_metadata(),
                )
            ]
        ),
    ]
    thread = Thread(
        id="thread-1",
        database_name="test",
        title="Report",
        created_at=1.0,
        ended_at=None,
        last_activity_at=1.0,
        model_name="test:model",
    )

    html = render_thread_html(thread, messages)

    assert "report.csv" in html
    assert publication.artifacts[0].uri in html
    assert "secret-data" not in html


def test_threads_artifacts_lists_retained_publication(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda app_name: str(tmp_path / app_name),
    )
    store = cli_artifact_store()
    publication = asyncio.run(
        store.publish(
            ArtifactBundle(
                kind="report",
                artifacts=(Artifact("report.csv", b"a,b\n", "text/csv"),),
            ),
            context=ArtifactContext(conversation_id="conversation-1"),
        )
    )
    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "create_report",
                    "done",
                    metadata=publication.to_metadata(),
                )
            ]
        )
    ]
    thread_storage = ThreadStorage()
    thread_storage.db_path = tmp_path / "threads.db"
    thread_id = asyncio.run(
        thread_storage.save_snapshot(
            messages_json=ModelMessagesTypeAdapter.dump_json(messages),
            database_name="test",
        )
    )

    with (
        patch("sqlsaber.threads.ThreadStorage", return_value=thread_storage),
        pytest.raises(SystemExit) as exit_info,
    ):
        create_threads_app()(["artifacts", thread_id])

    assert exit_info.value.code == 0

    output = capsys.readouterr().out
    assert publication.id in output
    assert "report.csv" in output
    assert "file://" in output


async def test_cli_artifact_store_persists_under_user_data_dir(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda app_name: str(tmp_path / app_name),
    )
    context = ArtifactContext(conversation_id="conversation-1")
    publication = await cli_artifact_store().publish(
        ArtifactBundle(
            kind="report",
            artifacts=(Artifact("report.csv", b"a,b\n", "text/csv"),),
        ),
        context=context,
    )

    reconstructed = cli_artifact_store()
    loaded = await reconstructed.get(publication.artifacts[0].id, context=context)

    assert reconstructed.root == (tmp_path / "sqlsaber" / "artifacts").absolute()
    assert loaded.data == b"a,b\n"
