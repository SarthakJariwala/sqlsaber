from __future__ import annotations

import json
import sqlite3

import pytest
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart

from sqlsaber.artifacts import (
    Artifact,
    ArtifactBundle,
    ArtifactContext,
    ArtifactUnavailable,
    FilesystemArtifactStore,
)
from sqlsaber.cli.artifact_gc import collect_cli_artifacts
from sqlsaber.threads.storage import ThreadStorage


async def test_gc_aborts_when_any_thread_snapshot_is_unreadable(tmp_path) -> None:
    now = 2_000_000.0
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    publication = await store.publish(
        ArtifactBundle(
            kind="report",
            artifacts=(Artifact("orphan.csv", b"orphan", "text/csv"),),
        ),
        context=ArtifactContext(),
    )
    manifest_path = store.root / publication.id[3:5] / publication.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = now - 100
    manifest_path.write_text(json.dumps(manifest))

    threads = ThreadStorage()
    threads.db_path = tmp_path / "threads.db"
    await threads.save_snapshot(messages_json=b"[]", database_name="test")
    with sqlite3.connect(threads.db_path) as db:
        db.execute("UPDATE threads SET messages_json = ?", (b"not-json",))
        db.commit()

    result = await collect_cli_artifacts(
        threads,
        store,
        force=True,
        grace_seconds=50,
        now=now,
    )

    assert result.complete is False
    assert result.deleted == 0
    loaded = await store.get(
        publication.artifacts[0].id,
        context=ArtifactContext(),
    )
    assert loaded.data == b"orphan"


async def test_gc_preserves_referenced_publication_and_deletes_old_orphan(
    tmp_path,
) -> None:
    now = 2_000_000.0
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    context = ArtifactContext(conversation_id="conversation-1")
    live = await store.publish(
        ArtifactBundle(
            kind="report",
            artifacts=(Artifact("live.csv", b"live", "text/csv"),),
        ),
        context=context,
    )
    orphan = await store.publish(
        ArtifactBundle(
            kind="report",
            artifacts=(Artifact("orphan.csv", b"orphan", "text/csv"),),
        ),
        context=context,
    )
    for publication in (live, orphan):
        manifest_path = (
            store.root / publication.id[3:5] / publication.id / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text())
        manifest["created_at"] = now - 100
        manifest_path.write_text(json.dumps(manifest))

    messages = [
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "create_report",
                    "done",
                    metadata=live.to_metadata(),
                )
            ]
        )
    ]
    threads = ThreadStorage()
    threads.db_path = tmp_path / "threads.db"
    await threads.save_snapshot(
        messages_json=ModelMessagesTypeAdapter.dump_json(messages),
        database_name="test",
    )

    result = await collect_cli_artifacts(
        threads,
        store,
        force=True,
        grace_seconds=50,
        now=now,
    )

    assert result.deleted == 1
    assert await store.get(live.artifacts[0].id, context=context)
    with pytest.raises(ArtifactUnavailable):
        await store.get(orphan.artifacts[0].id, context=context)
