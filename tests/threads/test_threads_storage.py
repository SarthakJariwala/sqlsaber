"""Tests for ThreadStorage (pydantic-ai snapshot threads)."""

import tempfile
from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from sqlsaber.threads.storage import ThreadStorage


def _messages_bytes(user_text: str, *assistant_texts: str) -> bytes:
    msgs = [ModelRequest(parts=[UserPromptPart(user_text)])]
    for t in assistant_texts:
        msgs.append(ModelResponse(parts=[TextPart(t)]))
    return ModelMessagesTypeAdapter.dump_json(msgs)


@pytest.mark.asyncio
async def test_thread_create_and_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ThreadStorage()
        store.db_path = Path(tmp) / "threads.db"

        # Create snapshot with a simple 2-message conversation
        b1 = _messages_bytes("Hello", "Hi there!")
        thread_id = await store.save_snapshot(messages_json=b1, database_name="db1")
        assert thread_id
        await store.save_metadata(thread_id=thread_id, title="Hello")

        # Read back metadata and messages
        t = await store.get_thread(thread_id)
        assert t is not None
        assert t.database_name == "db1"
        assert t.title == "Hello"
        msgs = await store.get_thread_messages(thread_id)
        assert len(msgs) == 2

        # Update the snapshot by appending another assistant message
        b2 = _messages_bytes("Hello", "Hi there!", "How can I help?")
        await store.save_snapshot(
            messages_json=b2, database_name="db1", thread_id=thread_id
        )
        msgs2 = await store.get_thread_messages(thread_id)
        assert len(msgs2) == 3
        # Verify user prompt remains the same
        assert isinstance(msgs2[0], ModelRequest)


@pytest.mark.asyncio
async def test_list_end_delete_threads():
    with tempfile.TemporaryDirectory() as tmp:
        store = ThreadStorage()
        store.db_path = Path(tmp) / "threads.db"

        # Create two threads for different databases
        t1 = await store.save_snapshot(
            messages_json=_messages_bytes("A", "B"), database_name="db1"
        )
        t2 = await store.save_snapshot(
            messages_json=_messages_bytes("C", "D"), database_name="db2"
        )
        assert t1 != t2

        # List all
        all_threads = await store.list_threads()
        assert len(all_threads) == 2

        # Filter by db
        db1_threads = await store.list_threads(database_name="db1")
        assert len(db1_threads) == 1

        # End first thread
        ok = await store.end_thread(t1)
        assert ok
        t1_meta = await store.get_thread(t1)
        assert t1_meta is not None and t1_meta.ended_at is not None

        # Delete second thread
        deleted = await store.delete_thread(t2)
        assert deleted
        remaining = await store.list_threads()
        assert len(remaining) == 1
