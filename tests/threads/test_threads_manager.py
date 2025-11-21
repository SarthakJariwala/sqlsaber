import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from sqlsaber.threads.manager import ThreadManager
from sqlsaber.threads.storage import ThreadStorage


def _messages_bytes(user_text: str, assistant_text: str | None = None) -> bytes:
    parts = [UserPromptPart(user_text)]
    msgs = [ModelRequest(parts=parts)]
    if assistant_text:
        msgs.append(ModelResponse(parts=[TextPart(assistant_text)]))
    return ModelMessagesTypeAdapter.dump_json(msgs)


@pytest.fixture
def temp_storage():
    """Fixture for a real ThreadStorage backed by a temp file."""
    with tempfile.TemporaryDirectory() as tmp:
        store = ThreadStorage()
        store.db_path = Path(tmp) / "threads.db"
        yield store


@pytest.fixture
def thread_manager(temp_storage):
    """Fixture for ThreadManager using the temp storage."""
    return ThreadManager(storage=temp_storage)


@pytest.mark.asyncio
async def test_init_state(thread_manager):
    """Test initial state of ThreadManager."""
    assert thread_manager.current_thread_id is None
    assert thread_manager.first_message is True


@pytest.mark.asyncio
async def test_init_with_id(temp_storage):
    """Test initialization with an existing thread ID."""
    # Pre-create a thread in storage so it's "real"
    tm = ThreadManager(initial_thread_id="existing-id", storage=temp_storage)
    assert tm.current_thread_id == "existing-id"
    assert tm.first_message is False


@pytest.mark.asyncio
async def test_end_current_thread_active(thread_manager, temp_storage):
    """Test ending an active thread."""
    # Create a thread first
    messages_json = b"[]"
    thread_id = await temp_storage.save_snapshot(
        messages_json=messages_json, database_name="db1"
    )

    thread_manager.current_thread_id = thread_id

    result = await thread_manager.end_current_thread()

    assert result == thread_id

    # Verify in storage
    t = await temp_storage.get_thread(thread_id)
    assert t.ended_at is not None


@pytest.mark.asyncio
async def test_end_current_thread_none(thread_manager):
    """Test ending when no thread is active."""
    result = await thread_manager.end_current_thread()
    assert result is None


@pytest.mark.asyncio
async def test_clear_current_thread(thread_manager, temp_storage):
    """Test clearing the current thread."""
    # Create a thread first
    messages_json = b"[]"
    thread_id = await temp_storage.save_snapshot(
        messages_json=messages_json, database_name="db1"
    )

    thread_manager.current_thread_id = thread_id
    thread_manager.first_message = False

    await thread_manager.clear_current_thread()

    assert thread_manager.current_thread_id is None
    assert thread_manager.first_message is True

    # Check it was ended in storage
    t = await temp_storage.get_thread(thread_id)
    assert t.ended_at is not None


@pytest.mark.asyncio
async def test_save_run_new_thread(thread_manager, temp_storage):
    """Test saving a run for a new thread (first message)."""
    # Setup valid JSON for run_result
    json_bytes = _messages_bytes("hello")

    # Setup mocks for run_result
    run_result = MagicMock()
    run_result.all_messages_json.return_value = json_bytes
    run_result.all_messages.return_value = ["msg1"]

    # execute
    result = await thread_manager.save_run(
        run_result=run_result,
        database_name="db1",
        user_query="hello",
        model_name="gpt-4",
    )

    # Assertions
    assert result == ["msg1"]
    assert thread_manager.current_thread_id is not None
    assert thread_manager.first_message is False

    thread_id = thread_manager.current_thread_id

    # Verify in storage
    t = await temp_storage.get_thread(thread_id)
    assert t.database_name == "db1"
    assert t.title == "hello"
    assert t.model_name == "gpt-4"

    # Verify messages can be read back
    msgs = await temp_storage.get_thread_messages(thread_id)
    assert len(msgs) == 1


@pytest.mark.asyncio
async def test_save_run_existing_thread(thread_manager, temp_storage):
    """Test saving a run for an existing thread."""
    # Create initial thread
    thread_id = await temp_storage.save_snapshot(
        messages_json=_messages_bytes("original"), database_name="db1"
    )
    await temp_storage.save_metadata(
        thread_id=thread_id, title="original", model_name="gpt-4"
    )

    # Setup state
    thread_manager.current_thread_id = thread_id
    thread_manager.first_message = False

    # Setup valid JSON for updated run
    updated_bytes = _messages_bytes("original", "response")

    # Setup mocks
    run_result = MagicMock()
    run_result.all_messages_json.return_value = updated_bytes
    run_result.all_messages.return_value = ["msg1", "msg2"]

    # execute
    result = await thread_manager.save_run(
        run_result=run_result,
        database_name="db1",
        user_query="continuation",
        model_name="gpt-4",
    )

    # Assertions
    assert result == ["msg1", "msg2"]
    assert thread_manager.current_thread_id == thread_id

    # Verify messages updated in storage
    msgs = await temp_storage.get_thread_messages(thread_id)
    assert len(msgs) == 2

    # Verify title didn't change because first_message was False
    t = await temp_storage.get_thread(thread_id)
    assert t.title == "original"
