from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.threads.manager import ThreadManager
from sqlsaber.threads.storage import ThreadStorage


def _messages_bytes(user_text: str, *assistant_texts: str) -> bytes:
    messages = [ModelRequest(parts=[UserPromptPart(user_text)])]
    for text in assistant_texts:
        messages.append(ModelResponse(parts=[TextPart(text)]))
    return ModelMessagesTypeAdapter.dump_json(messages)


@dataclass
class FakeRunResult:
    output: str
    _messages_json: bytes

    def usage(self) -> None:
        return None

    def new_messages(self) -> list[ModelMessage]:
        return ModelMessagesTypeAdapter.validate_json(self._messages_json)

    def all_messages(self) -> list[ModelMessage]:
        return ModelMessagesTypeAdapter.validate_json(self._messages_json)

    def all_messages_json(self) -> bytes:
        return self._messages_json


@dataclass
class OutputOnlyRunResult:
    output: str

    def usage(self) -> None:
        return None

    def new_messages(self) -> list[ModelMessage]:
        return []

    def all_messages(self) -> list[ModelMessage]:
        return []


@pytest.mark.asyncio
async def test_api_thread_manager_persists_queries_and_ends_on_close(
    temp_dir, monkeypatch
):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_manager = ThreadManager(storage=storage)
    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        thread_manager=thread_manager,
    )
    saber = SQLSaber(options=options)

    payloads = iter(
        [
            _messages_bytes("first", "first-response"),
            _messages_bytes("first", "first-response", "second-response"),
        ]
    )

    async def fake_run(prompt: str, **kwargs):
        _ = prompt, kwargs
        return FakeRunResult(output="ok", _messages_json=next(payloads))

    monkeypatch.setattr(saber.agent, "run", fake_run)

    thread_id: str | None = None
    try:
        await saber.query("first")
        thread_id = thread_manager.current_thread_id
        assert thread_id is not None

        metadata = await storage.get_thread(thread_id)
        assert metadata is not None
        assert metadata.database_name == saber.db_name
        assert metadata.title == "first"
        assert metadata.model_name is not None

        await saber.query("second")
        assert thread_manager.current_thread_id == thread_id

        messages = await storage.get_thread_messages(thread_id)
        assert len(messages) == 3
    finally:
        await saber.close()

    assert thread_id is not None
    ended = await storage.get_thread(thread_id)
    assert ended is not None
    assert ended.ended_at is not None


@pytest.mark.asyncio
async def test_api_without_thread_manager_does_not_require_run_snapshot_methods(
    temp_dir, monkeypatch
):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    saber = SQLSaber(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
    )

    async def fake_run(prompt: str, **kwargs):
        _ = prompt, kwargs
        return OutputOnlyRunResult(output="ok")

    monkeypatch.setattr(saber.agent, "run", fake_run)

    try:
        result = await saber.query("hello")
        assert str(result) == "ok"
    finally:
        await saber.close()


class _FailingThreadManager:
    def __init__(self) -> None:
        self.save_calls = 0
        self.end_calls = 0

    async def save_run(self, **kwargs) -> list[ModelMessage]:
        _ = kwargs
        self.save_calls += 1
        raise RuntimeError("save failed")

    async def end_current_thread(self) -> None:
        self.end_calls += 1


@pytest.mark.asyncio
async def test_api_thread_persistence_failures_do_not_fail_query(temp_dir, monkeypatch):
    config_dir = temp_dir / "config"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )

    failing_manager = _FailingThreadManager()
    options = SQLSaberOptions(
        database="sqlite:///:memory:",
        model_name="anthropic:claude-3-5-sonnet",
        api_key="test-key",
        thread_manager=failing_manager,
    )
    saber = SQLSaber(options=options)

    async def fake_run(prompt: str, **kwargs):
        _ = prompt, kwargs
        return OutputOnlyRunResult(output="ok")

    monkeypatch.setattr(saber.agent, "run", fake_run)

    result = await saber.query("hello")
    assert str(result) == "ok"
    assert failing_manager.save_calls == 1

    await saber.close()
    assert failing_manager.end_calls == 1
