from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from sqlsaber.config.settings import Config
from sqlsaber.threads.manager import ThreadManager
from sqlsaber.threads.storage import ThreadStorage


def _settings() -> Config:
    return Config.in_memory(
        model_name="anthropic:claude-3-5-sonnet",
        api_keys={"anthropic": "test-key"},
    )


def _sqlite_uri(path: Path) -> str:
    return f"sqlite:///{path}"


def _messages_bytes(user_text: str, assistant_text: str) -> bytes:
    return ModelMessagesTypeAdapter.dump_json(
        [
            ModelRequest(parts=[UserPromptPart(user_text)]),
            ModelResponse(parts=[TextPart(assistant_text)]),
        ]
    )


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


@pytest.mark.asyncio
async def test_multi_database_api_exposes_connections_and_rejects_single_connection(
    temp_dir,
):
    a = temp_dir / "a.db"
    b = temp_dir / "b.db"

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[_sqlite_uri(a), _sqlite_uri(b)],
            settings=_settings(),
        )
    )

    try:
        assert saber.db_name == "a + b"
        assert set(saber.connections or {}) == {"a", "b"}
        with pytest.raises(
            RuntimeError, match=r"multiple database connections.*connections"
        ):
            _ = saber.connection
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_all_csv_list_stays_single_connection(temp_dir):
    users = temp_dir / "users.csv"
    orders = temp_dir / "orders.csv"
    users.write_text("id,name\n1,Alice\n", encoding="utf-8")
    orders.write_text("id,user_id,total\n10,1,9.99\n", encoding="utf-8")

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[str(users), str(orders)],
            settings=_settings(),
        )
    )

    try:
        assert saber.connections is None
        rows = await saber.connection.execute_query(
            'SELECT u.name, o.total FROM "users" u JOIN "orders" o ON u.id = o.user_id'
        )
        assert rows == [{"name": "Alice", "total": 9.99}]
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_multi_session_precreates_threads_and_saves_parent_run(
    temp_dir, monkeypatch
):
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_manager = ThreadManager(storage=storage)
    a = temp_dir / "a.db"
    b = temp_dir / "b.db"

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[_sqlite_uri(a), _sqlite_uri(b)],
            settings=_settings(),
            thread_manager=thread_manager,
        )
    )

    async def fake_run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        assert prompt == "compare counts"
        assert kwargs["message_history"] is None
        assert kwargs["event_stream_handler"] is None
        return FakeRunResult(
            output="combined answer",
            _messages_json=_messages_bytes("compare counts", "combined answer"),
        )

    monkeypatch.setattr(
        "sqlsaber.agents.multi_database_agent.MultiDatabaseCoordinator.run",
        fake_run,
    )

    try:
        result = await saber.query("compare counts")
        assert str(result) == "combined answer"

        parent_id = thread_manager.current_thread_id
        assert parent_id is not None
        parent = await storage.get_thread(parent_id)
        assert parent is not None
        assert parent.database_name == "a + b"
        assert parent.title == "compare counts"
        parent_metadata = json.loads(parent.extra_metadata or "{}")
        assert parent_metadata["kind"] == "multi_database_parent"
        assert {item["database_id"] for item in parent_metadata["child_threads"]} == {
            "a",
            "b",
        }

        child_ids = {
            item["database_id"]: item["thread_id"]
            for item in parent_metadata["child_threads"]
        }
        assert all(child_ids.values())

        for database_id, child_id in child_ids.items():
            child = await storage.get_thread(child_id)
            assert child is not None
            assert child.title == f"[{database_id}] compare counts"
            child_metadata = json.loads(child.extra_metadata or "{}")
            assert child_metadata == {
                "kind": "multi_database_child",
                "parent_thread_id": parent_id,
                "database_id": database_id,
            }
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_multi_session_close_ends_parent_and_child_threads(temp_dir, monkeypatch):
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_manager = ThreadManager(storage=storage)
    a = temp_dir / "a.db"
    b = temp_dir / "b.db"

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[_sqlite_uri(a), _sqlite_uri(b)],
            settings=_settings(),
            thread_manager=thread_manager,
        )
    )

    async def fake_run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        _ = self, prompt, kwargs
        return FakeRunResult(
            output="combined answer",
            _messages_json=_messages_bytes("compare counts", "combined answer"),
        )

    monkeypatch.setattr(
        "sqlsaber.agents.multi_database_agent.MultiDatabaseCoordinator.run",
        fake_run,
    )

    await saber.query("compare counts")
    parent_id = thread_manager.current_thread_id
    assert parent_id is not None
    parent = await storage.get_thread(parent_id)
    assert parent is not None
    parent_metadata = json.loads(parent.extra_metadata or "{}")
    child_ids = [item["thread_id"] for item in parent_metadata["child_threads"]]
    assert all(child_ids)

    await saber.close()

    ended_parent = await storage.get_thread(parent_id)
    assert ended_parent is not None
    assert ended_parent.ended_at is not None

    for child_id in child_ids:
        child = await storage.get_thread(child_id)
        assert child is not None
        assert child.ended_at is not None


@pytest.mark.asyncio
async def test_multi_session_followup_reuses_parent_and_child_threads(
    temp_dir, monkeypatch
):
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_manager = ThreadManager(storage=storage)
    a = temp_dir / "a.db"
    b = temp_dir / "b.db"

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[_sqlite_uri(a), _sqlite_uri(b)],
            settings=_settings(),
            thread_manager=thread_manager,
        )
    )

    payloads = iter(
        [
            FakeRunResult("first answer", _messages_bytes("first", "first answer")),
            FakeRunResult("second answer", _messages_bytes("second", "second answer")),
        ]
    )

    async def fake_run(self, prompt: str, **kwargs: Any) -> FakeRunResult:
        _ = self, prompt, kwargs
        return next(payloads)

    monkeypatch.setattr(
        "sqlsaber.agents.multi_database_agent.MultiDatabaseCoordinator.run",
        fake_run,
    )

    try:
        await saber.query("first")
        parent_id = thread_manager.current_thread_id
        assert parent_id is not None
        parent = await storage.get_thread(parent_id)
        assert parent is not None
        initial_metadata = json.loads(parent.extra_metadata or "{}")
        child_ids = {
            item["database_id"]: item["thread_id"]
            for item in initial_metadata["child_threads"]
        }

        await saber.query("second")
        assert thread_manager.current_thread_id == parent_id
        updated_parent = await storage.get_thread(parent_id)
        assert updated_parent is not None
        assert updated_parent.title == "first"
        updated_metadata = json.loads(updated_parent.extra_metadata or "{}")
        updated_child_ids = {
            item["database_id"]: item["thread_id"]
            for item in updated_metadata["child_threads"]
        }
        assert updated_child_ids == child_ids
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_repeated_non_csv_databases_use_multi_session(temp_dir):
    a = temp_dir / "a.db"
    b = temp_dir / "b.db"

    saber = SQLSaber(
        options=SQLSaberOptions(
            database=[_sqlite_uri(a), _sqlite_uri(b)],
            settings=_settings(),
        )
    )

    try:
        assert saber.connections is not None
        assert set(saber.connections) == {"a", "b"}
        with pytest.raises(RuntimeError):
            _ = saber.connection
    finally:
        await saber.close()
