from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, dataclass
from typing import Any

import aiosqlite
import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from sqlsaber import (
    RunInProgressError,
    SQLSaber,
    SQLSaberInfo,
    SQLSaberOptions,
    SQLSaberResult,
    ThinkingLevel,
    ThinkingState,
    ThreadDatabaseRequiredError,
    ThreadDatabaseUnavailableError,
    ThreadNotFoundError,
    ThreadResumeHistoryError,
)
from sqlsaber.config.database import DatabaseConfig, DatabaseConfigManager
from sqlsaber.config.settings import Config
from sqlsaber.threads.metadata import (
    encode_thread_extra_metadata,
    encode_thread_resume_disabled_metadata,
)
from sqlsaber.threads.storage import ThreadStorage


def _options(**overrides: Any) -> SQLSaberOptions:
    values: dict[str, Any] = {
        "database": "sqlite:///:memory:",
        "settings": Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
    }
    values.update(overrides)
    return SQLSaberOptions(**values)


def _turn(
    prompt: str,
    answer: str,
    *,
    input_tokens: int = 10,
) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(
            parts=[TextPart(answer)],
            usage=RequestUsage(input_tokens=input_tokens, output_tokens=2),
        ),
    ]


@dataclass
class _RunResult:
    output: str
    created_messages: list[ModelMessage]
    history: list[ModelMessage]

    @property
    def response(self) -> ModelResponse:
        response = self.created_messages[-1]
        assert isinstance(response, ModelResponse)
        return response

    def usage(self) -> RunUsage:
        return RunUsage(input_tokens=20, output_tokens=4, requests=2)

    def new_messages(self) -> list[ModelMessage]:
        return list(self.created_messages)

    def all_messages(self) -> list[ModelMessage]:
        return list(self.history)

    def all_messages_json(self) -> bytes:
        return ModelMessagesTypeAdapter.dump_json(self.history)


@pytest.mark.asyncio
async def test_query_owns_history_and_preserves_explicit_history_compatibility(
    monkeypatch,
) -> None:
    saber = SQLSaber(options=_options())
    received_histories: list[list[ModelMessage]] = []

    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        history = list(kwargs["message_history"])
        received_histories.append(history)
        created = _turn(prompt, f"answer-{len(received_histories)}")
        return _RunResult(
            output=f"answer-{len(received_histories)}",
            created_messages=created,
            history=[*history, *created],
        )

    monkeypatch.setattr(saber.agent, "run", fake_run)
    explicit_history = _turn("embedded", "embedded-answer")

    try:
        first = await saber.query("first")
        second = await saber.query("second")
        explicit = await saber.query("third", message_history=explicit_history)
        await saber.query("fourth")
    finally:
        await saber.close()

    assert received_histories[0] == []
    assert received_histories[1] == first.all_messages
    assert received_histories[2] == explicit_history
    assert received_histories[3] == explicit.all_messages
    assert second.text == "answer-2"


@pytest.mark.asyncio
async def test_result_exposes_complete_run_data(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    first_response = ModelResponse(
        parts=[TextPart("working")],
        usage=RequestUsage(input_tokens=11, output_tokens=3),
    )
    final_response = ModelResponse(
        parts=[TextPart("done")],
        usage=RequestUsage(input_tokens=17, output_tokens=4),
    )
    created: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("question")]),
        first_response,
        final_response,
    ]

    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        del prompt, kwargs
        return _RunResult(output="done", created_messages=created, history=created)

    monkeypatch.setattr(saber.agent, "run", fake_run)
    try:
        result = await saber.query("question")
    finally:
        await saber.close()

    assert isinstance(result, SQLSaberResult)
    assert result.text == "done"
    assert result.usage == RunUsage(input_tokens=20, output_tokens=4, requests=2)
    assert result.request_usages == [first_response.usage, final_response.usage]
    assert result.final_context_tokens == 17
    assert result.new_messages == created
    assert result.all_messages == created
    assert result.query_results == []
    assert result.artifacts == []


@pytest.mark.asyncio
async def test_close_rejects_an_active_query(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    started = asyncio.Event()
    release = asyncio.Event()
    created = _turn("question", "answer")

    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        del prompt, kwargs
        started.set()
        await release.wait()
        return _RunResult(output="answer", created_messages=created, history=created)

    monkeypatch.setattr(saber.agent, "run", fake_run)
    query_task = asyncio.create_task(saber.query("question"))
    await started.wait()

    try:
        with pytest.raises(RunInProgressError):
            await saber.close()
        release.set()
        assert await query_task == "answer"
    finally:
        release.set()
        if not query_task.done():
            await query_task
        await saber.close()


@pytest.mark.asyncio
async def test_info_and_thinking_controls_are_public_and_immutable() -> None:
    saber = SQLSaber(options=_options(allow_dangerous=True))
    try:
        initial = saber.info
        assert isinstance(initial, SQLSaberInfo)
        assert initial.database_names == (":memory:",)
        assert initial.primary_database_name == ":memory:"
        assert initial.primary_database_type == "SQLite"
        assert initial.model_name == "claude-3-5-sonnet"
        assert initial.model_id == "anthropic:claude-3-5-sonnet"
        assert initial.dangerous_mode is True
        assert initial.thread_id is None
        assert initial.is_new_thread is True
        assert saber.artifact_store is None
        assert saber.query_result_store is not None
        assert "execute_sql" in saber.display_registry

        thinking = saber.set_thinking(enabled=True, level=ThinkingLevel.HIGH)
        assert thinking == ThinkingState(enabled=True, level=ThinkingLevel.HIGH)
        assert saber.info.thinking == thinking
        with pytest.raises(FrozenInstanceError):
            setattr(thinking, "enabled", False)
    finally:
        await saber.close()


@pytest.mark.asyncio
async def test_list_tables_returns_completion_metadata(temp_dir) -> None:
    saber = SQLSaber(
        options=_options(database=f"sqlite:///{temp_dir / 'tables.sqlite'}")
    )
    try:
        await saber.connection.execute_query(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY)",
            commit=True,
        )
        tables = await saber.list_tables()
    finally:
        await saber.close()

    customers = next(table for table in tables if table.name == "customers")
    assert customers.database_name == "tables"
    assert customers.schema_name == "main"
    assert customers.qualified_name == "main.customers"
    assert customers.completion_name == "main.customers"


@pytest.mark.asyncio
async def test_handoff_uses_sdk_owned_history(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    created = _turn("question", "answer")
    captured: dict[str, object] = {}

    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        del prompt, kwargs
        return _RunResult(output="answer", created_messages=created, history=created)

    class FakeHandoffAgent:
        async def generate_draft(
            self,
            message_history: list[ModelMessage],
            goal: str,
        ) -> str:
            captured["history"] = message_history
            captured["goal"] = goal
            return "draft"

    monkeypatch.setattr(saber.agent, "run", fake_run)
    monkeypatch.setattr(
        "sqlsaber.agents.handoff_agent.HandoffAgent",
        FakeHandoffAgent,
    )

    try:
        await saber.query("question")
        draft = await saber.draft_handoff("continue elsewhere")
    finally:
        await saber.close()

    assert draft == "draft"
    assert captured == {"history": created, "goal": "continue elsewhere"}


def _configure_database(temp_dir, monkeypatch, name: str = "analytics") -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir",
        lambda *args, **kwargs: str(temp_dir / "config"),
    )
    DatabaseConfigManager().add_database(
        DatabaseConfig(
            name=name,
            type="sqlite",
            host=None,
            port=None,
            database=str(temp_dir / f"{name}.sqlite"),
            username=None,
        )
    )


async def _stored_thread(
    storage: ThreadStorage,
    *,
    database_name: str,
    extra_metadata: str,
) -> tuple[str, list[ModelMessage]]:
    history = _turn("stored question", "stored answer")
    thread_id = await storage.save_snapshot(
        messages_json=ModelMessagesTypeAdapter.dump_json(history),
        database_name=database_name,
        extra_metadata=extra_metadata,
    )
    return thread_id, history


@pytest.mark.asyncio
async def test_resume_loads_history_and_owns_thread_controls(
    temp_dir, monkeypatch
) -> None:
    _configure_database(temp_dir, monkeypatch)
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_id, stored_history = await _stored_thread(
        storage,
        database_name="analytics",
        extra_metadata=encode_thread_extra_metadata(database_selector="analytics"),
    )
    saber = await SQLSaber.resume(
        thread_id,
        options=_options(database=None),
        storage=storage,
    )
    received_history: list[ModelMessage] = []

    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        del prompt
        received_history.extend(kwargs["message_history"])
        created = _turn("follow up", "continued")
        return _RunResult(
            output="continued",
            created_messages=created,
            history=[*received_history, *created],
        )

    monkeypatch.setattr(saber.agent, "run", fake_run)
    try:
        assert saber.info.thread_id == thread_id
        assert saber.info.is_new_thread is False
        assert saber.info.primary_database_name == "analytics"
        await saber.query("follow up")
        ended_id = await saber.end_thread()
        previous_id = await saber.new_thread()
        assert saber.info.thread_id is None
        assert saber.info.is_new_thread is True
    finally:
        await saber.close()

    assert received_history == stored_history
    assert ended_id == thread_id
    assert previous_id == thread_id
    stored = await storage.get_thread(thread_id)
    assert stored is not None
    assert stored.ended_at is not None


@pytest.mark.asyncio
async def test_resume_raises_typed_errors_for_missing_or_unsafe_threads(
    temp_dir,
) -> None:
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"

    with pytest.raises(ThreadNotFoundError):
        await SQLSaber.resume(
            "missing",
            options=_options(database=None),
            storage=storage,
        )

    unsafe_id, _ = await _stored_thread(
        storage,
        database_name="database",
        extra_metadata=encode_thread_resume_disabled_metadata(
            reason="A raw DSN was used."
        ),
    )
    with pytest.raises(ThreadDatabaseRequiredError):
        await SQLSaber.resume(
            unsafe_id,
            options=_options(database=None),
            storage=storage,
        )


@pytest.mark.asyncio
async def test_corrupt_history_cannot_resume_and_overwrite_the_snapshot(
    temp_dir,
    monkeypatch,
) -> None:
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_id, _ = await _stored_thread(
        storage,
        database_name="database",
        extra_metadata=encode_thread_resume_disabled_metadata(
            reason="An explicit database override is required."
        ),
    )
    corrupt_history = b"not-json"
    async with aiosqlite.connect(storage.db_path) as db:
        await db.execute(
            "UPDATE threads SET messages_json = ? WHERE id = ?",
            (corrupt_history, thread_id),
        )
        await db.commit()

    resume_error: ThreadResumeHistoryError | None = None
    try:
        saber = await SQLSaber.resume(
            thread_id,
            options=_options(database="sqlite:///:memory:"),
            storage=storage,
        )
        created = _turn("replacement", "replacement answer")

        async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
            del prompt, kwargs
            return _RunResult(
                output="replacement answer",
                created_messages=created,
                history=created,
            )

        monkeypatch.setattr(saber.agent, "run", fake_run)
        try:
            await saber.query("replacement")
        finally:
            await saber.close()
    except ThreadResumeHistoryError as exc:
        resume_error = exc

    async with aiosqlite.connect(storage.db_path) as db:
        async with db.execute(
            "SELECT messages_json FROM threads WHERE id = ?", (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()

    assert resume_error is not None
    assert resume_error.thread_id == thread_id
    assert "corrupt" in resume_error.reason.lower()
    assert row is not None
    assert bytes(row[0]) == corrupt_history


@pytest.mark.asyncio
async def test_resume_rejects_unavailable_configured_names_and_accepts_override(
    temp_dir,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "platformdirs.user_config_dir",
        lambda *args, **kwargs: str(temp_dir / "config"),
    )
    storage = ThreadStorage()
    storage.db_path = temp_dir / "threads.db"
    thread_id, _ = await _stored_thread(
        storage,
        database_name="removed",
        extra_metadata=encode_thread_extra_metadata(database_selector="removed"),
    )

    with pytest.raises(ThreadDatabaseUnavailableError) as error:
        await SQLSaber.resume(
            thread_id,
            options=_options(database=None),
            storage=storage,
        )
    assert error.value.database_names == ("removed",)

    saber = await SQLSaber.resume(
        thread_id,
        options=_options(database="sqlite:///:memory:"),
        storage=storage,
    )
    try:
        assert saber.info.thread_id == thread_id
        assert saber.info.primary_database_type == "SQLite"
    finally:
        await saber.close()
