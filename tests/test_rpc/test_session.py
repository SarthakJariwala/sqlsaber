from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from sqlsaber import SQLSaber, SQLSaberOptions
from sqlsaber.config.settings import Config
from sqlsaber.query_results import (
    InMemoryQueryResultStore,
    QueryResultContext,
    QueryResultData,
    descriptor_for_data,
    new_query_result_id,
)
from sqlsaber.rpc.protocol import OVERSIZE_LINE, parse_command
from sqlsaber.rpc.session import RpcSession, map_sdk_error, serve
from sqlsaber.sdk.errors import RunInProgressError, SQLSaberClosedError


def _options(**overrides: Any) -> SQLSaberOptions:
    values: dict[str, Any] = {
        "database": "sqlite:///:memory:",
        "settings": Config.in_memory(
            model_name="anthropic:claude-3-5-sonnet",
            api_keys={"anthropic": "test-key"},
        ),
        "query_result_store": InMemoryQueryResultStore(),
    }
    values.update(overrides)
    return SQLSaberOptions(**values)


def _turn(prompt: str, answer: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(prompt)]),
        ModelResponse(
            parts=[TextPart(answer)],
            usage=RequestUsage(input_tokens=10, output_tokens=2),
        ),
    ]


@dataclass
class _RunResult:
    output: str
    created_messages: list[ModelMessage]
    history: list[ModelMessage]

    def usage(self) -> RunUsage:
        return RunUsage(input_tokens=20, output_tokens=4, requests=1)

    def new_messages(self) -> list[ModelMessage]:
        return list(self.created_messages)

    def all_messages(self) -> list[ModelMessage]:
        return list(self.history)

    def all_messages_json(self) -> bytes:
        return ModelMessagesTypeAdapter.dump_json(self.history)


class QueueReader:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def push(self, obj: dict[str, Any] | bytes) -> None:
        if isinstance(obj, bytes):
            self._queue.put_nowait(obj)
        else:
            self._queue.put_nowait((json.dumps(obj) + "\n").encode())

    def eof(self) -> None:
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


def _load(buf: list[bytes]) -> list[dict[str, Any]]:
    return [json.loads(item) for item in buf]


async def _wait_for(
    buf: list[bytes],
    predicate,
    *,
    timeout: float = 3.0,
) -> list[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        records = _load(buf)
        if predicate(records):
            return records
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for records: {_load(buf)}")


def _patch_answer(monkeypatch, saber: SQLSaber, *, hang: asyncio.Event | None = None):
    async def fake_run(prompt: str, **kwargs: Any) -> _RunResult:
        handler = kwargs.get("event_stream_handler")
        if handler is not None:
            ctx = SimpleNamespace(model=SimpleNamespace(model_name="claude-3-5-sonnet"))

            async def events():
                yield PartStartEvent(index=0, part=TextPart("Hello"))
                if hang is not None:
                    await hang.wait()
                yield PartEndEvent(index=0, part=TextPart("Hello"))

            await handler(ctx, events())
        created = _turn(prompt, "Hello")
        return _RunResult(
            output="Hello",
            created_messages=created,
            history=[*list(saber.messages), *created],
        )

    monkeypatch.setattr(saber.agent, "run", fake_run)


@pytest.mark.asyncio
async def test_prompt_streams_then_agent_end(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    _patch_answer(monkeypatch, saber)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=False)
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        assert _load(buf)[0]["thinkingLevel"] == "off"
        reader.push({"id": "q1", "type": "prompt", "message": "hi"})
        records = await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_end" for row in rows)
        )
        types = [row["type"] for row in records]
        assert types[:3] == ["ready", "response", "agent_start"]
        assert records[1]["command"] == "prompt" and records[1]["success"] is True
        assert records[2].get("promptId") == "q1"
        assert "message_start" in types
        assert any(
            row.get("type") == "message_update"
            and row.get("assistantMessageEvent", {}).get("type") == "text_delta"
            for row in records
        )
        end = next(row for row in records if row["type"] == "agent_end")
        assert end["status"] == "completed"
        assert end["text"] == "Hello"
        reader.push({"type": "get_messages"})
        records = await _wait_for(
            buf, lambda rows: any(row.get("command") == "get_messages" for row in rows)
        )
        messages = next(row for row in records if row.get("command") == "get_messages")[
            "data"
        ]["messages"]
        assert messages[0]["role"] == "user"
        assert messages[-1]["role"] == "assistant"
        reader.push({"type": "shutdown"})
        await task
    finally:
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_overlapping_prompt_is_rejected(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    hang = asyncio.Event()
    _patch_answer(monkeypatch, saber, hang=hang)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=False)
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push({"id": 1, "type": "prompt", "message": "first"})
        await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_start" for row in rows)
        )
        reader.push({"id": 2, "type": "prompt", "message": "second"})
        records = await _wait_for(
            buf,
            lambda rows: any(
                row.get("command") == "prompt" and row.get("id") == 2 for row in rows
            ),
        )
        rejected = next(row for row in records if row.get("id") == 2)
        assert rejected["success"] is False
        assert "already running" in rejected["error"]
        hang.set()
        reader.push({"type": "shutdown"})
        await task
    finally:
        hang.set()
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_abort_idle_is_idempotent(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    _patch_answer(monkeypatch, saber)
    buf: list[bytes] = []
    session = RpcSession(saber, write=buf.append, persist_thread=False)
    first = await session.handle(parse_command(b'{"id":"a","type":"abort"}'))
    second = await session.handle(parse_command(b'{"id":"b","type":"abort"}'))
    session.send(first)
    session.send(second)
    records = _load(buf)
    assert records[0]["data"] == {"aborted": False}
    assert records[1]["data"] == {"aborted": False}
    await saber.close()


@pytest.mark.asyncio
async def test_abort_mid_run_keeps_reading_stdin(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    hang = asyncio.Event()
    _patch_answer(monkeypatch, saber, hang=hang)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(
            saber,
            reader=reader,
            write=buf.append,
            persist_thread=False,
            abort_grace=0.05,
        )
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push({"id": "q", "type": "prompt", "message": "slow"})
        await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_start" for row in rows)
        )
        reader.push({"id": "a", "type": "abort"})
        reader.push({"id": "s", "type": "get_state"})
        records = await _wait_for(
            buf,
            lambda rows: (
                any(row.get("command") == "abort" for row in rows)
                and any(row.get("type") == "agent_end" for row in rows)
            ),
        )
        state = next(row for row in records if row.get("command") == "get_state")
        abort = next(row for row in records if row.get("command") == "abort")
        end = next(row for row in records if row.get("type") == "agent_end")
        assert state["success"] is True
        assert records.index(state) < records.index(end)
        assert records.index(end) < records.index(abort)
        assert end["status"] == "aborted"
        assert abort["data"] == {"aborted": True}
        assert saber.messages == []
        reader.push({"type": "shutdown"})
        await task
    finally:
        hang.set()
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_error_after_accept_is_agent_end(monkeypatch) -> None:
    saber = SQLSaber(options=_options())

    async def boom(prompt: str, **kwargs: Any) -> _RunResult:
        del prompt, kwargs
        raise RuntimeError("anthropic: 529 overloaded_error")

    monkeypatch.setattr(saber.agent, "run", boom)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=False)
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push({"id": "q", "type": "prompt", "message": "hi"})
        records = await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_end" for row in rows)
        )
        accepted = next(
            row
            for row in records
            if row.get("command") == "prompt" and row.get("id") == "q"
        )
        assert accepted["success"] is True
        end = next(row for row in records if row["type"] == "agent_end")
        assert end["status"] == "error"
        assert "529" in end["error"]
        reader.push({"type": "shutdown"})
        await task
    finally:
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_new_session_twice_and_mid_run_reads(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    hang = asyncio.Event()
    _patch_answer(monkeypatch, saber, hang=hang)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=True)
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push({"type": "new_session"})
        reader.push({"id": "n2", "type": "new_session"})
        await _wait_for(
            buf,
            lambda rows: (
                sum(1 for row in rows if row.get("command") == "new_session") == 2
            ),
        )
        reader.push({"id": "q", "type": "prompt", "message": "hi"})
        await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_start" for row in rows)
        )
        reader.push({"type": "get_state"})
        reader.push({"type": "get_messages"})
        reader.push({"type": "new_session"})
        reader.push({"type": "get_tables"})
        records = await _wait_for(
            buf,
            lambda rows: (
                any(
                    row.get("command") == "new_session" and row.get("success") is False
                    for row in rows
                )
                and any(row.get("command") == "get_tables" for row in rows)
            ),
        )
        running_state = next(
            row
            for row in records
            if row.get("command") == "get_state" and row.get("success")
        )
        assert running_state["data"]["state"] == "running"
        assert running_state["data"]["threadPersistence"] is True
        rejected_session = next(
            row
            for row in records
            if row.get("command") == "new_session" and row.get("success") is False
        )
        rejected_tables = next(
            row
            for row in records
            if row.get("command") == "get_tables" and row.get("success") is False
        )
        assert rejected_session["success"] is False
        assert rejected_tables["success"] is False
        hang.set()
        reader.push({"type": "shutdown"})
        await task
    finally:
        hang.set()
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_shutdown_while_running_aborts_first(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    hang = asyncio.Event()
    _patch_answer(monkeypatch, saber, hang=hang)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(
            saber,
            reader=reader,
            write=buf.append,
            persist_thread=False,
            abort_grace=0.05,
        )
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push({"type": "prompt", "message": "hi"})
        await _wait_for(
            buf, lambda rows: any(row.get("type") == "agent_start" for row in rows)
        )
        reader.push({"id": "x", "type": "shutdown"})
        records = await _wait_for(
            buf,
            lambda rows: (
                any(row.get("command") == "shutdown" for row in rows)
                and any(row.get("type") == "agent_end" for row in rows)
            ),
        )
        end = next(row for row in records if row["type"] == "agent_end")
        shutdown = next(row for row in records if row.get("command") == "shutdown")
        assert records.index(end) < records.index(shutdown)
        assert end["status"] == "aborted"
        assert shutdown["success"] is True
        await task
        assert all(json.loads(item) for item in buf)
    finally:
        hang.set()
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_eof_closes_without_shutdown_response(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    _patch_answer(monkeypatch, saber)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=False)
    )
    await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
    reader.eof()
    await task
    records = _load(buf)
    assert records[0]["type"] == "ready"
    assert all(row.get("command") != "shutdown" for row in records)
    await saber.close()


@pytest.mark.asyncio
async def test_oversize_line_is_a_parse_error(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    _patch_answer(monkeypatch, saber)
    buf: list[bytes] = []
    reader = QueueReader()
    task = asyncio.create_task(
        serve(saber, reader=reader, write=buf.append, persist_thread=False)
    )
    try:
        await _wait_for(buf, lambda rows: rows and rows[0]["type"] == "ready")
        reader.push(OVERSIZE_LINE)
        records = await _wait_for(
            buf, lambda rows: any(row.get("command") == "parse" for row in rows)
        )
        error = next(row for row in records if row.get("command") == "parse")
        assert error["success"] is False
        assert "1 MiB" in error["error"]
        reader.push({"type": "shutdown"})
        await task
    finally:
        if not task.done():
            reader.eof()
            await task
        await saber.close()


@pytest.mark.asyncio
async def test_get_query_result_pages(monkeypatch) -> None:
    store = InMemoryQueryResultStore()
    saber = SQLSaber(options=_options(query_result_store=store))
    _patch_answer(monkeypatch, saber)
    rows = [{"n": index} for index in range(5)]
    payload = json.dumps({"columns": ["n"], "results": rows}).encode()
    descriptor = descriptor_for_data(
        payload,
        result_id=new_query_result_id(),
        file="result_call.json",
        row_count=5,
        columns=("n",),
    )
    stored = await store.put(
        QueryResultData(payload),
        descriptor=descriptor,
        context=QueryResultContext(),
    )
    buf: list[bytes] = []
    session = RpcSession(saber, write=buf.append, persist_thread=False)
    response = await session.handle(
        parse_command(
            json.dumps(
                {
                    "type": "get_query_result",
                    "resultId": stored.id,
                    "offset": 2,
                    "limit": 2,
                }
            ).encode()
        )
    )
    session.send(response)
    page = _load(buf)[0]["data"]
    assert page["offset"] == 2
    assert page["limit"] == 2
    assert page["rows"] == [{"n": 2}, {"n": 3}]
    assert page["hasMore"] is True
    missing = await session.handle(
        parse_command(
            b'{"type":"get_query_result","resultId":"qr_ffffffffffffffffffffffffffffffff"}'
        )
    )
    session.send(missing)
    error = _load(buf)[1]
    assert error["success"] is False
    assert "unavailable" in error["error"].lower()
    await saber.close()


@pytest.mark.asyncio
async def test_set_thinking_off_is_real_off(monkeypatch) -> None:
    saber = SQLSaber(options=_options())
    buf: list[bytes] = []
    session = RpcSession(saber, write=buf.append, persist_thread=False)
    await session.handle(parse_command(b'{"type":"set_thinking_level","level":"high"}'))
    off = await session.handle(
        parse_command(b'{"type":"set_thinking_level","level":"off"}')
    )
    session.send(off)
    assert _load(buf)[0]["data"]["thinkingLevel"] == "off"
    assert saber.info.thinking.enabled is False
    await saber.close()


def test_map_sdk_error_is_a_string_table() -> None:
    assert "already running" in map_sdk_error(RunInProgressError("busy"))
    assert map_sdk_error(SQLSaberClosedError("closed")) == "SQLSaber is closed"
    assert map_sdk_error(ValueError("bad")) == "bad"
    assert map_sdk_error(RuntimeError("x")).startswith("Internal error:")
