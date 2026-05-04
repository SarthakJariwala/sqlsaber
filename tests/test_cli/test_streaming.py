"""Tests for streaming query handling."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.theme.manager import create_console


@pytest.mark.asyncio
async def test_event_stream_updates_replay_messages(monkeypatch: pytest.MonkeyPatch):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    on_event = AsyncMock()
    monkeypatch.setattr(handler, "on_event", on_event)

    set_replay_messages = Mock(wraps=handler.display.set_replay_messages)
    monkeypatch.setattr(handler.display, "set_replay_messages", set_replay_messages)

    messages = [SimpleNamespace(parts=[])]
    ctx = SimpleNamespace(messages=messages)

    async def _events():
        yield object()
        yield object()

    await handler._event_stream_handler(ctx, _events())

    assert set_replay_messages.call_count == 2
    set_replay_messages.assert_called_with(messages)
    assert on_event.await_count == 2


@pytest.mark.asyncio
async def test_execute_sql_part_start_shows_generating_status(
    monkeypatch: pytest.MonkeyPatch,
):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    start_status = Mock()
    monkeypatch.setattr(handler.display.live, "start_status", start_status)

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name="execute_sql",
                args={},
                tool_call_id="call-1",
            ),
        ),
        SimpleNamespace(messages=[]),
    )

    start_status.assert_called_once_with("Generating SQL...")


@pytest.mark.asyncio
async def test_execute_sql_delta_name_shows_generating_status(
    monkeypatch: pytest.MonkeyPatch,
):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    start_status = Mock()
    monkeypatch.setattr(handler.display.live, "start_status", start_status)

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name="execute",
                args={},
                tool_call_id="call-2",
            ),
        ),
        SimpleNamespace(messages=[]),
    )
    await handler.on_event(
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(tool_name_delta="_sql")),
        SimpleNamespace(messages=[]),
    )

    start_status.assert_called_once_with("Generating SQL...")


@pytest.mark.asyncio
async def test_ask_database_part_start_shows_querying_status(
    monkeypatch: pytest.MonkeyPatch,
):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    start_status = Mock()
    monkeypatch.setattr(handler.display.live, "start_status", start_status)

    await handler.on_event(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name="ask_database",
                args={},
                tool_call_id="call-3",
            ),
        ),
        SimpleNamespace(messages=[]),
    )

    start_status.assert_called_once_with("Querying database...")


@pytest.mark.asyncio
async def test_ask_database_tool_call_shows_target_database_status(
    monkeypatch: pytest.MonkeyPatch,
):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    start_status = Mock()
    monkeypatch.setattr(handler.display.live, "start_status", start_status)
    monkeypatch.setattr(handler.display.live, "end_status", Mock())
    monkeypatch.setattr(handler.display.live, "end_if_active", Mock())
    monkeypatch.setattr(handler.display, "show_tool_executing", Mock())

    await handler.on_event(
        FunctionToolCallEvent(
            part=ToolCallPart(
                tool_name="ask_database",
                args={"database_id": "orders", "question": "How many orders?"},
                tool_call_id="call-4",
            )
        ),
        SimpleNamespace(messages=[]),
    )

    handler.display.show_tool_executing.assert_called_once_with(
        "ask_database",
        {"database_id": "orders", "question": "How many orders?"},
        tool_call_id="call-4",
    )
    start_status.assert_called_once_with("Querying database orders...")


@pytest.mark.asyncio
async def test_tool_result_forwards_tool_call_id_to_display(
    monkeypatch: pytest.MonkeyPatch,
):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    show_tool_result = Mock()
    monkeypatch.setattr(handler.display, "show_tool_result", show_tool_result)
    monkeypatch.setattr(handler.display.live, "end_status", Mock())
    monkeypatch.setattr(handler.display, "show_newline", Mock())
    monkeypatch.setattr(handler.display.live, "start_status", Mock())

    await handler.on_event(
        FunctionToolResultEvent(
            result=ToolReturnPart(
                tool_name="ask_database",
                content="child answer",
                tool_call_id="call-5",
            )
        ),
        SimpleNamespace(messages=[]),
    )

    show_tool_result.assert_called_once_with(
        "ask_database", "child answer", tool_call_id="call-5"
    )
