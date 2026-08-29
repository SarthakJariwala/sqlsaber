"""Tests for streaming query handling."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    ToolCallPart,
    ToolCallPartDelta,
)

from sqlsaber.cli.stream_presenter import AgentStreamPresenter
from sqlsaber.render.terminal import PlainSurface


@pytest.mark.asyncio
async def test_event_stream_updates_replay_messages(monkeypatch: pytest.MonkeyPatch):
    handler = AgentStreamPresenter(PlainSurface(StringIO()))

    on_event = AsyncMock()
    monkeypatch.setattr(handler, "on_event", on_event)

    set_replay_messages = Mock(wraps=handler.set_replay_messages)
    monkeypatch.setattr(handler, "set_replay_messages", set_replay_messages)

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
    handler = AgentStreamPresenter(PlainSurface(StringIO()))
    start_status = Mock()
    monkeypatch.setattr(handler.surface, "status", start_status)

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

    start_status.assert_called_with("Generating SQL...")


@pytest.mark.asyncio
async def test_execute_sql_delta_name_shows_generating_status(
    monkeypatch: pytest.MonkeyPatch,
):
    handler = AgentStreamPresenter(PlainSurface(StringIO()))
    start_status = Mock()
    monkeypatch.setattr(handler.surface, "status", start_status)

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

    start_status.assert_called_with("Generating SQL...")
