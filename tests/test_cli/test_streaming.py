"""Tests for streaming query handling."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from sqlsaber.cli.streaming import StreamingQueryHandler
from sqlsaber.theme.manager import create_console


@pytest.mark.asyncio
async def test_event_stream_updates_replay_messages(monkeypatch: pytest.MonkeyPatch):
    console = create_console(file=StringIO(), width=120, legacy_windows=False)
    handler = StreamingQueryHandler(console)

    on_event = AsyncMock()
    monkeypatch.setattr(handler, "on_event", on_event)

    set_replay_messages = Mock(wraps=handler.display.set_replay_messages)
    monkeypatch.setattr(
        handler.display, "set_replay_messages", set_replay_messages
    )

    messages = [SimpleNamespace(parts=[])]
    ctx = SimpleNamespace(messages=messages)

    async def _events():
        yield object()
        yield object()

    await handler._event_stream_handler(ctx, _events())

    assert set_replay_messages.call_count == 2
    set_replay_messages.assert_called_with(messages)
    assert on_event.await_count == 2
