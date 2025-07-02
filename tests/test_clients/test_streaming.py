"""Tests for the streaming adapter."""

import asyncio

import pytest

from sqlsaber.clients.streaming import (
    AnthropicStreamAdapter,
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageStartEvent,
    MessageStopEvent,
    StreamingResponse,
)


class TestAnthropicStreamAdapter:
    """Test cases for AnthropicStreamAdapter."""

    def test_init(self):
        """Test adapter initialization."""
        adapter = AnthropicStreamAdapter()
        assert adapter.content_blocks == []
        assert adapter.tool_use_blocks == []

    @pytest.mark.asyncio
    async def test_process_simple_text_stream(self):
        """Test processing a simple text stream."""
        raw_events = [
            {
                "type": "message_start",
                "data": {"message": {"id": "msg_123"}},
            },
            {
                "type": "content_block_start",
                "data": {
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            },
            {
                "type": "content_block_delta",
                "data": {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
            },
            {
                "type": "content_block_delta",
                "data": {
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " world"},
                },
            },
            {
                "type": "content_block_stop",
                "data": {"index": 0},
            },
            {
                "type": "message_stop",
                "data": {},
            },
        ]

        adapter = AnthropicStreamAdapter()
        events = []

        async def mock_stream():
            for event in raw_events:
                yield event

        async for event in adapter.process_stream(mock_stream()):
            events.append(event)

        # Should have converted all events
        assert len(events) == 6

        # Check event types
        assert isinstance(events[0], MessageStartEvent)
        assert isinstance(events[1], ContentBlockStartEvent)
        assert isinstance(events[2], ContentBlockDeltaEvent)
        assert isinstance(events[3], ContentBlockDeltaEvent)
        assert isinstance(events[5], MessageStopEvent)

        # Check that content was accumulated
        content_blocks = adapter.get_content_blocks()
        assert len(content_blocks) == 1
        assert content_blocks[0]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_process_tool_use_stream(self):
        """Test processing a stream with tool use."""
        raw_events = [
            {
                "type": "message_start",
                "data": {"message": {"id": "msg_123"}},
            },
            {
                "type": "content_block_start",
                "data": {
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_123",
                        "name": "test_tool",
                    },
                },
            },
            {
                "type": "content_block_delta",
                "data": {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"param": '},
                },
            },
            {
                "type": "content_block_delta",
                "data": {
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"value"}'},
                },
            },
            {
                "type": "content_block_stop",
                "data": {"index": 0},
            },
            {
                "type": "message_stop",
                "data": {},
            },
        ]

        adapter = AnthropicStreamAdapter()
        events = []

        async def mock_stream():
            for event in raw_events:
                yield event

        async for event in adapter.process_stream(mock_stream()):
            events.append(event)

        # Check that tool input was parsed
        assert len(adapter.tool_use_blocks) == 1
        tool_block = adapter.tool_use_blocks[0]
        assert tool_block["name"] == "test_tool"
        assert tool_block["input"] == {"param": "value"}

        # Check stop reason
        assert adapter.get_stop_reason() == "tool_use"

    @pytest.mark.asyncio
    async def test_process_stream_with_cancellation(self):
        """Test stream processing with cancellation."""
        raw_events = [
            {
                "type": "message_start",
                "data": {"message": {"id": "msg_123"}},
            },
            {
                "type": "content_block_start",
                "data": {"index": 0, "content_block": {"type": "text"}},
            },
        ]

        adapter = AnthropicStreamAdapter()
        cancellation_token = asyncio.Event()
        events = []

        async def mock_stream():
            for i, event in enumerate(raw_events):
                if i == 1:  # Cancel after first event
                    cancellation_token.set()
                yield event

        async for event in adapter.process_stream(mock_stream(), cancellation_token):
            events.append(event)

        # Should have processed first event before cancellation
        assert len(events) == 1

    def test_adapt_unknown_event(self):
        """Test adapting unknown event type."""
        adapter = AnthropicStreamAdapter()
        raw_event = {"type": "unknown_event", "data": {}}

        result = adapter._adapt_event(raw_event)
        assert result is None

    def test_get_stop_reason_without_tools(self):
        """Test get_stop_reason when no tool use blocks."""
        adapter = AnthropicStreamAdapter()
        assert adapter.get_stop_reason() == "stop"

    def test_get_stop_reason_with_tools(self):
        """Test get_stop_reason when tool use blocks exist."""
        adapter = AnthropicStreamAdapter()
        adapter.tool_use_blocks.append({"id": "test", "name": "test_tool"})
        assert adapter.get_stop_reason() == "tool_use"

    def test_finalize_tool_blocks(self):
        """Test finalizing tool blocks."""
        adapter = AnthropicStreamAdapter()
        adapter.tool_use_blocks = [
            {
                "id": "tool_123",
                "name": "test_tool",
                "input": {"param": "value"},
                "_partial": '{"param": "value"}',
            }
        ]

        adapter._finalize_tool_blocks()

        # Should have cleaned up _partial and added to content blocks
        assert len(adapter.content_blocks) == 1
        tool_block = adapter.content_blocks[0]
        assert tool_block["type"] == "tool_use"
        assert "_partial" not in tool_block
        assert tool_block["name"] == "test_tool"


class TestStreamingResponse:
    """Test cases for StreamingResponse."""

    def test_init(self):
        """Test StreamingResponse initialization."""
        content = [{"type": "text", "text": "Hello"}]
        response = StreamingResponse(content, "end_turn")

        assert response.content == content
        assert response.stop_reason == "end_turn"


class TestStreamEventClasses:
    """Test stream event classes."""

    def test_message_start_event(self):
        """Test MessageStartEvent."""
        message_data = {"id": "msg_123", "role": "assistant"}
        event = MessageStartEvent(message_data)

        assert event.type == "message_start"
        assert event.message == message_data

    def test_content_block_start_event(self):
        """Test ContentBlockStartEvent."""
        content_block_data = {"type": "text", "text": ""}
        event = ContentBlockStartEvent(0, content_block_data)

        assert event.type == "content_block_start"
        assert event.index == 0
        assert event.content_block.type == "text"

    def test_content_block_delta_event(self):
        """Test ContentBlockDeltaEvent."""
        delta_data = {"type": "text_delta", "text": "Hello"}
        event = ContentBlockDeltaEvent(0, delta_data)

        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta.text == "Hello"

    def test_content_block_delta_event_input_json(self):
        """Test ContentBlockDeltaEvent with input JSON delta."""
        delta_data = {"type": "input_json_delta", "partial_json": '{"test"'}
        event = ContentBlockDeltaEvent(0, delta_data)

        assert event.type == "content_block_delta"
        assert event.index == 0
        assert event.delta.partial_json == '{"test"'

    def test_mock_delta_getattr(self):
        """Test MockDelta __getattr__ method."""
        from sqlsaber.clients.streaming import MockDelta

        delta_data = {"custom_field": "value", "type": "text_delta"}
        delta = MockDelta(delta_data)

        # Should be able to access any field from data
        assert delta.custom_field == "value"
        assert delta.type == "text_delta"
