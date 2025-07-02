"""Streaming adapters and utilities for LLM clients."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


class AnthropicStreamAdapter:
    """Adapter to convert raw Anthropic stream events to standardized format.

    This adapter converts the raw SSE events from Anthropic API into objects
    that match the structure expected by the current AnthropicSQLAgent.
    """

    def __init__(self):
        self.content_blocks: list[dict[str, Any]] = []
        self.tool_use_blocks: list[dict[str, Any]] = []

    async def process_stream(
        self,
        raw_stream: AsyncIterator[dict[str, Any]],
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator[Any]:
        """Process raw stream events and yield adapted events.

        Args:
            raw_stream: Raw stream events from the API
            cancellation_token: Optional cancellation token

        Yields:
            Adapted stream events that match the SDK format
        """
        async for raw_event in raw_stream:
            # Check for cancellation
            if cancellation_token is not None and cancellation_token.is_set():
                return

            # Convert raw event to SDK-like event
            adapted_event = self._adapt_event(raw_event)
            if adapted_event:
                yield adapted_event

    def _adapt_event(self, raw_event: dict[str, Any]) -> Any | None:
        """Adapt a raw stream event to match SDK format.

        Args:
            raw_event: Raw event from the API

        Returns:
            Adapted event object or None if event should be filtered
        """
        event_type = raw_event.get("type")
        event_data = raw_event.get("data", {})

        if event_type == "ping":
            # Create a ping event object
            return PingEvent()

        elif event_type == "message_start":
            # Create message start event
            return MessageStartEvent(event_data.get("message", {}))

        elif event_type == "content_block_start":
            # Create content block start event
            index = event_data.get("index", 0)
            content_block = event_data.get("content_block", {})

            # Initialize content blocks list if needed
            while len(self.content_blocks) <= index:
                self.content_blocks.append({"type": "text", "text": ""})

            if content_block.get("type") == "tool_use":
                # Add to tool use blocks tracking
                tool_block = {
                    "id": content_block.get("id"),
                    "name": content_block.get("name"),
                    "input": {},
                    "_partial": "",
                }
                self.tool_use_blocks.append(tool_block)

            return ContentBlockStartEvent(index, content_block)

        elif event_type == "content_block_delta":
            # Create content block delta event
            index = event_data.get("index", 0)
            delta = event_data.get("delta", {})

            # Update content blocks tracking
            if index < len(self.content_blocks):
                if delta.get("type") == "text_delta":
                    self.content_blocks[index]["text"] += delta.get("text", "")
                elif delta.get("type") == "input_json_delta":
                    # Update tool use input tracking
                    if self.tool_use_blocks:
                        current_tool = self.tool_use_blocks[-1]
                        current_tool["_partial"] += delta.get("partial_json", "")
                        try:
                            current_tool["input"] = json.loads(current_tool["_partial"])
                        except json.JSONDecodeError:
                            pass  # Partial JSON, continue accumulating

            return ContentBlockDeltaEvent(index, delta)

        elif event_type == "content_block_stop":
            # Create content block stop event
            index = event_data.get("index", 0)
            return ContentBlockStopEvent(index)

        elif event_type == "message_delta":
            # Create message delta event
            delta = event_data.get("delta", {})
            usage = event_data.get("usage", {})
            return MessageDeltaEvent(delta, usage)

        elif event_type == "message_stop":
            # Finalize tool blocks
            self._finalize_tool_blocks()
            return MessageStopEvent()

        elif event_type == "error":
            # Create error event
            return ErrorEvent(event_data)

        else:
            # Unknown event type, log and ignore
            logger.debug(f"Unknown event type: {event_type}")
            return None

    def _finalize_tool_blocks(self):
        """Finalize tool use blocks by cleaning up and adding to content blocks."""
        for block in self.tool_use_blocks:
            block["type"] = "tool_use"
            if "_partial" in block:
                del block["_partial"]
            self.content_blocks.append(block)

    def get_stop_reason(self) -> str:
        """Get the stop reason based on current state."""
        if self.tool_use_blocks:
            return "tool_use"
        return "stop"

    def get_content_blocks(self) -> list[dict[str, Any]]:
        """Get the current content blocks."""
        return self.content_blocks.copy()


# Event classes that match the SDK structure
class BaseStreamEvent:
    """Base class for stream events."""

    def __init__(self, event_type: str):
        self.type = event_type


class PingEvent(BaseStreamEvent):
    """Ping event."""

    def __init__(self):
        super().__init__("ping")


class MessageStartEvent(BaseStreamEvent):
    """Message start event."""

    def __init__(self, message: dict[str, Any]):
        super().__init__("message_start")
        self.message = message


class ContentBlockStartEvent(BaseStreamEvent):
    """Content block start event."""

    def __init__(self, index: int, content_block: dict[str, Any]):
        super().__init__("content_block_start")
        self.index = index
        self.content_block = MockContentBlock(content_block)


class ContentBlockDeltaEvent(BaseStreamEvent):
    """Content block delta event."""

    def __init__(self, index: int, delta: dict[str, Any]):
        super().__init__("content_block_delta")
        self.index = index
        self.delta = MockDelta(delta)


class ContentBlockStopEvent(BaseStreamEvent):
    """Content block stop event."""

    def __init__(self, index: int):
        super().__init__("content_block_stop")
        self.index = index


class MessageDeltaEvent(BaseStreamEvent):
    """Message delta event."""

    def __init__(self, delta: dict[str, Any], usage: dict[str, Any]):
        super().__init__("message_delta")
        self.delta = delta
        self.usage = usage


class MessageStopEvent(BaseStreamEvent):
    """Message stop event."""

    def __init__(self):
        super().__init__("message_stop")


class ErrorEvent(BaseStreamEvent):
    """Error event."""

    def __init__(self, error_data: dict[str, Any]):
        super().__init__("error")
        self.error = error_data


# Mock classes to match SDK object structure
class MockContentBlock:
    """Mock content block object that matches SDK structure."""

    def __init__(self, data: dict[str, Any]):
        self.type = data.get("type")
        self.id = data.get("id")
        self.name = data.get("name")
        self.input = data.get("input", {})


class MockDelta:
    """Mock delta object that matches SDK structure."""

    def __init__(self, data: dict[str, Any]):
        self.data = data
        # Set attributes based on delta type
        if data.get("type") == "text_delta":
            self.text = data.get("text", "")
        elif data.get("type") == "input_json_delta":
            self.partial_json = data.get("partial_json", "")

    def __getattr__(self, name):
        """Allow access to any attribute from the data."""
        return self.data.get(name)


class StreamingResponse:
    """Response object for streaming that matches the current agent's expectations."""

    def __init__(self, content: list[dict[str, Any]], stop_reason: str):
        self.content = content
        self.stop_reason = stop_reason
