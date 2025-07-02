"""Tests for the Anthropic client implementation."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from sqlsaber.clients.anthropic import AnthropicClient
from sqlsaber.clients.exceptions import (
    AuthenticationError,
    LLMClientError,
)
from sqlsaber.clients.models import (
    CreateMessageRequest,
    Message,
    MessageRole,
)


@pytest.fixture
def client():
    """Create an Anthropic client for testing."""
    return AnthropicClient(api_key="test-key")


@pytest.fixture
def sample_request():
    """Create a sample request for testing."""
    return CreateMessageRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message(MessageRole.USER, "Hello")],
        max_tokens=100,
    )


class TestAnthropicClient:
    """Test cases for AnthropicClient."""

    def test_init(self):
        """Test client initialization."""
        client = AnthropicClient("test-key")
        assert client.api_key == "test-key"
        assert client.base_url == "https://api.anthropic.com"
        assert client.session is None

    def test_init_with_custom_base_url(self):
        """Test client initialization with custom base URL."""
        client = AnthropicClient("test-key", "https://custom.api.com")
        assert client.base_url == "https://custom.api.com"

    def test_get_headers(self, client):
        """Test header generation."""
        headers = client._get_headers()
        expected = {
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        assert headers == expected

    @pytest.mark.asyncio
    async def test_create_message_success(self, client, sample_request):
        """Test successful message creation."""
        mock_response_data = {
            "id": "msg_123",
            "model": "claude-sonnet-4-20250514",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"request-id": "req_123"}
        mock_response.json = AsyncMock(return_value=mock_response_data)

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(client, "_get_session", return_value=mock_session):
            response = await client.create_message(sample_request)

        assert response.id == "msg_123"
        assert response.model == "claude-sonnet-4-20250514"
        assert len(response.content) == 1
        assert response.content[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_create_message_error(self, client, sample_request):
        """Test message creation with API error."""
        error_response = {
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key",
            },
        }

        mock_response = Mock()
        mock_response.status = 401
        mock_response.headers = {"request-id": "req_123"}
        mock_response.json = AsyncMock(return_value=error_response)

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with patch.object(client, "_get_session", return_value=mock_session):
            with pytest.raises(AuthenticationError) as exc_info:
                await client.create_message(sample_request)

        assert exc_info.value.status_code == 401
        assert exc_info.value.request_id == "req_123"

    @pytest.mark.asyncio
    async def test_create_message_stream_error_for_non_stream(
        self, client, sample_request
    ):
        """Test that create_message raises error for streaming requests."""
        sample_request.stream = True

        with pytest.raises(
            ValueError, match="Use create_message_stream for streaming requests"
        ):
            await client.create_message(sample_request)

    @pytest.mark.asyncio
    async def test_create_message_stream_success(self, client, sample_request):
        """Test successful streaming message creation."""
        # Mock SSE data
        sse_data = [
            "event: message_start\n",
            'data: {"type":"message_start","message":{"id":"msg_123"}}\n',
            "\n",
            "event: content_block_start\n",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
            "\n",
            "event: content_block_delta\n",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n',
            "\n",
            "event: message_stop\n",
            'data: {"type":"message_stop"}\n',
            "\n",
        ]

        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {"request-id": "req_123"}

        # Mock the content iterator
        mock_content = Mock()
        mock_content.iter_any = AsyncMock()
        mock_content.iter_any.return_value = (chunk.encode() for chunk in sse_data)
        mock_response.content = mock_content

        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        events = []
        with patch.object(client, "_get_session", return_value=mock_session):
            async for event in client.create_message_stream(sample_request):
                events.append(event)

        # Should have received events from the stream
        assert len(events) > 0
        # Check that we got message_start event
        message_start_events = [
            e for e in events if hasattr(e, "type") and e.type == "message_start"
        ]
        assert len(message_start_events) > 0

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test client cleanup."""
        mock_session = AsyncMock()
        mock_session.closed = False
        client.session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client.session is None

    @pytest.mark.asyncio
    async def test_close_no_session(self, client):
        """Test cleanup when no session exists."""
        # Should not raise any errors
        await client.close()

    @pytest.mark.asyncio
    async def test_close_already_closed_session(self, client):
        """Test cleanup when session is already closed."""
        mock_session = AsyncMock()
        mock_session.closed = True
        client.session = mock_session

        await client.close()

        # Should not call close on already closed session
        mock_session.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_sse_stream_processing_with_ping(self, client):
        """Test SSE stream processing with ping events."""
        sse_data = [
            "event: ping\n",
            'data: {"type": "ping"}\n',
            "\n",
            "event: message_start\n",
            'data: {"type":"message_start","message":{}}\n',
            "\n",
        ]

        mock_response = Mock()
        mock_content = Mock()
        mock_content.iter_any = AsyncMock()
        mock_content.iter_any.return_value = (chunk.encode() for chunk in sse_data)
        mock_response.content = mock_content

        events = []
        async for event in client._process_sse_stream(mock_response):
            events.append(event)

        # Should have ping and message_start events
        assert len(events) == 2
        assert events[0]["type"] == "ping"
        assert events[1]["type"] == "message_start"

    @pytest.mark.asyncio
    async def test_sse_stream_processing_with_error(self, client):
        """Test SSE stream processing with error event."""
        sse_data = [
            "event: error\n",
            'data: {"type": "error", "message": "Something went wrong"}\n',
            "\n",
        ]

        mock_response = Mock()
        mock_content = Mock()
        mock_content.iter_any = AsyncMock()
        mock_content.iter_any.return_value = (chunk.encode() for chunk in sse_data)
        mock_response.content = mock_content

        with pytest.raises(LLMClientError, match="Something went wrong"):
            async for event in client._process_sse_stream(mock_response):
                pass

    @pytest.mark.asyncio
    async def test_sse_stream_processing_with_cancellation(self, client):
        """Test SSE stream processing with cancellation."""
        sse_data = [
            "event: message_start\n",
            'data: {"type":"message_start"}\n',
            "\n",
        ]

        mock_response = Mock()
        mock_content = Mock()
        mock_content.iter_any = AsyncMock()
        mock_content.iter_any.return_value = (chunk.encode() for chunk in sse_data)
        mock_response.content = mock_content

        cancellation_token = asyncio.Event()
        cancellation_token.set()  # Cancel immediately

        events = []
        async for event in client._process_sse_stream(
            mock_response, cancellation_token
        ):
            events.append(event)

        # Should not process any events due to immediate cancellation
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        async with AnthropicClient("test-key") as client:
            assert client.api_key == "test-key"
        # Client should be cleaned up after exiting context
