"""Tests for the Anthropic client implementation."""

from unittest.mock import AsyncMock

import pytest

from sqlsaber.clients.anthropic import AnthropicClient
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
    async def test_context_manager(self):
        """Test using client as async context manager."""
        async with AnthropicClient("test-key") as client:
            assert client.api_key == "test-key"
        # Client should be cleaned up after exiting context
