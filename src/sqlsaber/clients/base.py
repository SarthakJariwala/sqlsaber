"""Abstract base class for LLM clients."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional

from sqlsaber.clients.models import CreateMessageRequest, MessageResponse


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize the client with API key and optional base URL.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API (optional, uses default if not provided)
        """
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        """Create a message and return the response.

        Args:
            request: The message creation request

        Returns:
            The message response

        Raises:
            LLMClientError: If the request fails
        """
        pass

    @abstractmethod
    async def create_message_stream(
        self,
        request: CreateMessageRequest,
        cancellation_token: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Create a message and stream the response.

        Args:
            request: The message creation request
            cancellation_token: Optional event to signal cancellation

        Yields:
            Stream events as dictionaries

        Raises:
            LLMClientError: If the request fails
        """
        pass

    async def close(self):
        """Close the client and clean up resources."""
        # Default implementation does nothing
        # Subclasses can override to clean up HTTP sessions, etc.
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
