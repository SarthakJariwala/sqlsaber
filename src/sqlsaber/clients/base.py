"""Abstract base class for LLM clients."""

from abc import ABC


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""

    def __init__(self, api_key: str, base_url: str | None = None):
        """Initialize the client with API key and optional base URL.

        Args:
            api_key: API key for authentication
            base_url: Base URL for the API (optional, uses default if not provided)
        """
        self.api_key = api_key
        self.base_url = base_url

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
