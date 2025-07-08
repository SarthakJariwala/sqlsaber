"""Anthropic API client implementation."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx

from .base import BaseLLMClient
from .exceptions import LLMClientError, create_exception_from_response
from .models import CreateMessageRequest
from .streaming import AnthropicStreamAdapter, StreamingResponse

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        oauth_token: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for the API (defaults to Anthropic's API)
        """
        super().__init__(api_key or "", base_url)

        if not api_key and not oauth_token:
            raise ValueError("Either api_key or oauth_token must be provided")

        self.oauth_token = oauth_token
        self.use_oauth = oauth_token is not None
        self.base_url = base_url or "https://api.anthropic.com"
        self.client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self.client is None or self.client.is_closed:
            # Configure timeouts and connection limits for reliability
            timeout = httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=60.0,  # Read timeout for streaming
                write=10.0,  # Write timeout
                pool=10.0,  # Pool timeout
            )
            limits = httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            )
            self.client = httpx.AsyncClient(
                timeout=timeout, limits=limits, follow_redirects=True
            )
        return self.client

    def _get_headers(self) -> dict[str, str]:
        """Get the standard headers for API requests."""
        if self.use_oauth:
            # OAuth headers for Claude Pro authentication (matching Claude Code CLI)
            return {
                "Authorization": f"Bearer {self.oauth_token}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "oauth-2025-04-20",
                "User-Agent": "ClaudeCode/1.0 (Anthropic Claude Code CLI)",
                "Accept": "application/json",
                "X-Client-Name": "claude-code",
                "X-Client-Version": "1.0.0",
            }
        else:
            # API key headers for standard authentication
            return {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }

    async def create_message_with_tools(
        self,
        request: CreateMessageRequest,
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator[Any]:
        """Create a message with tool support and stream the response.

        This method handles the full message creation flow including tool use,
        similar to what the current AnthropicSQLAgent expects.

        Args:
            request: The message creation request
            cancellation_token: Optional event to signal cancellation

        Yields:
            Stream events and final StreamingResponse
        """
        request.stream = True

        client = self._get_client()
        url = f"{self.base_url}/v1/messages"
        headers = self._get_headers()
        data = request.to_dict()

        try:
            async with client.stream(
                "POST", url, headers=headers, json=data
            ) as response:
                request_id = response.headers.get("request-id")

                if response.status_code != 200:
                    response_content = await response.aread()
                    response_data = json.loads(response_content.decode())
                    raise create_exception_from_response(
                        response.status_code, response_data, request_id
                    )

                # Use stream adapter to convert raw events and track state
                adapter = AnthropicStreamAdapter()
                raw_stream = self._process_sse_stream(response, cancellation_token)

                async for event in adapter.process_stream(
                    raw_stream, cancellation_token
                ):
                    yield event

                # Create final response object with proper state
                response_obj = StreamingResponse(
                    content=adapter.get_content_blocks(),
                    stop_reason=adapter.get_stop_reason(),
                )

                # Yield special event with response
                yield {"type": "response_ready", "data": response_obj}

        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.debug("Stream cancelled")
            return
        except Exception as e:
            if not isinstance(e, LLMClientError):
                raise LLMClientError(f"Stream processing error: {str(e)}")
            raise

    def _handle_ping_event(self, event_data: str) -> dict[str, Any]:
        """Handle ping event data.

        Args:
            event_data: Raw event data string

        Returns:
            Parsed ping event
        """
        try:
            return {"type": "ping", "data": json.loads(event_data)}
        except json.JSONDecodeError:
            return {"type": "ping", "data": {}}

    def _handle_error_event(self, event_data: str) -> None:
        """Handle error event data.

        Args:
            event_data: Raw event data string

        Raises:
            LLMClientError: Always raises with error details
        """
        try:
            error_data = json.loads(event_data)
            raise LLMClientError(
                error_data.get("message", "Stream error"),
                error_data.get("type", "stream_error"),
            )
        except json.JSONDecodeError:
            raise LLMClientError("Stream error with invalid JSON")

    def _parse_event_data(
        self, event_type: str | None, event_data: str
    ) -> dict[str, Any] | None:
        """Parse event data based on event type.

        Args:
            event_type: Type of the event
            event_data: Raw event data string

        Returns:
            Parsed event or None if parsing failed
        """
        try:
            parsed_data = json.loads(event_data)
            return {"type": event_type, "data": parsed_data}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse stream data for event {event_type}: {e}")
            return None

    def _process_sse_line(
        self, line: str, event_type: str | None
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Process a single SSE line.

        Args:
            line: Line to process
            event_type: Current event type

        Returns:
            Tuple of (new_event_type, event_to_yield)
        """
        if line.startswith("event: "):
            return line[7:], None
        elif line.startswith("data: "):
            event_data = line[6:]

            if event_type == "ping":
                return event_type, self._handle_ping_event(event_data)
            elif event_type == "error":
                self._handle_error_event(event_data)
                return event_type, None  # Never reached due to exception
            else:
                parsed_event = self._parse_event_data(event_type, event_data)
                return event_type, parsed_event

        return event_type, None

    async def _process_sse_stream(
        self,
        response: httpx.Response,
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Process server-sent events from the response stream.

        Args:
            response: The HTTP response object
            cancellation_token: Optional event to signal cancellation

        Yields:
            Parsed stream events

        Raises:
            LLMClientError: If stream processing fails
        """
        buffer = ""
        event_type = None

        try:
            async for chunk in response.aiter_bytes():
                if cancellation_token is not None and cancellation_token.is_set():
                    return

                try:
                    buffer += chunk.decode("utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to decode chunk: {e}")
                    continue

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    event_type, event_to_yield = self._process_sse_line(
                        line, event_type
                    )
                    if event_to_yield is not None:
                        yield event_to_yield

        except httpx.TimeoutException as e:
            raise LLMClientError(f"Stream timeout error: {str(e)}")
        except httpx.NetworkError as e:
            raise LLMClientError(f"Network error during streaming: {str(e)}")
        except httpx.HTTPError as e:
            raise LLMClientError(f"HTTP error during streaming: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Stream timeout")
        except Exception as e:
            raise LLMClientError(f"Unexpected error during streaming: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        if self.client and not self.client.is_closed:
            await self.client.aclose()
            self.client = None
