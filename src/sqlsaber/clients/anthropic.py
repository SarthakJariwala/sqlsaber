"""Anthropic API client implementation."""

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp

from .base import BaseLLMClient
from .exceptions import LLMClientError, create_exception_from_response
from .models import CreateMessageRequest, MessageResponse
from .streaming import AnthropicStreamAdapter, StreamingResponse

logger = logging.getLogger(__name__)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude API."""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            base_url: Base URL for the API (defaults to Anthropic's API)
        """
        super().__init__(api_key, base_url)
        self.base_url = base_url or "https://api.anthropic.com"
        self.session: Optional[aiohttp.ClientSession] = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def _get_headers(self) -> Dict[str, str]:
        """Get the standard headers for API requests."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    async def create_message(self, request: CreateMessageRequest) -> MessageResponse:
        """Create a message and return the response.

        Args:
            request: The message creation request

        Returns:
            The message response

        Raises:
            LLMClientError: If the request fails
        """
        if request.stream:
            raise ValueError("Use create_message_stream for streaming requests")

        session = self._get_session()
        url = f"{self.base_url}/v1/messages"
        headers = self._get_headers()
        data = request.to_dict()

        async with session.post(url, headers=headers, json=data) as response:
            request_id = response.headers.get("request-id")
            response_data = await response.json()

            if response.status != 200:
                raise create_exception_from_response(
                    response.status, response_data, request_id
                )

            return MessageResponse.from_dict(response_data)

    async def create_message_stream(
        self,
        request: CreateMessageRequest,
        cancellation_token: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Any]:
        """Create a message and stream the response.

        Args:
            request: The message creation request
            cancellation_token: Optional event to signal cancellation

        Yields:
            Stream events compatible with the current agent

        Raises:
            LLMClientError: If the request fails
        """
        # Force streaming to be enabled
        request.stream = True

        session = self._get_session()
        url = f"{self.base_url}/v1/messages"
        headers = self._get_headers()
        data = request.to_dict()

        try:
            async with session.post(url, headers=headers, json=data) as response:
                request_id = response.headers.get("request-id")

                if response.status != 200:
                    response_data = await response.json()
                    raise create_exception_from_response(
                        response.status, response_data, request_id
                    )

                # Use stream adapter to convert raw events
                adapter = AnthropicStreamAdapter()
                raw_stream = self._process_sse_stream(response, cancellation_token)

                async for event in adapter.process_stream(
                    raw_stream, cancellation_token
                ):
                    yield event

        except asyncio.CancelledError:
            # Handle cancellation gracefully
            logger.debug("Stream cancelled")
            return
        except Exception as e:
            if not isinstance(e, LLMClientError):
                raise LLMClientError(f"Stream processing error: {str(e)}")
            raise

    async def create_message_with_tools(
        self,
        request: CreateMessageRequest,
        cancellation_token: Optional[asyncio.Event] = None,
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
        # Force streaming to be enabled
        request.stream = True

        session = self._get_session()
        url = f"{self.base_url}/v1/messages"
        headers = self._get_headers()
        data = request.to_dict()

        try:
            async with session.post(url, headers=headers, json=data) as response:
                request_id = response.headers.get("request-id")

                if response.status != 200:
                    response_data = await response.json()
                    raise create_exception_from_response(
                        response.status, response_data, request_id
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

    async def _process_sse_stream(
        self,
        response: aiohttp.ClientResponse,
        cancellation_token: Optional[asyncio.Event] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
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
            async for chunk in response.content.iter_any():
                # Check for cancellation
                if cancellation_token is not None and cancellation_token.is_set():
                    return

                try:
                    buffer += chunk.decode("utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(f"Failed to decode chunk: {e}")
                    continue

                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue

                    # Parse server-sent event format
                    if line.startswith("event: "):
                        event_type = line[7:]
                    elif line.startswith("data: "):
                        event_data = line[6:]

                        # Handle special event types
                        if event_type == "ping":
                            try:
                                yield {"type": "ping", "data": json.loads(event_data)}
                            except json.JSONDecodeError:
                                yield {"type": "ping", "data": {}}
                            continue
                        elif event_type == "error":
                            try:
                                error_data = json.loads(event_data)
                                raise LLMClientError(
                                    error_data.get("message", "Stream error"),
                                    error_data.get("type", "stream_error"),
                                )
                            except json.JSONDecodeError:
                                raise LLMClientError("Stream error with invalid JSON")

                        # Parse JSON data
                        try:
                            parsed_data = json.loads(event_data)
                            yield {
                                "type": event_type,
                                "data": parsed_data,
                            }
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Failed to parse stream data for event {event_type}: {e}"
                            )
                            continue

        except aiohttp.ClientError as e:
            raise LLMClientError(f"Network error during streaming: {str(e)}")
        except asyncio.TimeoutError:
            raise LLMClientError("Stream timeout")
        except Exception as e:
            raise LLMClientError(f"Unexpected error during streaming: {str(e)}")

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
