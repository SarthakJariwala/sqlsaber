"""Public Python API for SQLSaber.

This module provides a simplified programmatic interface to SQLSaber's capabilities,
allowing you to run natural language queries against databases from Python code.
"""

from collections.abc import AsyncIterable, Awaitable, Sequence
from types import TracebackType
from typing import Any, Callable, Protocol, Self

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.options import SQLSaberOptions
from sqlsaber.session import create_session


class SQLSaberRunResult(Protocol):
    """Protocol for pydantic-ai run result objects."""

    def usage(self) -> Any: ...
    def new_messages(self) -> list[ModelMessage]: ...
    def all_messages(self) -> list[ModelMessage]: ...


class SQLSaberResult(str):
    """Result of a SQLSaber query.

    Behaves like a string (contains the agent's text response) but also
    provides access to the full execution details including usage stats
    and message history (which contains the generated SQL).
    """

    run_result: SQLSaberRunResult

    def __new__(cls, content: str, run_result: SQLSaberRunResult) -> Self:
        obj = super().__new__(cls, content)
        obj.run_result = run_result
        return obj

    @property
    def usage(self) -> Any | None:
        """Token usage statistics."""
        usage_attr = getattr(self.run_result, "usage", None)
        if callable(usage_attr):
            return usage_attr()
        return usage_attr

    @property
    def messages(self) -> list[ModelMessage]:
        """All messages from this run, including tool calls (SQL)."""
        return self.run_result.new_messages()

    @property
    def all_messages(self) -> list[ModelMessage]:
        """All messages including history."""
        return self.run_result.all_messages()


class SQLSaber:
    """Main entry point for the SQLSaber Python API.

    Example:
        >>> from sqlsaber import SQLSaber, SQLSaberOptions
        >>> import asyncio
        >>>
        >>> async def main():
        ...     options = SQLSaberOptions(database="sqlite:///my.db")
        ...     async with SQLSaber(options=options) as saber:
        ...         result = await saber.query("Show me the top 5 users")
        ...         print(result)  # Prints the answer
        ...         print(result.usage)  # Prints token usage
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        *,
        options: SQLSaberOptions,
    ):
        """Initialize SQLSaber.

        Args:
            options: Session options bag used to build the SQLSaber session.
        """
        self._session = create_session(options)
        self.db_name = self._session.db_name
        self.agent = self._session.agent
        self.connections = getattr(self._session, "connections", None)

    @property
    def connection(self) -> BaseDatabaseConnection:
        """Return the single active connection, or require .connections for multi-db."""
        connection = getattr(self._session, "connection", None)
        if connection is None:
            raise RuntimeError(
                "This SQLSaber instance has multiple database connections; "
                "use .connections to access them by database id."
            )
        return connection

    async def query(
        self,
        prompt: str,
        message_history: Sequence[ModelMessage] | None = None,
        event_stream_handler: Callable[
            [RunContext[Any], AsyncIterable[AgentStreamEvent]],
            Awaitable[None],
        ]
        | None = None,
    ) -> SQLSaberResult:
        """Run a natural language query against the database.

        Args:
            prompt: The natural language query or instruction.
            message_history: Optional history of messages for context.
            event_stream_handler: Optional streaming handler for AgentStreamEvent.
                Use this to process streaming events as they arrive.

        Returns:
            A SQLSaberResult object (subclass of str) containing the agent's response.
            Access .usage, .messages, etc. for more details.
        """
        result = await self._session.query(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
        )

        content = ""
        if hasattr(result, "data"):
            content = str(result.data)
        elif hasattr(result, "output"):
            content = str(result.output)
        else:
            content = str(result)

        return SQLSaberResult(content, result)

    async def close(self) -> None:
        """Close the database connection."""
        await self._session.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
