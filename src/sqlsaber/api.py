"""Public Python API for SQLSaber.

This module provides a simplified programmatic interface to SQLSaber's capabilities,
allowing you to run natural language queries against databases from Python code.
"""

from collections.abc import AsyncIterable, Awaitable, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Protocol, Self

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.options import SQLSaberOptions
from sqlsaber.overrides import ToolOveridesInput
from sqlsaber.session import SQLSaberSession

if TYPE_CHECKING:
    from sqlsaber.knowledge.manager import KnowledgeManager


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
        >>> from sqlsaber import SQLSaber
        >>> import asyncio
        >>>
        >>> async def main():
        ...     async with SQLSaber(database="sqlite:///my.db") as saber:
        ...         result = await saber.query("Show me the top 5 users")
        ...         print(result)  # Prints the answer
        ...         print(result.usage)  # Prints token usage
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        database: str | list[str] | tuple[str, ...] | None = None,
        thinking: bool = False,
        thinking_level: ThinkingLevel | str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        memory: str | Path | None = None,
        system_prompt: str | Path | None = None,
        tool_overrides: ToolOveridesInput | None = None,
        knowledge_manager: "KnowledgeManager | None" = None,
        *,
        options: SQLSaberOptions | None = None,
    ):
        """Initialize SQLSaber.

        Args:
            database: Database connection string, name, or file path.
                If None, uses the default configured database.

                You can also pass multiple CSVs by providing a list/tuple of CSV
                file paths or CSV connection strings. Each CSV becomes its own
                DuckDB view (named after the file stem).

                Examples:
                - "postgresql://user:pass@localhost/db"
                - "sqlite:///data.db"
                - "my-saved-db"
                - ["users.csv", "orders.csv"]
                - ("csv:///users.csv", "csv:///orders.csv")
            thinking: Whether to enable "thinking" mode for supported models.
                Automatically set to True if thinking_level is provided.
            thinking_level: The thinking level for extended thinking models.
                Can be a ThinkingLevel enum value or a string ("minimal", "low",
                "medium", "high", "maximum"). If provided, thinking is automatically
                enabled. Defaults to the configured level (or "medium" if unset).
            model_name: Override model (format: 'provider:model',
                e.g., 'anthropic:claude-sonnet-4-20250514').
            api_key: Override API key for the model provider.
            memory: Optional extra context to inject into the system prompt.
                If this points to an existing file path, its contents are read.
                If provided (even as an empty string), it overrides any saved
                database memories for this session.
            system_prompt: Custom system prompt text to replace SQLSaber's default.
                If this points to an existing file path, its contents are read.
            tool_overrides: Optional runtime model/api-key overrides per tool name.
                Example:
                {"viz": ModelOverides(model_name="openai:gpt-5-mini")}
            knowledge_manager: Optional knowledge manager dependency to use for
                knowledge tool operations.
            options: Optional session options bag. If provided, this takes
                precedence over all other constructor arguments.
        """
        resolved_options = options
        if resolved_options is None:
            resolved_options = SQLSaberOptions(
                database=database,
                model_name=model_name,
                api_key=api_key,
                thinking_enabled=thinking,
                thinking_level=thinking_level,
                memory=memory,
                system_prompt=system_prompt,
                tool_overrides=tool_overrides,
                knowledge_manager=knowledge_manager,
            )

        self._session = SQLSaberSession(resolved_options)
        self.db_name = self._session.db_name
        self.connection = self._session.connection
        self.agent = self._session.agent

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
