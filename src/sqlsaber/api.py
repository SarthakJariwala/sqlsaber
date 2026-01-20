"""Public Python API for SQLSaber.

This module provides a simplified programmatic interface to SQLSaber's capabilities,
allowing you to run natural language queries against databases from Python code.
"""

import errno
import os
from collections.abc import AsyncIterable, Awaitable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Protocol, Self

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.database import DatabaseConnection
from sqlsaber.database.resolver import resolve_database


def _resolve_memory_input(memory: str | Path | None) -> str | None:
    """Resolve memory input.

    - If `memory` is a path (or a string path) to an existing file, read it.
    - Otherwise treat it as literal text.

    Passing an empty string disables memory injection.
    """

    if memory is None:
        return None

    if isinstance(memory, str) and not memory.strip():
        # Empty string explicitly disables memory injection.
        # Avoid interpreting it as a filesystem path (Path("") -> ".").
        return ""

    # If it's obviously literal text (contains newlines or is too long to be
    # a valid filename), skip filesystem probing to avoid ENAMETOOLONG errors.
    if isinstance(memory, str):
        if "\n" in memory or "\r" in memory:
            return memory

        seps = {os.sep}
        if os.altsep:
            seps.add(os.altsep)

        has_sep = any(sep in memory for sep in seps)
        # 255 is the max filename component length on most POSIX filesystems
        if not has_sep and len(memory) > 255:
            return memory

    if isinstance(memory, Path):
        candidate = memory.expanduser()
    else:
        candidate = Path(memory).expanduser()

    try:
        if candidate.exists():
            if candidate.is_dir():
                raise ValueError(
                    f"Memory path '{candidate}' is a directory, expected a file"
                )
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        # Treat as literal memory text when path probing fails due to invalid path
        if e.errno in (errno.ENAMETOOLONG, errno.EINVAL):
            return str(memory)
        raise

    return str(memory)


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
        """

        self._config_manager = DatabaseConfigManager()

        database_spec: str | list[str] | None
        if isinstance(database, tuple):
            database_spec = list(database)
        else:
            database_spec = database

        self._resolved = resolve_database(database_spec, self._config_manager)

        self.db_name = self._resolved.name
        self.connection = DatabaseConnection(
            self._resolved.connection_string,
            excluded_schemas=self._resolved.excluded_schemas,
        )

        resolved_thinking_level: ThinkingLevel | None = None
        thinking_enabled = thinking
        if thinking_level is not None:
            if isinstance(thinking_level, str):
                resolved_thinking_level = ThinkingLevel.from_string(thinking_level)
            else:
                resolved_thinking_level = thinking_level
            thinking_enabled = True

        memory_text = _resolve_memory_input(memory)
        self.agent = SQLSaberAgent(
            self.connection,
            self.db_name,
            thinking_enabled=thinking_enabled,
            thinking_level=resolved_thinking_level,
            model_name=model_name,
            api_key=api_key,
            memory=memory_text,
        )

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
        result = await self.agent.run(
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
        await self.connection.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
