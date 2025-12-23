"""
Public Python API for SQLSaber.

This module provides a simplified programmatic interface to SQLSaber's capabilities,
allowing you to run natural language queries against databases from Python code.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, Self

from pydantic_ai.messages import ModelMessage

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.database import DatabaseConnection
from sqlsaber.database.resolver import resolve_database


class SQLSaberRunResult(Protocol):
    """Protocol for pydantic-ai run result objects."""

    def usage(self) -> Any: ...
    def new_messages(self) -> list[ModelMessage]: ...
    def all_messages(self) -> list[ModelMessage]: ...


class SQLSaberResult(str):
    """
    Result of a SQLSaber query.

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
    """
    Main entry point for the SQLSaber Python API.

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
        database: str | None = None,
        thinking: bool = False,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize SQLSaber.

        Args:
            database: Database connection string, name, or file path.
                      If None, uses the default configured database.
                      Examples:
                      - "postgresql://user:pass@localhost/db"
                      - "sqlite:///data.db"
                      - "my-saved-db"
            thinking: Whether to enable "thinking" mode for supported models.
            model_name: Override model (format: 'provider:model',
                        e.g., 'anthropic:claude-sonnet-4-20250514').
            api_key: Override API key for the model provider.
        """
        self._config_manager = DatabaseConfigManager()
        self._resolved = resolve_database(database, self._config_manager)

        self.db_name = self._resolved.name
        self.connection = DatabaseConnection(
            self._resolved.connection_string,
            excluded_schemas=self._resolved.excluded_schemas,
        )
        self.agent = SQLSaberAgent(
            self.connection,
            self.db_name,
            thinking_enabled=thinking,
            model_name=model_name,
            api_key=api_key,
        )

    async def query(
        self, prompt: str, message_history: list[ModelMessage] | None = None
    ) -> SQLSaberResult:
        """
        Run a natural language query against the database.

        Args:
            prompt: The natural language query or instruction.
            message_history: Optional history of messages for context.

        Returns:
            A SQLSaberResult object (subclass of str) containing the agent's response.
            Access .usage, .messages, etc. for more details.
        """
        # Handle OAuth injection if needed
        prepared_prompt: str | list[str] = prompt

        # If we have no history and using OAuth (Claude Code), we need to inject context
        if self.agent.is_oauth and not message_history:
            injected = self.agent.system_prompt_text(include_memory=True)
            if injected and str(injected).strip():
                prepared_prompt = [injected, prompt]

        result = await self.agent.agent.run(
            prepared_prompt, message_history=message_history
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
