"""Session lifecycle owner for SQLSaber SDK usage."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database import DatabaseConnection
from sqlsaber.database.resolver import resolve_database
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.options import SQLSaberOptions
from sqlsaber.utils.text_input import resolve_text_input

logger = get_logger(__name__)


class SQLSaberSession:
    """Owns the lifecycle of a SQLSaber SDK session."""

    def __init__(self, options: SQLSaberOptions):
        self.options = options
        self._closed = False

        database_spec: str | list[str] | None
        if isinstance(options.database, tuple):
            database_spec = list(options.database)
        else:
            database_spec = options.database

        self._resolved = resolve_database(database_spec)
        self.db_name = self._resolved.name
        self.connection = DatabaseConnection(
            self._resolved.connection_string,
            excluded_schemas=self._resolved.excluded_schemas,
        )

        self.settings = options.settings or Config.default()
        self.knowledge_manager = options.knowledge_manager or KnowledgeManager()
        self.thread_manager = options.thread_manager
        self._owns_knowledge_manager = options.knowledge_manager is None

        resolved_thinking_level: ThinkingLevel | None = None
        thinking_enabled = options.thinking_enabled
        if options.thinking_level is not None:
            if isinstance(options.thinking_level, str):
                resolved_thinking_level = ThinkingLevel.from_string(
                    options.thinking_level
                )
            else:
                resolved_thinking_level = options.thinking_level
            thinking_enabled = True

        system_prompt_text = resolve_text_input(options.system_prompt)

        self.agent = SQLSaberAgent(
            self.connection,
            self.db_name,
            settings=self.settings,
            knowledge_manager=self.knowledge_manager,
            thinking_enabled=thinking_enabled,
            thinking_level=resolved_thinking_level,
            model_name=options.model_name,
            api_key=options.api_key,
            allow_dangerous=options.allow_dangerous,
            system_prompt=system_prompt_text,
            tool_overides=options.tool_overrides,
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
    ) -> Any:
        """Run a natural language query against the configured database."""
        run_result = await self.agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
        )

        if self.thread_manager is not None:
            try:
                resolved_model_name = getattr(self.agent.agent.model, "model_name", "")
                if not isinstance(resolved_model_name, str) or not resolved_model_name:
                    resolved_model_name = self.agent.config.model.name

                await self.thread_manager.save_run(
                    run_result=run_result,
                    database_name=self.db_name,
                    user_query=prompt,
                    model_name=resolved_model_name,
                )
            except Exception as exc:
                logger.warning("sdk.thread.save_failed", error=str(exc))

        return run_result

    async def close(self) -> None:
        """Close resources owned by this session."""
        if self._closed:
            return
        self._closed = True

        errors: list[BaseException] = []

        if self.thread_manager is not None:
            try:
                await self.thread_manager.end_current_thread()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning("sdk.thread.end_failed", error=str(exc))

        try:
            await self.agent.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            errors.append(exc)

        if self._owns_knowledge_manager:
            try:
                await self.knowledge_manager.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                errors.append(exc)

        try:
            await self.connection.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            errors.append(exc)

        if errors:
            raise errors[0]
