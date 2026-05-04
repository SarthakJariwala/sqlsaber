"""Multi-database session lifecycle owner for SQLSaber SDK usage."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.agents.multi_database_agent import (
    DatabaseChild,
    DatabaseDescriptor,
    MultiDatabaseCoordinator,
)
from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database import BaseDatabaseConnection, DatabaseConnection
from sqlsaber.database.resolver import ResolvedDatabase, resolve_databases
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.options import SQLSaberOptions
from sqlsaber.threads.manager import ThreadManager
from sqlsaber.utils.text_input import resolve_text_input

logger = get_logger(__name__)


class MultiDatabaseSession:
    """Owns the lifecycle of a SQLSaber SDK session with multiple databases."""

    def __init__(self, options: SQLSaberOptions):
        self.options = options
        self._closed = False

        database_spec: str | list[str] | None
        if isinstance(options.database, tuple):
            database_spec = list(options.database)
        else:
            database_spec = options.database

        self._resolved = resolve_databases(database_spec)
        if len(self._resolved) < 2:
            raise ValueError("MultiDatabaseSession requires at least two databases.")

        self.db_name = " + ".join(database.name for database in self._resolved)
        self.connections: dict[str, BaseDatabaseConnection] = {
            self._database_id(database): DatabaseConnection(
                database.connection_string,
                excluded_schemas=database.excluded_schemas,
            )
            for database in self._resolved
        }

        self.settings = options.settings or Config.default()
        self.knowledge_manager = options.knowledge_manager or KnowledgeManager()
        self.thread_manager = options.thread_manager or ThreadManager()
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

        children: dict[str, DatabaseChild] = {}
        for database in self._resolved:
            database_id = self._database_id(database)
            child_thread_manager = ThreadManager(storage=self.thread_manager.storage)
            child_agent = SQLSaberAgent(
                self.connections[database_id],
                database.name,
                settings=self.settings,
                knowledge_manager=self.knowledge_manager,
                thinking_enabled=thinking_enabled,
                thinking_level=resolved_thinking_level,
                model_name=options.model_name,
                api_key=options.api_key,
                allow_dangerous=options.allow_dangerous,
                system_prompt=system_prompt_text,
                tool_overides=options.tool_overrides,
                output_type=str,
            )
            descriptor = DatabaseDescriptor(
                id=database_id,
                name=database.name,
                type=database.type,
                description=database.description,
                summary=database.description,
                thread_id=None,
            )
            children[database_id] = DatabaseChild(
                descriptor=descriptor,
                agent=child_agent,
                thread_manager=child_thread_manager,
                database_name=database.name,
            )

        self.children = children
        self.agent = MultiDatabaseCoordinator(
            children=children,
            database_label=self.db_name,
            settings=self.settings,
            thinking_enabled=thinking_enabled,
            thinking_level=resolved_thinking_level,
            model_name=options.model_name,
            api_key=options.api_key,
        )

    async def _ensure_threads(self, prompt: str) -> None:
        """Pre-create parent and child threads before coordinator execution."""
        model_name = self._model_name()
        parent_was_new = self.thread_manager.current_thread_id is None
        parent_thread_id = await self.thread_manager.ensure_thread(
            database_name=self.db_name,
            title=prompt if parent_was_new else None,
            model_name=model_name if parent_was_new else None,
            extra_metadata=json.dumps(
                {"kind": "multi_database_parent", "child_threads": []}
            ),
        )

        child_threads: list[dict[str, str | None]] = []
        for database_id, child in self.children.items():
            child_was_new = child.thread_manager.current_thread_id is None
            child_thread_id = await child.thread_manager.ensure_thread(
                database_name=child.database_name,
                title=f"[{child.database_name}] |-> {prompt}"
                if child_was_new
                else None,
                model_name=model_name if child_was_new else None,
                extra_metadata=json.dumps(
                    {
                        "kind": "multi_database_child",
                        "parent_thread_id": parent_thread_id,
                        "database_id": database_id,
                    }
                ),
            )
            child.descriptor.thread_id = child_thread_id
            child_threads.append(
                {
                    "database_id": database_id,
                    "database_name": child.database_name,
                    "thread_id": child_thread_id,
                }
            )

        await self.thread_manager.ensure_thread(
            database_name=self.db_name,
            extra_metadata=json.dumps(
                {
                    "kind": "multi_database_parent",
                    "child_threads": child_threads,
                }
            ),
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
        """Run a natural language query across configured databases."""
        await self._ensure_threads(prompt)
        run_result = await self.agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
        )

        try:
            await self.thread_manager.save_run(
                run_result=run_result,
                database_name=self.db_name,
                user_query=prompt,
                model_name=self._model_name(),
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

        try:
            await self.thread_manager.end_current_thread()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("sdk.thread.end_failed", error=str(exc))

        for database_id, child in self.children.items():
            try:
                await child.thread_manager.end_current_thread()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                logger.warning(
                    "sdk.thread.child_end_failed",
                    database_id=database_id,
                    error=str(exc),
                )

        try:
            await self.agent.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            errors.append(exc)

        if self._owns_knowledge_manager:
            try:
                await self.knowledge_manager.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                errors.append(exc)

        for connection in self.connections.values():
            try:
                await connection.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                errors.append(exc)

        if errors:
            raise errors[0]

    def _model_name(self) -> str:
        model = getattr(self.agent.agent, "model", None)
        model_name = getattr(model, "model_name", None)
        if isinstance(model_name, str) and model_name:
            return model_name
        return self.settings.model.name

    def _database_id(self, database: ResolvedDatabase) -> str:
        return database.id or database.name
