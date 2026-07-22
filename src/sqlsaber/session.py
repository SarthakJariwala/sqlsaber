"""Session lifecycle owner for SQLSaber SDK usage."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage

from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.logging import get_logger
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database.base import BaseDatabaseConnection
from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.database.resolver import resolve_databases
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.options import SQLSaberOptions
from sqlsaber.threads.metadata import (
    encode_thread_extra_metadata,
    encode_thread_resume_disabled_metadata,
)
from sqlsaber.utils.text_input import resolve_text_input

logger = get_logger(__name__)


class SQLSaberSession:
    """Owns the lifecycle of a SQLSaber SDK session."""

    def __init__(self, options: SQLSaberOptions):
        if options.artifact_failure_mode not in ("required", "best_effort"):
            raise ValueError(
                "artifact_failure_mode must be 'required' or 'best_effort'"
            )
        self.options = options
        self._closed = False

        database_spec: str | list[str] | None
        if isinstance(options.database, tuple):
            database_spec = list(options.database)
        else:
            database_spec = options.database

        self._database_spec = (
            list(database_spec) if isinstance(database_spec, list) else database_spec
        )
        self._resolved = resolve_databases(database_spec)
        self.registry: DatabaseRegistry = DatabaseRegistry.from_resolved(self._resolved)
        self.db_names: list[str] = self.registry.names()
        self.connections: dict[str, BaseDatabaseConnection] = {
            entry.name: entry.connection for entry in self.registry
        }
        # Primary attrs preserved for back-compat with single-DB callers/tests.
        self.db_name = self.registry.primary()
        self.connection = self.connections[self.db_name]

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
            registry=self.registry,
            settings=self.settings,
            knowledge_manager=self.knowledge_manager,
            thinking_enabled=thinking_enabled,
            thinking_level=resolved_thinking_level,
            model_name=options.model_name,
            api_key=options.api_key,
            allow_dangerous=options.allow_dangerous,
            system_prompt=system_prompt_text,
            tool_overides=options.tool_overrides,
            extra_capabilities=options.extra_capabilities,
            artifact_publisher=options.artifact_publisher,
            artifact_failure_mode=options.artifact_failure_mode,
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
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Run a natural language query against the configured database."""
        run_result = await self.agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
            conversation_id=conversation_id,
            metadata=metadata,
        )

        if self.thread_manager is not None:
            try:
                resolved_model_name = getattr(self.agent.agent.model, "model_name", "")
                if not isinstance(resolved_model_name, str) or not resolved_model_name:
                    resolved_model_name = self.agent.config.model.name

                # Keep database_name display-oriented while storing the original
                # selector separately so multi-DB resumes never treat
                # comma-joined names as one database spec.
                if len(self.registry) > 1:
                    database_name = ",".join(self.db_names)
                else:
                    database_name = self.db_name

                database_selector = self._configured_database_selector()
                if database_selector is None:
                    extra_metadata = encode_thread_resume_disabled_metadata(
                        reason=(
                            "Only configured database names are saved for automatic "
                            "resume."
                        )
                    )
                else:
                    extra_metadata = encode_thread_extra_metadata(
                        database_selector=database_selector
                    )

                await self.thread_manager.save_run(
                    run_result=run_result,
                    database_name=database_name,
                    user_query=prompt,
                    model_name=resolved_model_name,
                    extra_metadata=extra_metadata,
                )
            except Exception as exc:
                logger.warning("sdk.thread.save_failed", error=str(exc))

        return run_result

    def _configured_database_selector(self) -> str | list[str] | None:
        """Return a safe, configured selector for resume metadata, if available.

        Raw DSNs and paths are intentionally not persisted so credentials never
        land in thread metadata. Those sessions require an explicit database
        override when resuming.
        """
        if self._database_spec is None:
            return self.db_name

        config_mgr = DatabaseConfigManager()
        if isinstance(self._database_spec, str):
            if config_mgr.get_database(self._database_spec) is None:
                return None
            return self._database_spec

        configured_names: list[str] = []
        for selector in self._database_spec:
            if config_mgr.get_database(selector) is None:
                return None
            configured_names.append(selector)
        return configured_names

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
            await self.registry.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            errors.append(exc)

        if errors:
            raise errors[0]
