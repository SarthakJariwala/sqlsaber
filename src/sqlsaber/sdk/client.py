"""Public stateful Python API for SQLSaber."""

from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from dataclasses import replace
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Callable, Self

from pydantic_ai import RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimits

from ._runtime import _SQLSaberRuntime
from sqlsaber.artifact_resolution import artifact_references_from_messages
from sqlsaber.artifacts import (
    ArtifactContext,
    ArtifactStore,
    ArtifactUnavailable,
    LoadedArtifact,
    StoredArtifact,
    validate_loaded_artifact,
)
from sqlsaber.config.database import DatabaseConfigManager
from sqlsaber.config.settings import ThinkingLevel
from sqlsaber.database.base import BaseDatabaseConnection
from sqlsaber.database.registry import DatabaseRegistry
from .options import SQLSaberOptions
from sqlsaber.query_result_resolution import query_result_references_from_messages
from sqlsaber.query_results import (
    LoadedQueryResult,
    QueryResultContext,
    QueryResultStore,
    StoredQueryResult,
)
from .errors import (
    RunInProgressError,
    SQLSaberClosedError,
    ThreadDatabaseRequiredError,
    ThreadDatabaseUnavailableError,
    ThreadNotFoundError,
    ThreadResumeHistoryError,
)
from .types import SQLSaberInfo, TableInfo, ThinkingState
from sqlsaber.threads.manager import ThreadManager
from sqlsaber.threads.metadata import resolve_thread_database_selector
from sqlsaber.threads.storage import ThreadStorage
from sqlsaber.tools.base import Tool

if TYPE_CHECKING:
    from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent


def _result_messages(run_result: Any, method_name: str) -> list[ModelMessage]:
    method = getattr(run_result, method_name, None)
    if not callable(method):
        return []
    messages = method()
    return list(messages) if messages is not None else []


def _result_usage(run_result: Any) -> RunUsage | None:
    usage = getattr(run_result, "usage", None)
    return usage() if callable(usage) else usage


def _request_usages(messages: Sequence[ModelMessage]) -> list[RequestUsage]:
    return [message.usage for message in messages if isinstance(message, ModelResponse)]


def _final_context_tokens(
    run_result: Any, request_usages: Sequence[RequestUsage]
) -> int:
    response = getattr(run_result, "response", None)
    response_usage = getattr(response, "usage", None)
    tokens = getattr(response_usage, "input_tokens", None)
    if isinstance(tokens, int):
        return tokens
    if request_usages:
        return request_usages[-1].input_tokens
    return 0


class SQLSaberResult(str):
    """Text response together with a complete snapshot of one query run."""

    run_result: Any
    _usage: RunUsage | None
    _request_usages: tuple[RequestUsage, ...]
    _final_context_tokens: int
    _new_messages: tuple[ModelMessage, ...]
    _all_messages: tuple[ModelMessage, ...]
    _query_results: tuple[StoredQueryResult, ...]
    _artifacts: tuple[StoredArtifact, ...]

    def __new__(cls, content: str, run_result: Any) -> Self:
        obj = super().__new__(cls, content)
        new_messages = _result_messages(run_result, "new_messages")
        all_messages = _result_messages(run_result, "all_messages")
        request_usages = _request_usages(new_messages)

        obj.run_result = run_result
        obj._usage = _result_usage(run_result)
        obj._request_usages = tuple(request_usages)
        obj._final_context_tokens = _final_context_tokens(run_result, request_usages)
        obj._new_messages = tuple(new_messages)
        obj._all_messages = tuple(all_messages)
        obj._query_results = tuple(
            reference.descriptor
            for reference in query_result_references_from_messages(new_messages)
            if reference.descriptor is not None
        )
        obj._artifacts = tuple(
            artifact
            for reference in artifact_references_from_messages(new_messages)
            for artifact in reference.artifacts
        )
        return obj

    @property
    def text(self) -> str:
        """Agent response text."""
        return str(self)

    @property
    def usage(self) -> RunUsage | None:
        """Aggregate token usage for the run."""
        return self._usage

    @property
    def request_usages(self) -> list[RequestUsage]:
        """Token usage for each model request in execution order."""
        return list(self._request_usages)

    @property
    def final_context_tokens(self) -> int:
        """Input tokens used by the final model request."""
        return self._final_context_tokens

    @property
    def new_messages(self) -> list[ModelMessage]:
        """Messages created by this query."""
        return list(self._new_messages)

    @property
    def messages(self) -> list[ModelMessage]:
        """Compatibility alias for messages created by this query."""
        return self.new_messages

    @property
    def all_messages(self) -> list[ModelMessage]:
        """Complete conversation history after this query."""
        return list(self._all_messages)

    @property
    def query_results(self) -> list[StoredQueryResult]:
        """Durable query results created during this query."""
        return list(self._query_results)

    @property
    def artifacts(self) -> list[StoredArtifact]:
        """Durable artifacts created during this query."""
        return list(self._artifacts)


class SQLSaber:
    """Canonical lifecycle and conversation owner for SQLSaber clients."""

    def __init__(self, *, options: SQLSaberOptions):
        self._runtime = _SQLSaberRuntime(options)
        self._message_history: list[ModelMessage] = []
        thread_manager = self._runtime.thread_manager
        self._is_new_thread = bool(getattr(thread_manager, "first_message", True))
        self._query_in_progress = False
        self._closed = False

    @classmethod
    async def resume(
        cls,
        thread_id: str,
        *,
        options: SQLSaberOptions,
        storage: ThreadStorage | None = None,
    ) -> Self:
        """Resume a stored thread without trusting persisted connection strings."""
        thread_storage = storage or ThreadStorage()
        thread = await thread_storage.get_thread(thread_id)
        if thread is None:
            raise ThreadNotFoundError(thread_id)

        database_selector = options.database
        if database_selector is None:
            try:
                database_selector = resolve_thread_database_selector(
                    database_name=thread.database_name,
                    extra_metadata=thread.extra_metadata,
                )
            except ValueError as exc:
                raise ThreadDatabaseRequiredError(thread_id, str(exc)) from exc

            if database_selector is None:
                raise ThreadDatabaseRequiredError(
                    thread_id,
                    "No configured database selector is stored for this thread.",
                )

            selectors = (
                [database_selector]
                if isinstance(database_selector, str)
                else database_selector
            )
            if len(set(selectors)) != len(selectors):
                raise ThreadDatabaseRequiredError(
                    thread_id,
                    "Stored database selectors must be unique.",
                )

            config_manager = DatabaseConfigManager()
            missing = tuple(
                selector
                for selector in selectors
                if config_manager.get_database(selector) is None
            )
            if missing:
                raise ThreadDatabaseUnavailableError(thread_id, missing)

        try:
            history = await thread_storage.get_thread_messages_strict(thread_id)
        except Exception as exc:
            raise ThreadResumeHistoryError(thread_id, str(exc)) from exc

        thread_manager = ThreadManager(
            initial_thread_id=thread_id,
            storage=thread_storage,
        )
        resumed_options = replace(
            options,
            database=database_selector,
            thread_manager=thread_manager,
        )
        saber = cls(options=resumed_options)
        saber._message_history = list(history)
        saber._is_new_thread = False
        return saber

    @property
    def info(self) -> SQLSaberInfo:
        """Return immutable metadata without exposing managed agent internals."""
        agent = self._runtime.agent
        model = agent.agent.model
        model_name = getattr(model, "model_name", agent.config.model.name)
        model_id = getattr(model, "model_id", None)
        thread_manager = self._runtime.thread_manager
        thread_id = (
            getattr(thread_manager, "current_thread_id", None)
            if thread_manager is not None
            else None
        )
        return SQLSaberInfo(
            database_names=tuple(self._runtime.db_names),
            primary_database_name=self._runtime.db_name,
            primary_database_type=self._runtime.connection.display_name,
            model_name=str(model_name),
            model_id=str(model_id) if model_id is not None else None,
            thinking=ThinkingState(
                enabled=agent.thinking_enabled,
                level=agent.thinking_level,
            ),
            dangerous_mode=agent.allow_dangerous,
            thread_id=str(thread_id) if thread_id is not None else None,
            is_new_thread=self._is_new_thread,
        )

    @property
    def display_registry(self) -> Mapping[str, Tool]:
        """Read-only display adapters for streaming renderers."""
        return MappingProxyType(dict(self._runtime.agent.display_registry))

    @property
    def artifact_store(self) -> ArtifactStore | None:
        return self._runtime.artifact_store

    @property
    def query_result_store(self) -> QueryResultStore:
        return self._runtime.query_result_store

    @property
    def registry(self) -> DatabaseRegistry:
        """Compatibility access to the managed database registry."""
        return self._runtime.registry

    @property
    def db_names(self) -> list[str]:
        """Compatibility access to configured database names."""
        return self._runtime.db_names

    @property
    def connections(self) -> dict[str, BaseDatabaseConnection]:
        """Compatibility access to managed database connections."""
        return self._runtime.connections

    @property
    def db_name(self) -> str:
        """Compatibility access to the primary database name."""
        return self._runtime.db_name

    @property
    def connection(self) -> BaseDatabaseConnection:
        """Compatibility access to the primary database connection."""
        return self._runtime.connection

    @property
    def agent(self) -> SQLSaberAgent:
        """Compatibility access to the managed agent for embedded callers."""
        return self._runtime.agent

    def _ensure_open(self) -> None:
        if self._closed:
            raise SQLSaberClosedError("SQLSaber is closed.")

    def _ensure_not_running(self) -> None:
        self._ensure_open()
        if self._query_in_progress:
            raise RunInProgressError("A SQLSaber query is already running.")

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
        usage_limits: UsageLimits | None = None,
    ) -> SQLSaberResult:
        """Run a query and commit its completed history to this SDK instance."""
        self._ensure_not_running()
        self._query_in_progress = True
        try:
            history = (
                list(message_history)
                if message_history is not None
                else list(self._message_history)
            )
            run_result = await self._runtime.query(
                prompt,
                message_history=history,
                event_stream_handler=event_stream_handler,
                conversation_id=conversation_id,
                metadata=metadata,
                usage_limits=usage_limits,
            )
            content = getattr(run_result, "data", None)
            if content is None:
                content = getattr(run_result, "output", run_result)
            result = SQLSaberResult(str(content), run_result)
            self._message_history = result.all_messages
            self._is_new_thread = False
            return result
        finally:
            self._query_in_progress = False

    def set_thinking(
        self,
        *,
        enabled: bool,
        level: ThinkingLevel | None = None,
    ) -> ThinkingState:
        """Update reasoning controls for subsequent queries."""
        self._ensure_not_running()
        agent = self._runtime.agent
        agent.set_thinking(enabled, level)
        return ThinkingState(
            enabled=agent.thinking_enabled,
            level=agent.thinking_level,
        )

    async def list_tables(self) -> tuple[TableInfo, ...]:
        """List tables across all managed databases in registry order."""
        self._ensure_open()
        tables: list[TableInfo] = []
        for entry in self._runtime.registry:
            payload = await entry.schema_manager.list_tables()
            raw_tables = payload.get("tables", [])
            if not isinstance(raw_tables, list):
                continue
            for raw_table in raw_tables:
                if not isinstance(raw_table, dict):
                    continue
                name = str(raw_table.get("name") or raw_table.get("table_name") or "")
                if not name:
                    continue
                schema_name = str(
                    raw_table.get("schema") or raw_table.get("table_schema") or ""
                )
                qualified_name = str(raw_table.get("full_name") or "")
                if not qualified_name:
                    qualified_name = f"{schema_name}.{name}" if schema_name else name
                tables.append(
                    TableInfo(
                        database_name=entry.name,
                        schema_name=schema_name,
                        name=name,
                        kind=str(
                            raw_table.get("type")
                            or raw_table.get("table_type")
                            or "table"
                        ),
                        qualified_name=qualified_name,
                        completion_name=qualified_name,
                    )
                )
        return tuple(tables)

    async def draft_handoff(self, goal: str) -> str:
        """Draft a handoff from the SDK-owned conversation history."""
        self._ensure_not_running()
        from sqlsaber.agents.handoff_agent import HandoffAgent

        handoff_agent = HandoffAgent()
        return await handoff_agent.generate_draft(
            message_history=list(self._message_history),
            goal=goal,
        )

    async def end_thread(self) -> str | None:
        """Mark the current persisted thread as ended."""
        self._ensure_not_running()
        thread_manager = self._runtime.thread_manager
        if thread_manager is None:
            return None
        return await thread_manager.end_current_thread()

    async def new_thread(self) -> str | None:
        """End the current thread and clear SDK-owned conversation history."""
        self._ensure_not_running()
        thread_manager = self._runtime.thread_manager
        previous_id = (
            thread_manager.current_thread_id if thread_manager is not None else None
        )
        if thread_manager is not None:
            await thread_manager.clear_current_thread()
        self._message_history.clear()
        self._is_new_thread = True
        return previous_id

    async def get_artifact(
        self,
        artifact: str | StoredArtifact,
        *,
        conversation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LoadedArtifact:
        """Retrieve an artifact through the configured authorized store."""
        self._ensure_open()
        if self.artifact_store is None:
            raise ArtifactUnavailable("No artifact store is configured.")
        if isinstance(artifact, StoredArtifact):
            expected = artifact
            artifact_id = artifact.id
        else:
            expected = None
            artifact_id = artifact
        loaded = await self.artifact_store.get(
            artifact_id,
            context=ArtifactContext(
                conversation_id=conversation_id,
                metadata=metadata or {},
            ),
        )
        return validate_loaded_artifact(loaded, expected=expected)

    async def get_query_result(
        self,
        result: str | StoredQueryResult,
        *,
        conversation_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LoadedQueryResult:
        """Retrieve a complete query result through the configured store."""
        self._ensure_open()
        result_id = result.id if isinstance(result, StoredQueryResult) else result
        return await self.query_result_store.get(
            result_id,
            context=QueryResultContext(
                conversation_id=conversation_id,
                metadata=metadata or {},
            ),
        )

    async def close(self) -> None:
        """Close resources owned by this SDK instance."""
        if self._closed:
            return
        if self._query_in_progress:
            raise RunInProgressError("Cannot close SQLSaber while a query is running.")
        self._closed = True
        await self._runtime.close()

    async def __aenter__(self) -> Self:
        self._ensure_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
