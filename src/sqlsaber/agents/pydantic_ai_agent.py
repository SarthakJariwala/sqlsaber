"""Managed pydantic-ai agent assembled from SQLSaber capabilities."""

from collections.abc import AsyncIterable, Awaitable, Mapping, Sequence
from typing import Any, Callable

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Thinking
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models.anthropic import AnthropicModelSettings

from sqlsaber.agents.model_factory import UNIFIED_EFFORT_MAP, build_model
from sqlsaber.artifacts import ArtifactFailureMode, ArtifactPublisher
from sqlsaber.capabilities import Knowledge, SqlTools
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext, discover_capabilities
from sqlsaber.config import providers
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.schema import SchemaManager
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.overrides import ToolOveridesInput, normalize_tool_overides
from sqlsaber.prompts.persona import PERSONA
from sqlsaber.tools.base import Tool


def _make_single_db_registry(
    connection: BaseDatabaseConnection, name: str | None
) -> DatabaseRegistry:
    """Build a single-entry registry from an existing connection."""
    entry = DatabaseEntry.from_connection(
        name=name or connection.display_name or "database",
        connection=connection,
        description=None,
        excluded_schemas=list(getattr(connection, "excluded_schemas", []) or []),
    )
    return DatabaseRegistry([entry])


class SQLSaberAgent:
    """Pydantic-AI Agent wrapper for SQLSaber with enhanced state management."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection | None = None,
        database_name: str | None = None,
        *,
        registry: DatabaseRegistry | None = None,
        settings: Config | None = None,
        knowledge_manager: KnowledgeManager | None = None,
        thinking_enabled: bool | None = None,
        thinking_level: ThinkingLevel | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        allow_dangerous: bool = False,
        system_prompt: str | None = None,
        tool_overides: ToolOveridesInput | None = None,
        extra_capabilities: Sequence[AbstractCapability[Any]] = (),
        artifact_publisher: ArtifactPublisher | None = None,
        artifact_failure_mode: ArtifactFailureMode = "required",
    ) -> None:
        if registry is None:
            if db_connection is None:
                raise ValueError(
                    "SQLSaberAgent requires either `registry` or `db_connection`."
                )
            registry = _make_single_db_registry(db_connection, database_name)

        self.registry = registry
        primary = registry.get(registry.primary())
        self.db_connection: BaseDatabaseConnection = primary.connection
        self.database_name = primary.name
        self.config = settings or Config.default()
        self._owns_knowledge_manager = knowledge_manager is None
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        self.system_prompt_override = (
            system_prompt
            if system_prompt is not None and system_prompt.strip()
            else None
        )
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.db_type = self.db_connection.display_name
        self.allow_dangerous = allow_dangerous
        self._tool_overides = normalize_tool_overides(tool_overides)
        self._extra_capabilities = tuple(extra_capabilities)
        self._artifact_publisher = artifact_publisher
        self._artifact_failure_mode = artifact_failure_mode
        self.schema_manager: SchemaManager = primary.schema_manager
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else self.config.model.thinking_enabled
        )
        self.thinking_level = (
            thinking_level
            if thinking_level is not None
            else self.config.model.thinking_level
        )

        self.capabilities: list[AbstractCapability[Any]] = []
        self._tools: dict[str, Tool] = {}
        self._closed = False
        self.agent = self._build_agent()

    @property
    def is_multi_db(self) -> bool:
        return len(self.registry) > 1

    @property
    def display_registry(self) -> Mapping[str, Tool]:
        """Tool implementations used by CLI renderers."""
        return self._tools

    def _build_agent(self) -> Agent:
        if self._api_key_override and not self._model_name_override:
            raise ValueError(
                "Model name is required when providing an api_key override."
            )

        model_name = self._model_name_override or self.config.model.name
        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        provider = providers.provider_from_model(model_name) or ""
        api_key = self._api_key_override or self.config.auth.get_api_key(model_name)
        model = build_model(model_name, api_key)

        include_guidance = self.system_prompt_override is None
        capabilities: list[AbstractCapability[Any]] = [
            Knowledge(
                knowledge_manager=self.knowledge_manager,
                registry=self.registry,
                database_name=self.database_name,
            ),
            SqlTools(
                registry=self.registry,
                allow_dangerous=self.allow_dangerous,
                include_catalog_instructions=include_guidance,
            ),
        ]
        capabilities.extend(
            discover_capabilities(
                PluginContext(
                    registry=self.registry,
                    knowledge_manager=self.knowledge_manager,
                    allow_dangerous=self.allow_dangerous,
                    tool_overrides=self._tool_overides,
                    config=self.config,
                    main_model_name=model_name,
                    main_api_key=api_key,
                    artifact_publisher=self._artifact_publisher,
                    artifact_failure_mode=self._artifact_failure_mode,
                )
            )
        )
        capabilities.extend(self._extra_capabilities)
        if self.thinking_enabled:
            capabilities.append(
                Thinking(effort=UNIFIED_EFFORT_MAP[self.thinking_level])
            )
        self.capabilities = capabilities
        self._tools = {
            name: tool
            for capability in capabilities
            if isinstance(capability, SqlSaberCapability)
            for name, tool in capability.display_specs.items()
        }

        model_settings = (
            AnthropicModelSettings(anthropic_cache=True)
            if provider == "anthropic"
            else None
        )
        return Agent(
            model,
            name="sqlsaber",
            instructions=self.system_prompt_override or PERSONA,
            model_settings=model_settings,
            capabilities=capabilities,
        )

    def system_prompt_text(self) -> str:
        """Return the combined managed-agent instructions as plain text."""
        if self.system_prompt_override is not None:
            return self.system_prompt_override
        sql_tools = next(
            capability
            for capability in self.capabilities
            if isinstance(capability, SqlTools)
        )
        return f"{PERSONA}\n{sql_tools.instructions_text()}"

    def set_thinking(self, enabled: bool, level: ThinkingLevel | None = None) -> None:
        """Update thinking settings and rebuild the agent."""
        self.thinking_enabled = enabled
        if level is not None:
            self.thinking_level = level
        self.agent = self._build_agent()

    async def run(
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
        """Run the agent without occupying the embedding agent's deps slot."""
        return await self.agent.run(
            prompt,
            message_history=message_history,
            conversation_id=conversation_id,
            metadata=metadata,
            event_stream_handler=event_stream_handler,
        )

    async def close(self) -> None:
        """Close capability and wrapper resources without masking cleanup failures."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []

        for capability in self.capabilities:
            if not isinstance(capability, SqlSaberCapability):
                continue
            try:
                await capability.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)

        if self._owns_knowledge_manager:
            try:
                await self.knowledge_manager.close()
            except BaseException as exc:  # pragma: no cover - defensive cleanup
                errors.append(exc)
            finally:
                self._owns_knowledge_manager = False

        if errors:
            primary = errors[0]
            for extra in errors[1:]:
                primary.add_note(f"Additional cleanup failure: {extra!r}")
            raise primary
