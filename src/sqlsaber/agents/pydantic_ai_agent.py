"""Pydantic-AI Agent for SQLSaber.

This replaces the custom AnthropicSQLAgent and uses pydantic-ai's Agent,
function tools, and streaming event types directly.
"""

from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable, cast

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models import Model

from sqlsaber.agents.provider_factory import ProviderFactory
from sqlsaber.config import providers
from sqlsaber.config.settings import Config, ThinkingLevel
from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.overrides import (
    ToolRunDeps,
    ToolOveridesInput,
    build_tool_run_deps,
    normalize_tool_overides,
)
from sqlsaber.prompts.claude import SONNET_4_5
from sqlsaber.prompts.dangerous_mode import DANGEROUS_MODE
from sqlsaber.prompts.memory import MEMORY_ADDITION
from sqlsaber.prompts.openai import GPT_5
from sqlsaber.tools.base import Tool
from sqlsaber.tools.knowledge_tool import KnowledgeTool
from sqlsaber.tools.registry import tool_registry
from sqlsaber.tools.sql_tools import SQLTool


class SQLSaberAgent:
    """Pydantic-AI Agent wrapper for SQLSaber with enhanced state management."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection,
        database_name: str | None = None,
        memory_manager: MemoryManager | None = None,
        knowledge_manager: KnowledgeManager | None = None,
        thinking_enabled: bool | None = None,
        thinking_level: ThinkingLevel | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        allow_dangerous: bool = False,
        memory: str | None = None,
        system_prompt: str | None = None,
        tool_overides: ToolOveridesInput | None = None,
    ):
        self.db_connection = db_connection
        self.database_name = database_name
        self.config = Config()
        self.memory_manager = memory_manager or MemoryManager()
        self._owns_knowledge_manager = knowledge_manager is None
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        self.memory_override = memory
        if system_prompt is not None and not system_prompt.strip():
            self.system_prompt_override = None
        else:
            self.system_prompt_override = system_prompt
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.db_type = self.db_connection.display_name
        self.allow_dangerous = allow_dangerous
        self._tool_overides = normalize_tool_overides(tool_overides)

        self.schema_manager = SchemaManager(self.db_connection)

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

        self._tools: dict[str, Tool] = {
            name: tool_registry.create_tool(name) for name in tool_registry.list_tools()
        }

        self._configure_sql_tools()
        self._configure_knowledge_tools()
        self.agent = self._build_agent()

    def _configure_sql_tools(self) -> None:
        """Ensure SQL tools receive the active database connection and session config."""
        for tool in self._tools.values():
            if isinstance(tool, SQLTool):
                tool.set_connection(self.db_connection, self.schema_manager)
                tool.allow_dangerous = self.allow_dangerous

    def _configure_knowledge_tools(self) -> None:
        """Ensure knowledge tools receive database and manager context."""
        for tool in self._tools.values():
            if isinstance(tool, KnowledgeTool):
                tool.set_context(self.database_name, self.knowledge_manager)

    def _build_agent(self) -> Agent:
        """Create and configure the pydantic-ai Agent."""
        if self._api_key_override and not self._model_name_override:
            raise ValueError(
                "Model name is required when providing an api_key override."
            )

        model_name = self._model_name_override or self.config.model.name
        model_name_only = (
            model_name.split(":", 1)[1] if ":" in model_name else model_name
        )

        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        provider = providers.provider_from_model(model_name) or ""

        api_key = self._api_key_override or self.config.auth.get_api_key(model_name)

        factory = ProviderFactory()
        agent = factory.create_agent(
            provider=provider,
            model_name=model_name_only,
            full_model_str=model_name,
            api_key=api_key,
            thinking_enabled=self.thinking_enabled,
            thinking_level=self.thinking_level,
        )

        self._setup_system_prompt(agent)
        self._register_tools(agent)
        return agent

    def _prompt_memory_text(self, include_memory: bool = True) -> str | None:
        if not include_memory:
            return None

        if self.memory_override is not None:
            mem = self.memory_override.strip()
            return mem or None

        if self.database_name:
            mem = self.memory_manager.format_memories_for_prompt(self.database_name)
            mem = mem.strip()
            return mem or None

        return None

    def _setup_system_prompt(self, agent: Agent) -> None:
        """Configure the agent's system prompt using a simple prompt string."""

        @agent.system_prompt(dynamic=True)
        async def sqlsaber_system_prompt(ctx: RunContext) -> str:
            use_gpt5 = (
                isinstance(agent.model, Model) and "gpt-5" in agent.model.model_name
            )
            base = self._base_system_prompt(use_gpt5=use_gpt5)
            return self._apply_prompt_extras(base, include_memory=True)

    def _base_system_prompt(self, *, use_gpt5: bool) -> str:
        if self.system_prompt_override is not None:
            return self.system_prompt_override
        if use_gpt5:
            return GPT_5.format(db=self.db_type)
        return SONNET_4_5.format(db=self.db_type)

    def _apply_prompt_extras(self, base: str, *, include_memory: bool) -> str:
        prompt = base
        if self.allow_dangerous:
            prompt += DANGEROUS_MODE

        mem = self._prompt_memory_text(include_memory=include_memory)
        if mem:
            prompt = f"{prompt}\n\n{MEMORY_ADDITION}\n\n{mem}"
        return prompt

    def system_prompt_text(self, include_memory: bool = True) -> str:
        """Return the SQLSaber system prompt as a single string."""
        base = self._base_system_prompt(use_gpt5=False)
        return self._apply_prompt_extras(base, include_memory=include_memory)

    def _register_tools(self, agent: Agent) -> None:
        """Register all the SQL tools with the agent."""
        for tool in self._tools.values():
            register = agent.tool if tool.requires_ctx else agent.tool_plain
            register(name=tool.name)(tool.execute)

    def set_thinking(self, enabled: bool, level: ThinkingLevel | None = None) -> None:
        """Update thinking settings and rebuild the agent.

        Args:
            enabled: Whether thinking is enabled.
            level: Optional thinking level to set. If not provided, keeps current level.
        """
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
    ) -> Any:
        """Run the agent."""
        run_agent = cast(Agent[ToolRunDeps, Any], self.agent)
        return await run_agent.run(
            prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
            deps=build_tool_run_deps(self._tool_overides),
        )

    async def close(self) -> None:
        """Close resources owned by the agent."""
        if self._owns_knowledge_manager:
            await self.knowledge_manager.close()
            self._owns_knowledge_manager = False
