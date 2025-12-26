"""Pydantic-AI Agent for SQLSaber.

This replaces the custom AnthropicSQLAgent and uses pydantic-ai's Agent,
function tools, and streaming event types directly.
"""

from collections.abc import AsyncIterable, Awaitable, Sequence
from typing import Any, Callable

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.models import Model

from sqlsaber.agents.provider_factory import ProviderFactory
from sqlsaber.config import providers
from sqlsaber.config.settings import Config
from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.prompts.claude import SONNET_4_5
from sqlsaber.prompts.dangerous_mode import DANGEROUS_MODE
from sqlsaber.prompts.memory import MEMORY_ADDITION
from sqlsaber.prompts.openai import GPT_5
from sqlsaber.tools.registry import tool_registry
from sqlsaber.tools.sql_tools import SQLTool


class SQLSaberAgent:
    """Pydantic-AI Agent wrapper for SQLSaber with enhanced state management."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection,
        database_name: str | None = None,
        memory_manager: MemoryManager | None = None,
        thinking_enabled: bool | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        allow_dangerous: bool = False,
    ):
        self.db_connection = db_connection
        self.database_name = database_name
        self.config = Config()
        self.memory_manager = memory_manager or MemoryManager()
        self._model_name_override = model_name
        self._api_key_override = api_key
        self.db_type = self.db_connection.display_name
        self.allow_dangerous = allow_dangerous

        # Initialize schema manager
        self.schema_manager = SchemaManager(self.db_connection)

        # Thinking configuration (CLI override or config default)
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else self.config.model.thinking_enabled
        )

        # Configure SQL tools with the database connection
        self._configure_sql_tools()

        # Create the pydantic-ai agent
        self.agent = self._build_agent()

    def _configure_sql_tools(self) -> None:
        """Ensure SQL tools receive the active database connection and session config."""
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if isinstance(tool, SQLTool):
                tool.set_connection(self.db_connection, self.schema_manager)
                tool.allow_dangerous = self.allow_dangerous

    def _build_agent(self) -> Agent:
        """Create and configure the pydantic-ai Agent."""
        if self._api_key_override and not self._model_name_override:
            raise ValueError(
                "Model name is required when providing an api_key override."
            )

        # Use override if provided, else fall back to config
        model_name = self._model_name_override or self.config.model.name
        model_name_only = (
            model_name.split(":", 1)[1] if ":" in model_name else model_name
        )

        # Validate/hydrate credentials for the effective model (not stored model)
        if not (self._model_name_override and self._api_key_override):
            self.config.auth.validate(model_name)

        provider = providers.provider_from_model(model_name) or ""

        # API key priority: explicit override > OAuth (if configured) > config
        oauth_token = (
            None
            if self._api_key_override
            else self.config.auth.get_oauth_token(model_name)
        )
        self.is_oauth = provider == "anthropic" and bool(oauth_token)

        if self._api_key_override:
            api_key = self._api_key_override
        elif not self.is_oauth:
            api_key = self.config.auth.get_api_key(model_name)
        else:
            api_key = None

        factory = ProviderFactory()
        agent = factory.create_agent(
            provider=provider,
            model_name=model_name_only,
            full_model_str=model_name,
            api_key=api_key,
            oauth_token=oauth_token,
            thinking_enabled=self.thinking_enabled,
            is_oauth=self.is_oauth,
        )

        self._setup_system_prompt(agent)
        self._register_tools(agent)

        return agent

    def _setup_system_prompt(self, agent: Agent) -> None:
        """Configure the agent's system prompt using a simple prompt string."""
        if not self.is_oauth:

            @agent.system_prompt(dynamic=True)
            async def sqlsaber_system_prompt(ctx: RunContext) -> str:
                if isinstance(agent.model, Model) and "gpt-5" in agent.model.model_name:
                    base = GPT_5.format(db=self.db_type)

                    if self.allow_dangerous:
                        base += DANGEROUS_MODE

                    if self.database_name:
                        mem = self.memory_manager.format_memories_for_prompt(
                            self.database_name
                        )
                        mem = mem.strip()
                        if mem:
                            return f"{base}\n\n{MEMORY_ADDITION}\n\n{mem}"
                    return base
                return self.system_prompt_text(include_memory=True)
        else:

            @agent.system_prompt(dynamic=True)
            async def sqlsaber_system_prompt(ctx: RunContext) -> str:
                # OAuth clients (Claude Code) ignore custom system prompts; we inject later.
                return "You are Claude Code, Anthropic's official CLI for Claude."

    def system_prompt_text(self, include_memory: bool = True) -> str:
        """Return the original SQLSaber system prompt as a single string."""
        base = SONNET_4_5.format(db=self.db_type)

        if self.allow_dangerous:
            base += DANGEROUS_MODE

        if include_memory and self.database_name:
            mem = self.memory_manager.format_memories_for_prompt(self.database_name)
            mem = mem.strip()
            if mem:
                return f"{base}\n\n{MEMORY_ADDITION}\n\n{mem}\n\n"
        return base

    def _register_tools(self, agent: Agent) -> None:
        """Register all the SQL tools with the agent."""
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            # Register the tool dynamically
            # We use the execute method which now has the correct signature
            agent.tool_plain(name=tool.name)(tool.execute)

    def set_thinking(self, enabled: bool) -> None:
        """Update thinking settings and rebuild the agent."""
        self.thinking_enabled = enabled
        # Rebuild agent with new thinking settings
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
        """
        Run the agent with centralized OAuth/system-prompt injection.

        Args:
            prompt: User prompt for this turn.
            message_history: Optional prior messages for context.
            event_stream_handler: Optional streaming handler for AgentStreamEvent.

        Returns:
            The pydantic-ai run result.
        """
        prepared_prompt: str | list[str] = prompt

        if self.is_oauth and not message_history:
            injected = self.system_prompt_text(include_memory=True)
            if injected and str(injected).strip():
                prepared_prompt = [injected, prompt]

        return await self.agent.run(
            prepared_prompt,
            message_history=message_history,
            event_stream_handler=event_stream_handler,
        )
