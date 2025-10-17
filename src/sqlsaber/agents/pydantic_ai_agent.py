"""Pydantic-AI Agent for SQLSaber.

This replaces the custom AnthropicSQLAgent and uses pydantic-ai's Agent,
function tools, and streaming event types directly.
"""

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.models.groq import GroqModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider

from sqlsaber.config import providers
from sqlsaber.config.settings import Config
from sqlsaber.database import (
    BaseDatabaseConnection,
    CSVConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.prompts.claude import SONNET_4_5
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
    ):
        self.db_connection = db_connection
        self.database_name = database_name
        self.config = Config()
        self.memory_manager = memory_manager or MemoryManager()
        self.db_type = self._get_database_type_name()

        # Thinking configuration (CLI override or config default)
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else self.config.thinking_enabled
        )

        # Configure SQL tools with the database connection
        self._configure_sql_tools()

        # Create the pydantic-ai agent
        self.agent = self._build_agent()

    def _configure_sql_tools(self) -> None:
        """Ensure SQL tools receive the active database connection."""
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if isinstance(tool, SQLTool):
                tool.set_connection(self.db_connection)

    def _build_agent(self) -> Agent:
        """Create and configure the pydantic-ai Agent."""
        self.config.validate()

        model_name_only = (
            self.config.model_name.split(":", 1)[1]
            if ":" in self.config.model_name
            else self.config.model_name
        )

        provider = providers.provider_from_model(self.config.model_name) or ""
        self.is_oauth = provider == "anthropic" and bool(
            getattr(self.config, "oauth_token", None)
        )

        agent = self._create_agent_for_provider(provider, model_name_only)
        self._setup_system_prompt(agent)
        self._register_tools(agent)

        return agent

    def _create_agent_for_provider(self, provider: str, model_name: str) -> Agent:
        """Create the agent based on the provider type."""
        if provider == "google":
            model_obj = GoogleModel(
                model_name, provider=GoogleProvider(api_key=self.config.api_key)
            )
            if self.thinking_enabled:
                settings = GoogleModelSettings(
                    google_thinking_config={"include_thoughts": True}
                )
                return Agent(model_obj, name="sqlsaber", model_settings=settings)
            return Agent(model_obj, name="sqlsaber")
        elif provider == "anthropic" and self.is_oauth:
            return self._create_oauth_anthropic_agent(model_name)
        elif provider == "anthropic":
            if self.thinking_enabled:
                settings = AnthropicModelSettings(
                    anthropic_thinking={
                        "type": "enabled",
                        "budget_tokens": 2048,
                    },
                    max_tokens=8192,
                )
                return Agent(
                    self.config.model_name, name="sqlsaber", model_settings=settings
                )
            return Agent(self.config.model_name, name="sqlsaber")
        elif provider == "openai":
            model_obj = OpenAIResponsesModel(model_name)
            if self.thinking_enabled:
                settings = OpenAIResponsesModelSettings(
                    openai_reasoning_effort="medium",
                    openai_reasoning_summary="auto",
                )
                return Agent(model_obj, name="sqlsaber", model_settings=settings)
            return Agent(model_obj, name="sqlsaber")
        elif provider == "groq":
            if self.thinking_enabled:
                settings = GroqModelSettings(groq_reasoning_format="parsed")
                return Agent(
                    self.config.model_name, name="sqlsaber", model_settings=settings
                )
            return Agent(self.config.model_name, name="sqlsaber")
        else:
            return Agent(self.config.model_name, name="sqlsaber")

    def _create_oauth_anthropic_agent(self, model_name: str) -> Agent:
        """Create an Anthropic agent with OAuth configuration."""

        async def add_oauth_headers(request: httpx.Request) -> None:  # type: ignore[override]
            if "x-api-key" in request.headers:
                del request.headers["x-api-key"]
            request.headers.update(
                {
                    "Authorization": f"Bearer {self.config.oauth_token}",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": "ClaudeCode/1.0 (Anthropic Claude Code CLI)",
                    "X-Client-Name": "claude-code",
                    "X-Client-Version": "1.0.0",
                }
            )

        http_client = httpx.AsyncClient(event_hooks={"request": [add_oauth_headers]})
        provider_obj = AnthropicProvider(api_key="placeholder", http_client=http_client)
        model_obj = AnthropicModel(model_name, provider=provider_obj)
        if self.thinking_enabled:
            settings = AnthropicModelSettings(
                anthropic_thinking={
                    "type": "enabled",
                    "budget_tokens": 2048,
                },
                max_tokens=8192,
            )
            return Agent(model_obj, name="sqlsaber", model_settings=settings)
        return Agent(model_obj, name="sqlsaber")

    def _setup_system_prompt(self, agent: Agent) -> None:
        """Configure the agent's system prompt using a simple prompt string."""
        if not self.is_oauth:

            @agent.system_prompt(dynamic=True)
            async def sqlsaber_system_prompt(ctx: RunContext) -> str:
                if "gpt-5" in agent.model.model_name:
                    base = GPT_5.format(db=self.db_type)

                    if self.database_name:
                        mem = self.memory_manager.format_memories_for_prompt(
                            self.database_name
                        )
                        mem = mem.strip()
                        if mem:
                            return f"{base}\n\n{MEMORY_ADDITION}\n\n{mem}"
                return self.system_prompt_text(include_memory=True)
        else:

            @agent.system_prompt(dynamic=True)
            async def sqlsaber_system_prompt(ctx: RunContext) -> str:
                # OAuth clients (Claude Code) ignore custom system prompts; we inject later.
                return "You are Claude Code, Anthropic's official CLI for Claude."

    def system_prompt_text(self, include_memory: bool = True) -> str:
        """Return the original SQLSaber system prompt as a single string."""
        base = SONNET_4_5.format(db=self.db_type)

        if include_memory and self.database_name:
            mem = self.memory_manager.format_memories_for_prompt(self.database_name)
            mem = mem.strip()
            if mem:
                return f"{base}\n\n{MEMORY_ADDITION}\n\n{mem}\n\n"
        return base

    def _register_tools(self, agent: Agent) -> None:
        """Register all the SQL tools with the agent."""

        @agent.tool(name="list_tables")
        async def list_tables(ctx: RunContext) -> str:
            """
            Get a list of all tables in the database with row counts.
            Use this first to discover available tables.
            """
            tool = tool_registry.get_tool("list_tables")
            return await tool.execute()

        @agent.tool(name="introspect_schema")
        async def introspect_schema(
            ctx: RunContext, table_pattern: str | None = None
        ) -> str:
            """
            Introspect database schema to understand table structures.

            Args:
                table_pattern: Optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%')
            """
            tool = tool_registry.get_tool("introspect_schema")
            return await tool.execute(table_pattern=table_pattern)

        @agent.tool(name="execute_sql")
        async def execute_sql(
            ctx: RunContext, query: str, limit: int | None = 100
        ) -> str:
            """
            Execute a SQL query and return the results.

            Args:
                query: SQL query to execute
                limit: Maximum number of rows to return (default: 100)
            """
            tool = tool_registry.get_tool("execute_sql")
            return await tool.execute(query=query, limit=limit)

    def set_thinking(self, enabled: bool) -> None:
        """Update thinking settings and rebuild the agent."""
        self.thinking_enabled = enabled
        # Rebuild agent with new thinking settings
        self.agent = self._build_agent()

    def _get_database_type_name(self) -> str:
        """Get the human-readable database type name."""
        if isinstance(self.db_connection, PostgreSQLConnection):
            return "PostgreSQL"
        elif isinstance(self.db_connection, MySQLConnection):
            return "MySQL"
        elif isinstance(self.db_connection, SQLiteConnection):
            return "SQLite"
        elif isinstance(self.db_connection, DuckDBConnection):
            return "DuckDB"
        elif isinstance(self.db_connection, CSVConnection):
            return "DuckDB"
        else:
            return "database"
