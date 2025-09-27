"""Pydantic-AI Agent for SQLSaber.

This replaces the custom AnthropicSQLAgent and uses pydantic-ai's Agent,
function tools, and streaming event types directly.
"""

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider

from sqlsaber.config import providers
from sqlsaber.config.settings import Config
from sqlsaber.database.connection import (
    BaseDatabaseConnection,
    CSVConnection,
    DuckDBConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.memory.manager import MemoryManager
from sqlsaber.tools.instructions import InstructionBuilder
from sqlsaber.tools.registry import tool_registry
from sqlsaber.tools.sql_tools import SQLTool


def build_sqlsaber_agent(
    db_connection: BaseDatabaseConnection,
    database_name: str | None,
) -> Agent:
    """Create and configure a pydantic-ai Agent for SQLSaber.

    - Registers function tools that delegate to the existing tool registry
    - Attaches dynamic system prompt built from InstructionBuilder + MemoryManager
    - Ensures SQL tools have the active DB connection
    """
    # Ensure SQL tools receive the active connection
    for tool_name in tool_registry.list_tools(category="sql"):
        tool = tool_registry.get_tool(tool_name)
        if isinstance(tool, SQLTool):
            tool.set_connection(db_connection)

    cfg = Config()
    # Ensure provider env var is hydrated from keyring for current provider (Config.validate handles it)
    cfg.validate()

    # Build model/agent. For some providers (e.g., google), construct provider model explicitly to
    # allow arbitrary model IDs even if not in pydantic-ai's KnownModelName.
    model_name_only = (
        cfg.model_name.split(":", 1)[1] if ":" in cfg.model_name else cfg.model_name
    )

    provider = providers.provider_from_model(cfg.model_name) or ""
    if provider == "google":
        model_obj = GoogleModel(
            model_name_only, provider=GoogleProvider(api_key=cfg.api_key)
        )
        agent = Agent(model_obj, name="sqlsaber")
    elif provider == "anthropic" and bool(getattr(cfg, "oauth_token", None)):
        # Build custom httpx client to inject OAuth headers for Anthropic
        async def add_oauth_headers(request: httpx.Request) -> None:  # type: ignore[override]
            # Remove API-key header if present and add OAuth headers
            if "x-api-key" in request.headers:
                del request.headers["x-api-key"]
            request.headers.update(
                {
                    "Authorization": f"Bearer {cfg.oauth_token}",
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "oauth-2025-04-20",
                    "User-Agent": "ClaudeCode/1.0 (Anthropic Claude Code CLI)",
                    "X-Client-Name": "claude-code",
                    "X-Client-Version": "1.0.0",
                }
            )

        http_client = httpx.AsyncClient(event_hooks={"request": [add_oauth_headers]})
        provider_obj = AnthropicProvider(api_key="placeholder", http_client=http_client)
        model_obj = AnthropicModel(model_name_only, provider=provider_obj)
        agent = Agent(model_obj, name="sqlsaber")
    elif provider == "openai":
        # Use OpenAI Responses Model for structured output capabilities
        model_obj = OpenAIResponsesModel(model_name_only)
        agent = Agent(model_obj, name="sqlsaber")
    else:
        agent = Agent(cfg.model_name, name="sqlsaber")

    # Memory + dynamic system prompt
    memory_manager = MemoryManager()
    instruction_builder = InstructionBuilder(tool_registry)

    is_oauth = provider == "anthropic" and bool(getattr(cfg, "oauth_token", None))

    if not is_oauth:

        @agent.system_prompt(dynamic=True)
        async def sqlsaber_system_prompt(ctx: RunContext) -> str:
            db_type = _get_database_type_name(db_connection)
            instructions = instruction_builder.build_instructions(db_type=db_type)

            # Add memory context if available
            if database_name:
                mem = memory_manager.format_memories_for_prompt(database_name)
            else:
                mem = ""

            parts = [p for p in (instructions, mem) if p and p.strip()]
            return "\n\n".join(parts) if parts else ""
    else:

        @agent.system_prompt(dynamic=True)
        async def sqlsaber_system_prompt(ctx: RunContext) -> str:
            # Minimal system prompt in OAuth mode to match Claude Code identity
            return "You are Claude Code, Anthropic's official CLI for Claude."

    # Expose helpers and context on agent instance
    agent._sqlsaber_memory_manager = memory_manager  # type: ignore[attr-defined]
    agent._sqlsaber_database_name = database_name  # type: ignore[attr-defined]
    agent._sqlsaber_instruction_builder = instruction_builder  # type: ignore[attr-defined]
    agent._sqlsaber_db_type = _get_database_type_name(db_connection)  # type: ignore[attr-defined]
    agent._sqlsaber_is_oauth = is_oauth  # type: ignore[attr-defined]

    # Tool wrappers that invoke the registered tools
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
    async def execute_sql(ctx: RunContext, query: str, limit: int | None = 100) -> str:
        """
        Execute a SQL query and return the results.

        Args:
            query: SQL query to execute
            limit: Maximum number of rows to return (default: 100)
        """
        tool = tool_registry.get_tool("execute_sql")
        return await tool.execute(query=query, limit=limit)

    return agent


def _get_database_type_name(db: BaseDatabaseConnection) -> str:
    """Get the human-readable database type name (mirrors BaseSQLAgent)."""

    if isinstance(db, PostgreSQLConnection):
        return "PostgreSQL"
    elif isinstance(db, MySQLConnection):
        return "MySQL"
    elif isinstance(db, SQLiteConnection):
        return "SQLite"
    elif isinstance(db, DuckDBConnection):
        return "DuckDB"
    elif isinstance(db, CSVConnection):
        return "DuckDB"
    else:
        return "database"
