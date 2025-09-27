"""Generic SQL agent implementation for MCP tools."""

from typing import AsyncIterator

from sqlsaber.agents.base import BaseSQLAgent
from sqlsaber.database import BaseDatabaseConnection


class MCPSQLAgent(BaseSQLAgent):
    """MCP SQL Agent for MCP tool operations without LLM-specific logic."""

    def __init__(self, db_connection: BaseDatabaseConnection):
        super().__init__(db_connection)

    async def query_stream(
        self, user_query: str, use_history: bool = True
    ) -> AsyncIterator:
        """Not implemented for generic agent as it's only used for tool operations."""
        raise NotImplementedError(
            "MCPSQLAgent does not support query streaming. Use specific agent implementations for conversation."
        )
