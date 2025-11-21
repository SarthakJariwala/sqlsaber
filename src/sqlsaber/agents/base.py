"""Abstract base class for SQL agents."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.tools import SQLTool, tool_registry


class BaseSQLAgent(ABC):
    """Abstract base class for SQL agents."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection,
        schema_manager: SchemaManager | None = None,
    ):
        self.db = db_connection
        self.schema_manager = schema_manager or SchemaManager(db_connection)

        # Initialize SQL tools with database connection
        self._init_tools()

    @abstractmethod
    async def query_stream(
        self,
        user_query: str,
        use_history: bool = True,
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator:
        """Process a user query and stream responses.

        Args:
            user_query: The user's query to process
            use_history: Whether to include conversation history
            cancellation_token: Optional event to signal cancellation
        """
        pass

    def _init_tools(self) -> None:
        """Initialize SQL tools with database connection."""
        # Get all SQL tools and set their database connection
        for tool_name in tool_registry.list_tools():
            tool = tool_registry.get_tool(tool_name)
            if isinstance(tool, SQLTool):
                tool.set_connection(self.db, self.schema_manager)

    async def process_tool_call(
        self, tool_name: str, tool_input: dict[str, Any]
    ) -> str:
        """Process a tool call and return the result."""
        try:
            tool = tool_registry.get_tool(tool_name)
            return await tool.execute(**tool_input)
        except KeyError:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            return json.dumps(
                {"error": f"Error executing tool '{tool_name}': {str(e)}"}
            )
