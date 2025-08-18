"""Abstract base class for SQL agents."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from sqlsaber.conversation.manager import ConversationManager
from sqlsaber.database.connection import (
    BaseDatabaseConnection,
    CSVConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.schema import SchemaManager
from sqlsaber.models.events import StreamEvent
from sqlsaber.tools import SQLTool, tool_registry


class BaseSQLAgent(ABC):
    """Abstract base class for SQL agents."""

    def __init__(self, db_connection: BaseDatabaseConnection):
        self.db = db_connection
        self.schema_manager = SchemaManager(db_connection)
        self.conversation_history: list[dict[str, Any]] = []

        # Conversation persistence
        self._conv_manager = ConversationManager()
        self._conversation_id: str | None = None
        self._msg_index: int = 0

        # Initialize SQL tools with database connection
        self._init_tools()

    @abstractmethod
    async def query_stream(
        self,
        user_query: str,
        use_history: bool = True,
        cancellation_token: asyncio.Event | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Process a user query and stream responses.

        Args:
            user_query: The user's query to process
            use_history: Whether to include conversation history
            cancellation_token: Optional event to signal cancellation
        """
        pass

    async def clear_history(self):
        """Clear conversation history."""
        # End current conversation in storage
        await self._end_conversation()

        # Clear in-memory history
        self.conversation_history = []

    def _get_database_type_name(self) -> str:
        """Get the human-readable database type name."""
        if isinstance(self.db, PostgreSQLConnection):
            return "PostgreSQL"
        elif isinstance(self.db, MySQLConnection):
            return "MySQL"
        elif isinstance(self.db, SQLiteConnection):
            return "SQLite"
        elif isinstance(self.db, CSVConnection):
            return "SQLite"  # we convert csv to in-memory sqlite
        else:
            return "database"  # Fallback

    def _init_tools(self) -> None:
        """Initialize SQL tools with database connection."""
        # Get all SQL tools and set their database connection
        for tool_name in tool_registry.list_tools(category="sql"):
            tool = tool_registry.get_tool(tool_name)
            if isinstance(tool, SQLTool):
                tool.set_connection(self.db)

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

    # Conversation persistence helpers

    async def _ensure_conversation(self) -> None:
        """Ensure a conversation is active for storing messages."""
        if self._conversation_id is None:
            db_name = getattr(self, "database_name", "unknown")
            self._conversation_id = await self._conv_manager.start_conversation(db_name)
            self._msg_index = 0

    async def _store_user_message(self, content: str | dict[str, Any]) -> None:
        """Store a user message in conversation history."""
        if self._conversation_id is None:
            return

        await self._conv_manager.add_user_message(
            self._conversation_id, content, self._msg_index
        )
        self._msg_index += 1

    async def _store_assistant_message(
        self, content: list[dict[str, Any]] | dict[str, Any]
    ) -> None:
        """Store an assistant message in conversation history."""
        if self._conversation_id is None:
            return

        await self._conv_manager.add_assistant_message(
            self._conversation_id, content, self._msg_index
        )
        self._msg_index += 1

    async def _store_tool_message(
        self, content: list[dict[str, Any]] | dict[str, Any]
    ) -> None:
        """Store a tool/system message in conversation history."""
        if self._conversation_id is None:
            return

        await self._conv_manager.add_tool_message(
            self._conversation_id, content, self._msg_index
        )
        self._msg_index += 1

    async def _end_conversation(self) -> None:
        """End the current conversation."""
        if self._conversation_id:
            await self._conv_manager.end_conversation(self._conversation_id)
        self._conversation_id = None
        self._msg_index = 0

    async def restore_conversation(self, conversation_id: str) -> bool:
        """Restore a conversation from storage to in-memory history.

        Args:
            conversation_id: ID of the conversation to restore

        Returns:
            True if successfully restored, False otherwise
        """
        success = await self._conv_manager.restore_conversation_to_agent(
            conversation_id, self.conversation_history
        )

        if success:
            # Set up for continuing this conversation
            self._conversation_id = conversation_id
            self._msg_index = len(self.conversation_history)

        return success

    async def list_conversations(self, limit: int = 50) -> list:
        """List conversations for this agent's database.

        Args:
            limit: Maximum number of conversations to return

        Returns:
            List of conversation data
        """
        db_name = getattr(self, "database_name", None)
        conversations = await self._conv_manager.list_conversations(db_name, limit)

        return [
            {
                "id": conv.id,
                "database_name": conv.database_name,
                "started_at": conv.formatted_start_time(),
                "ended_at": conv.formatted_end_time(),
                "duration": conv.duration_seconds(),
            }
            for conv in conversations
        ]
