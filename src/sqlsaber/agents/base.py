"""Abstract base class for SQL agents."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from sqlsaber.database.connection import BaseDatabaseConnection
from sqlsaber.models.events import StreamEvent


class BaseSQLAgent(ABC):
    """Abstract base class for SQL agents."""

    def __init__(self, db_connection: BaseDatabaseConnection):
        self.db = db_connection
        self.conversation_history: List[Dict[str, Any]] = []

    @abstractmethod
    async def query_stream(
        self, user_query: str, use_history: bool = True
    ) -> AsyncIterator[StreamEvent]:
        """Process a user query and stream responses."""
        pass

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    @abstractmethod
    async def process_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Process a tool call and return the result."""
        pass

    def _validate_write_operation(self, query: str) -> Optional[str]:
        """Validate if a write operation is allowed.

        Returns:
            None if operation is allowed, error message if not allowed.
        """
        query_upper = query.strip().upper()

        # Check for write operations
        write_keywords = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
        ]
        is_write_query = any(query_upper.startswith(kw) for kw in write_keywords)

        if is_write_query:
            return (
                "Write operations are not allowed. Only SELECT queries are permitted."
            )

        return None

    def _add_limit_to_query(self, query: str, limit: int = 100) -> str:
        """Add LIMIT clause to SELECT queries if not present."""
        query_upper = query.strip().upper()
        if query_upper.startswith("SELECT") and "LIMIT" not in query_upper:
            return f"{query.rstrip(';')} LIMIT {limit};"
        return query
