"""Abstract base class for SQL agents."""

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from sqlsaber.database.connection import (
    BaseDatabaseConnection,
    CSVConnection,
    MySQLConnection,
    PostgreSQLConnection,
    SQLiteConnection,
)
from sqlsaber.database.schema import SchemaManager
from sqlsaber.models.events import StreamEvent


class BaseSQLAgent(ABC):
    """Abstract base class for SQL agents."""

    def __init__(self, db_connection: BaseDatabaseConnection):
        self.db = db_connection
        self.schema_manager = SchemaManager(db_connection)
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

    async def introspect_schema(self, table_pattern: Optional[str] = None) -> str:
        """Introspect database schema to understand table structures."""
        try:
            # Pass table_pattern to get_schema_info for efficient filtering at DB level
            schema_info = await self.schema_manager.get_schema_info(table_pattern)

            # Format the schema information
            formatted_info = {}
            for table_name, table_info in schema_info.items():
                formatted_info[table_name] = {
                    "columns": {
                        col_name: {
                            "type": col_info["data_type"],
                            "nullable": col_info["nullable"],
                            "default": col_info["default"],
                        }
                        for col_name, col_info in table_info["columns"].items()
                    },
                    "primary_keys": table_info["primary_keys"],
                    "foreign_keys": [
                        f"{fk['column']} -> {fk['references']['table']}.{fk['references']['column']}"
                        for fk in table_info["foreign_keys"]
                    ],
                }

            return json.dumps(formatted_info)
        except Exception as e:
            return json.dumps({"error": f"Error introspecting schema: {str(e)}"})

    async def list_tables(self) -> str:
        """List all tables in the database with basic information."""
        try:
            tables_info = await self.schema_manager.list_tables()
            return json.dumps(tables_info)
        except Exception as e:
            return json.dumps({"error": f"Error listing tables: {str(e)}"})

    async def execute_sql(self, query: str, limit: Optional[int] = 100) -> str:
        """Execute a SQL query against the database."""
        try:
            # Security check - only allow SELECT queries unless write is enabled
            write_error = self._validate_write_operation(query)
            if write_error:
                return json.dumps(
                    {
                        "error": write_error,
                    }
                )

            # Add LIMIT if not present and it's a SELECT query
            query = self._add_limit_to_query(query, limit)

            # Execute the query (wrapped in a transaction for safety)
            results = await self.db.execute_query(query)

            # Format results
            actual_limit = limit if limit is not None else len(results)

            return json.dumps(
                {
                    "success": True,
                    "row_count": len(results),
                    "results": results[:actual_limit],  # Extra safety for limit
                    "truncated": len(results) > actual_limit,
                }
            )

        except Exception as e:
            error_msg = str(e)

            # Provide helpful error messages
            suggestions = []
            if "column" in error_msg.lower() and "does not exist" in error_msg.lower():
                suggestions.append(
                    "Check column names using the schema introspection tool"
                )
            elif "table" in error_msg.lower() and "does not exist" in error_msg.lower():
                suggestions.append(
                    "Check table names using the schema introspection tool"
                )
            elif "syntax error" in error_msg.lower():
                suggestions.append(
                    "Review SQL syntax, especially JOIN conditions and WHERE clauses"
                )

            return json.dumps({"error": error_msg, "suggestions": suggestions})

    async def process_tool_call(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> str:
        """Process a tool call and return the result."""
        if tool_name == "list_tables":
            return await self.list_tables()
        elif tool_name == "introspect_schema":
            return await self.introspect_schema(tool_input.get("table_pattern"))
        elif tool_name == "execute_sql":
            return await self.execute_sql(
                tool_input["query"], tool_input.get("limit", 100)
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

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
