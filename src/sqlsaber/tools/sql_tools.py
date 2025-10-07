"""SQL-related tools for database operations."""

import json
from typing import Any

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager

from .base import Tool
from .registry import register_tool
from .sql_guard import add_limit, validate_read_only


class SQLTool(Tool):
    """Base class for SQL tools that need database access."""

    def __init__(self, db_connection: BaseDatabaseConnection | None = None):
        """Initialize with optional database connection."""
        super().__init__()
        self.db = db_connection
        self.schema_manager = SchemaManager(db_connection) if db_connection else None

    def set_connection(self, db_connection: BaseDatabaseConnection) -> None:
        """Set the database connection after initialization."""
        self.db = db_connection
        self.schema_manager = SchemaManager(db_connection)


@register_tool
class ListTablesTool(SQLTool):
    """Tool for listing database tables."""

    @property
    def name(self) -> str:
        return "list_tables"

    @property
    def description(self) -> str:
        return "Get a list of all tables in the database with row counts. Use this first to discover available tables."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, **kwargs) -> str:
        """List all tables in the database."""
        if not self.db or not self.schema_manager:
            return json.dumps({"error": "No database connection available"})

        try:
            tables_info = await self.schema_manager.list_tables()
            return json.dumps(tables_info)
        except Exception as e:
            return json.dumps({"error": f"Error listing tables: {str(e)}"})


@register_tool
class IntrospectSchemaTool(SQLTool):
    """Tool for introspecting database schema."""

    @property
    def name(self) -> str:
        return "introspect_schema"

    @property
    def description(self) -> str:
        return "Introspect database schema to understand table structures."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "table_pattern": {
                    "type": "string",
                    "description": "Optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%')",
                }
            },
            "required": [],
        }

    async def execute(self, **kwargs) -> str:
        """Introspect database schema."""
        if not self.db or not self.schema_manager:
            return json.dumps({"error": "No database connection available"})

        try:
            table_pattern = kwargs.get("table_pattern")
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
                    "indexes": [
                        f"{idx['name']} ({', '.join(idx['columns'])})"
                        + (" UNIQUE" if idx["unique"] else "")
                        + (f" [{idx['type']}]" if idx["type"] else "")
                        for idx in table_info["indexes"]
                    ],
                }

            return json.dumps(formatted_info)
        except Exception as e:
            return json.dumps({"error": f"Error introspecting schema: {str(e)}"})


@register_tool
class ExecuteSQLTool(SQLTool):
    """Tool for executing SQL queries."""

    DEFAULT_LIMIT = 100

    @property
    def name(self) -> str:
        return "execute_sql"

    @property
    def description(self) -> str:
        return "Execute a SQL query against the database."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL query to execute",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Maximum number of rows to return (default: {self.DEFAULT_LIMIT})",
                    "default": self.DEFAULT_LIMIT,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs) -> str:
        """Execute a SQL query."""
        if not self.db:
            return json.dumps({"error": "No database connection available"})

        query = kwargs.get("query")
        if not query:
            return json.dumps({"error": "No query provided"})

        limit = kwargs.get("limit", self.DEFAULT_LIMIT)

        try:
            # Get the dialect for this database
            dialect = self.db.sqlglot_dialect

            # Security check using sqlglot AST analysis
            validation_result = validate_read_only(query, dialect)
            if not validation_result.allowed:
                return json.dumps({"error": validation_result.reason})

            # Add LIMIT if not present and it's a SELECT query
            if validation_result.is_select and limit:
                query = add_limit(query, dialect, limit)

            # Execute the query
            results = await self.db.execute_query(query)

            # Format results
            actual_limit = limit if limit is not None else len(results)

            return json.dumps(
                {
                    "success": True,
                    "row_count": len(results),
                    "results": results[:actual_limit],
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
