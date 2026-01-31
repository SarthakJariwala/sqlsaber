"""SQL-related tools for database operations."""

from pydantic_ai import RunContext

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.utils.json_utils import json_dumps

from .base import Tool
from .registry import register_tool
from .sql_guard import add_limit, validate_sql


class SQLTool(Tool):
    """Base class for SQL tools that need database access."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection | None = None,
        schema_manager: SchemaManager | None = None,
    ):
        """Initialize with optional database connection."""
        super().__init__()
        self.db = db_connection
        # allow_dangerous is set by SQLSaberAgent at session level
        # Do NOT expose this as a tool parameter to prevent LLM from escalating
        self.allow_dangerous: bool = False
        if schema_manager:
            self.schema_manager = schema_manager
        elif db_connection:
            self.schema_manager = SchemaManager(db_connection)
        else:
            self.schema_manager = None

    def set_connection(
        self,
        db_connection: BaseDatabaseConnection,
        schema_manager: SchemaManager | None = None,
    ) -> None:
        """Set the database connection after initialization."""
        self.db = db_connection
        if schema_manager:
            self.schema_manager = schema_manager
        else:
            self.schema_manager = SchemaManager(db_connection)


@register_tool
class ListTablesTool(SQLTool):
    """Tool for listing database tables."""

    @property
    def name(self) -> str:
        return "list_tables"

    async def execute(self) -> str:
        """List all tables in the database."""
        if not self.db or not self.schema_manager:
            return json_dumps({"error": "No database connection available"})

        try:
            tables_info = await self.schema_manager.list_tables()
            return json_dumps(tables_info)
        except Exception as e:
            return json_dumps({"error": f"Error listing tables: {str(e)}"})


@register_tool
class IntrospectSchemaTool(SQLTool):
    """Tool for introspecting database schema."""

    @property
    def name(self) -> str:
        return "introspect_schema"

    async def execute(self, table_pattern: str | None = None) -> str:
        """
        Introspect database schema.

        Args:
            table_pattern: Optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%')
        """
        if not self.db or not self.schema_manager:
            return json_dumps({"error": "No database connection available"})

        try:
            schema_info = await self.schema_manager.get_schema_info(table_pattern)

            # Format the schema information
            formatted_info = {}
            for table_name, table_info in schema_info.items():
                table_data = {}

                # Add table comment if present
                if table_info.get("comment"):
                    table_data["comment"] = table_info["comment"]

                # Add columns with comments if present
                table_data["columns"] = {}
                for col_name, col_info in table_info["columns"].items():
                    column_data = {
                        "type": col_info["data_type"],
                        "nullable": col_info["nullable"],
                        "default": col_info["default"],
                    }
                    if col_info.get("comment"):
                        column_data["comment"] = col_info["comment"]
                    table_data["columns"][col_name] = column_data

                # Add other schema information
                table_data["primary_keys"] = table_info["primary_keys"]
                table_data["foreign_keys"] = [
                    f"{fk['column']} -> {fk['references']['table']}.{fk['references']['column']}"
                    for fk in table_info["foreign_keys"]
                ]
                table_data["indexes"] = [
                    f"{idx['name']} ({', '.join(idx['columns'])})"
                    + (" UNIQUE" if idx["unique"] else "")
                    + (f" [{idx['type']}]" if idx["type"] else "")
                    for idx in table_info["indexes"]
                ]

                formatted_info[table_name] = table_data

            return json_dumps(formatted_info)
        except Exception as e:
            return json_dumps({"error": f"Error introspecting schema: {str(e)}"})


@register_tool
class ExecuteSQLTool(SQLTool):
    """Tool for executing SQL queries."""

    MAX_ROWS = 1000
    requires_ctx = True

    @property
    def name(self) -> str:
        return "execute_sql"

    async def execute(self, ctx: RunContext, query: str) -> str:
        """
        Execute a SQL query against the database.

        Args:
            query: SQL query to execute
        """
        if not self.db:
            return json_dumps({"error": "No database connection available"})

        if not query:
            return json_dumps({"error": "No query provided"})

        max_rows = self.MAX_ROWS

        try:
            # Get the dialect for this database
            dialect = self.db.sqlglot_dialect

            # Security check using sqlglot AST analysis
            validation_result = validate_sql(
                query, dialect, allow_dangerous=self.allow_dangerous
            )
            if not validation_result.allowed:
                return json_dumps({"error": validation_result.reason})

            # Add LIMIT if not present and it's a SELECT query
            if validation_result.is_select and max_rows:
                if not validation_result.has_limit:
                    query = add_limit(query, dialect, max_rows)

            query_type = validation_result.query_type or "other"

            # Commit only for DML/DDL statements in dangerous mode
            commit = bool(self.allow_dangerous and query_type in {"dml", "ddl"})

            # Execute the query
            results = await self.db.execute_query(query, commit=commit)

            # Format response based on query type
            tool_call_id = ctx.tool_call_id
            if query_type == "select":
                row_count = len(results)
                payload = {
                    "success": True,
                    "row_count": row_count,
                    "results": results,
                }
                if tool_call_id:
                    payload["file"] = f"result_{tool_call_id}.json"
                return json_dumps(payload)
            if query_type in {"dml", "ddl"}:
                payload = {"success": True}
                if tool_call_id:
                    payload["file"] = f"result_{tool_call_id}.json"
                return json_dumps(payload)
            payload = {
                "success": True,
                "row_count": len(results),
                "results": results,
            }
            if tool_call_id:
                payload["file"] = f"result_{tool_call_id}.json"
            return json_dumps(payload)

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

            return json_dumps({"error": error_msg})
