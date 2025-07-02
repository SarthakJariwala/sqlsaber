"""Abstract base class for SQL agents."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from uniplot import histogram, plot

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
        self.conversation_history: list[dict[str, Any]] = []

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

    async def introspect_schema(self, table_pattern: str | None = None) -> str:
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

    async def execute_sql(self, query: str, limit: int | None = None) -> str:
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
        self, tool_name: str, tool_input: dict[str, Any]
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
        elif tool_name == "plot_data":
            return await self.plot_data(
                y_values=tool_input["y_values"],
                x_values=tool_input.get("x_values"),
                plot_type=tool_input.get("plot_type", "line"),
                title=tool_input.get("title"),
                x_label=tool_input.get("x_label"),
                y_label=tool_input.get("y_label"),
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _validate_write_operation(self, query: str) -> str | None:
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

    async def plot_data(
        self,
        y_values: list[float],
        x_values: list[float] | None = None,
        plot_type: str = "line",
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> str:
        """Create a terminal plot using uniplot.

        Args:
            y_values: Y-axis data points
            x_values: X-axis data points (optional)
            plot_type: Type of plot - "line", "scatter", or "histogram"
            title: Plot title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            JSON string with success status and plot details
        """
        try:
            # Validate inputs
            if not y_values:
                return json.dumps({"error": "No data provided for plotting"})

            # Convert to floats if needed
            try:
                y_values = [float(v) if v is not None else None for v in y_values]
                if x_values:
                    x_values = [float(v) if v is not None else None for v in x_values]
            except (ValueError, TypeError) as e:
                return json.dumps({"error": f"Invalid data format: {str(e)}"})

            # Create the plot
            if plot_type == "histogram":
                # For histogram, we only need y_values
                histogram(
                    y_values,
                    title=title,
                    bins=min(20, len(set(y_values))),  # Adaptive bin count
                )
                plot_info = {
                    "type": "histogram",
                    "data_points": len(y_values),
                    "title": title or "Histogram",
                }
            elif plot_type in ["line", "scatter"]:
                # For line/scatter plots
                plot_kwargs = {
                    "ys": y_values,
                    "title": title,
                    "lines": plot_type == "line",
                }

                if x_values:
                    plot_kwargs["xs"] = x_values
                if x_label:
                    plot_kwargs["x_unit"] = x_label
                if y_label:
                    plot_kwargs["y_unit"] = y_label

                plot(**plot_kwargs)

                plot_info = {
                    "type": plot_type,
                    "data_points": len(y_values),
                    "title": title or f"{plot_type.capitalize()} Plot",
                    "has_x_values": x_values is not None,
                }
            else:
                return json.dumps({"error": f"Unsupported plot type: {plot_type}"})

            return json.dumps(
                {"success": True, "plot_rendered": True, "plot_info": plot_info}
            )

        except Exception as e:
            return json.dumps({"error": f"Error creating plot: {str(e)}"})
