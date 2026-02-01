"""SQL-related tools for database operations."""

import json
from html import escape
from typing import Any, cast

from pydantic_ai import RunContext
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from tabulate import tabulate

from sqlsaber.theme.manager import get_theme_manager

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.schema import SchemaManager
from sqlsaber.utils.json_utils import json_dumps

from .base import Tool
from .display import (
    ColumnDef,
    DisplayMetadata,
    ExecutingConfig,
    FieldMappings,
    ResultConfig,
    TableConfig,
    ToolDisplaySpec,
)
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

    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(message="Discovering available tables", icon="⚙️"),
        result=ResultConfig(
            format="table",
            title="Database Tables ({total_tables} total)",
            fields=FieldMappings(items="tables", error="error"),
            table=TableConfig(
                columns=[
                    ColumnDef(field="schema", header="Schema", style="column.schema"),
                    ColumnDef(field="name", header="Table Name", style="column.name"),
                    ColumnDef(field="type", header="Type", style="column.type"),
                ],
                max_rows=50,
            ),
        ),
        metadata=DisplayMetadata(display_name="List Tables"),
    )

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

    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(message="Examining schema", icon="⚙️"),
        metadata=DisplayMetadata(display_name="Introspect Schema"),
    )

    def render_result(self, console: Console, result: object) -> bool:
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return False

        if "error" in mapping and mapping["error"]:
            message = str(mapping["error"])
            if console.is_terminal:
                console.print(f"[error]Error:[/error] {message}")
            else:
                console.print(f"**Error:** {message}\n")
            return True

        if not mapping:
            if console.is_terminal:
                console.print("[warning]No schema information found.[/warning]")
            else:
                console.print("*No schema information found.*\n")
            return True

        tm = get_theme_manager()

        if not console.is_terminal:
            console.print(f"\n**Schema Information ({len(mapping)} tables):**\n")
            console.print(self._schema_to_markdown(mapping))
            console.print()
            return True

        console.print(f"\n[title]Schema Information ({len(mapping)} tables):[/title]")

        for table_name, table_info in mapping.items():
            table_mapping = self._coerce_mapping(table_info)
            if table_mapping is None:
                continue
            console.print(f"\n[heading]Table: {table_name}[/heading]")

            table_comment = table_mapping.get("comment")
            if table_comment:
                console.print(f"[muted]Comment: {table_comment}[/muted]")

            table_columns = self._coerce_mapping(table_mapping.get("columns")) or {}
            if table_columns:
                include_column_comments = any(
                    (col_mapping := self._coerce_mapping(col_info))
                    and col_mapping.get("comment")
                    for col_info in table_columns.values()
                )

                columns = [
                    {"name": "Column Name", "style": tm.style("column.name")},
                    {"name": "Type", "style": tm.style("column.type")},
                    {"name": "Nullable", "style": tm.style("info")},
                    {"name": "Default", "style": tm.style("muted")},
                ]
                if include_column_comments:
                    columns.append({"name": "Comment", "style": tm.style("muted")})

                col_table = self._create_table(columns, title="Columns", tm=tm)

                for col_name, col_info in table_columns.items():
                    col_mapping = self._coerce_mapping(col_info)
                    if col_mapping is None:
                        continue
                    nullable = "✓" if bool(col_mapping.get("nullable", False)) else "✗"
                    default_value = col_mapping.get("default")
                    default = str(default_value) if default_value else ""
                    row = [
                        col_name,
                        col_mapping.get("type", ""),
                        nullable,
                        default,
                    ]
                    if include_column_comments:
                        row.append(col_mapping.get("comment") or "")
                    col_table.add_row(
                        *[str(value) if value is not None else "" for value in row]
                    )

                console.print(col_table)

            primary_keys = table_mapping.get("primary_keys") or []
            if isinstance(primary_keys, list) and primary_keys:
                console.print(
                    f"[key.primary]Primary Keys:[/key.primary] {', '.join(self._stringify_list(primary_keys))}"
                )

            foreign_keys = table_mapping.get("foreign_keys") or []
            if isinstance(foreign_keys, list) and foreign_keys:
                console.print("[key.foreign]Foreign Keys:[/key.foreign]")
                for fk in foreign_keys:
                    console.print(f"  • {fk}")

            indexes = table_mapping.get("indexes") or []
            if isinstance(indexes, list) and indexes:
                console.print("[key.index]Indexes:[/key.index]")
                for idx in indexes:
                    console.print(f"  • {idx}")

        return True

    def render_result_html(self, result: object) -> str | None:
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None
        if "error" in mapping and mapping["error"]:
            return (
                f'<div class="sql-error"><strong>Error:</strong> '
                f"{self._escape_html(mapping['error'])}</div>"
            )
        if not mapping:
            return '<p class="result-count">No schema information found.</p>'

        parts: list[str] = []
        parts.append(
            f'<p class="result-count">{len(mapping)} table(s) introspected</p>'
        )

        for table_name, table_info in mapping.items():
            table_mapping = self._coerce_mapping(table_info)
            if table_mapping is None:
                continue
            parts.append('<div class="schema-table">')
            parts.append(f'<h4 class="table-name">{self._escape_html(table_name)}</h4>')

            if table_mapping.get("comment"):
                parts.append(
                    f'<p class="table-comment">'
                    f"{self._escape_html(table_mapping.get('comment'))}</p>"
                )

            columns = self._coerce_mapping(table_mapping.get("columns")) or {}
            if columns:
                rows = []
                for col_name, col_info in columns.items():
                    col_mapping = self._coerce_mapping(col_info)
                    if col_mapping is None:
                        continue
                    nullable = "✓" if col_mapping.get("nullable") else "✗"
                    default = self._escape_html(col_mapping.get("default") or "—")
                    comment = self._escape_html(col_mapping.get("comment") or "")
                    rows.append(
                        "<tr>"
                        f"<td>{self._escape_html(col_name)}</td>"
                        f"<td>{self._escape_html(col_mapping.get('type', ''))}</td>"
                        f"<td>{nullable}</td>"
                        f"<td>{default}</td>"
                        f"<td>{comment}</td>"
                        "</tr>"
                    )
                parts.append(
                    '<div class="table-wrapper">'
                    '<table class="sql-results schema-columns">'
                    "<thead><tr><th>Column</th><th>Type</th><th>Nullable</th>"
                    "<th>Default</th><th>Comment</th></tr></thead><tbody>"
                )
                parts.append("".join(rows))
                parts.append("</tbody></table></div>")

            pks = table_mapping.get("primary_keys") or []
            if isinstance(pks, list) and pks:
                parts.append(
                    f'<p class="key-info"><strong>Primary Keys:</strong> '
                    f"{self._escape_html(', '.join(self._stringify_list(pks)))}"
                    "</p>"
                )

            fks = table_mapping.get("foreign_keys") or []
            if isinstance(fks, list) and fks:
                fk_list = "".join(f"<li>{self._escape_html(fk)}</li>" for fk in fks)
                parts.append(
                    f'<p class="key-info"><strong>Foreign Keys:</strong></p>'
                    f'<ul class="key-list">{fk_list}</ul>'
                )

            parts.append("</div>")

        return "".join(parts)

    def _parse_result(self, result: object) -> object:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": result}
        return {"error": str(result)}

    def _schema_to_markdown(self, data: dict[str, object]) -> str:
        parts = []
        for table_name, table_info in data.items():
            table_mapping = self._coerce_mapping(table_info)
            if table_mapping is None:
                continue
            columns = self._coerce_mapping(table_mapping.get("columns")) or {}
            table_data = []
            for col_name, col_info in columns.items():
                col_mapping = self._coerce_mapping(col_info)
                if col_mapping is None:
                    continue
                table_data.append(
                    [
                        col_name,
                        col_mapping.get("type"),
                        col_mapping.get("nullable"),
                        col_mapping.get("default"),
                        col_mapping.get("comment", ""),
                    ]
                )
            md = f"**Table: {table_name}**\n\n"
            md += tabulate(
                table_data,
                headers=["Column", "Type", "Nullable", "Default", "Comments"],
                tablefmt="github",
            )
            primary_keys = table_mapping.get("primary_keys")
            if isinstance(primary_keys, list) and primary_keys:
                md += f"\n\n**Primary Keys:** {', '.join(self._stringify_list(primary_keys))}"
            foreign_keys = table_mapping.get("foreign_keys")
            if isinstance(foreign_keys, list) and foreign_keys:
                md += f"\n\n**Foreign Keys:** {', '.join(self._stringify_list(foreign_keys))}"
            parts.append(md)
        return "\n\n".join(parts)

    def _create_table(
        self,
        columns: list[dict[str, str]],
        title: str | None,
        tm,
    ) -> Table:
        table = Table(
            show_header=True, header_style=tm.style("table.header"), title=title
        )
        for col in columns:
            table.add_column(col["name"], style=col.get("style"))
        return table

    def _escape_html(self, value: object) -> str:
        if value is None:
            return ""
        return escape(str(value))

    def _coerce_mapping(self, data: object) -> dict[str, object] | None:
        if not isinstance(data, dict):
            return None
        return {str(key): value for key, value in data.items()}

    def _stringify_list(self, items: list[object] | list[Any]) -> list[str]:
        return [str(item) for item in items]

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

    display_spec = ToolDisplaySpec(
        metadata=DisplayMetadata(display_name="Execute SQL"),
    )

    def render_executing(self, console: Console, args: dict) -> bool:
        query = args.get("query") or args.get("sql") or ""
        if not isinstance(query, str) or not query.strip():
            return False

        if console.is_terminal:
            console.print("[muted bold]:gear: Executing SQL:[/muted bold]")
            console.print()
            console.print(
                Syntax(
                    query,
                    "sql",
                    theme=get_theme_manager().pygments_style_name,
                    background_color="default",
                    word_wrap=True,
                )
            )
        else:
            console.print("**Executing SQL:**\n")
            console.print(f"```sql\n{query}\n```\n")
        return True

    def render_result(self, console: Console, result: object) -> bool:
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return False

        if "error" in mapping and mapping["error"]:
            message = str(mapping["error"])
            if console.is_terminal:
                console.print(f"[error]SQL error:[/error] {message}")
            else:
                console.print(f"**SQL error:** {message}\n")
            return True

        results = mapping.get("results")
        if isinstance(results, list) and results:
            self._render_results_table(
                console, self._coerce_rows(cast(list[object], results))
            )
            return True

        if isinstance(results, list) and not results:
            if console.is_terminal:
                console.print("[warning]0 rows returned[/warning]")
            else:
                console.print("*0 rows returned*\n")
            return True

        if mapping.get("success"):
            if console.is_terminal:
                console.print("[success]✓ Query completed successfully[/success]")
            else:
                console.print("✓ Query completed successfully\n")
            return True

        return False

    def render_result_html(self, result: object) -> str | None:
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None
        if "error" in mapping and mapping["error"]:
            return (
                f'<div class="sql-error"><strong>Error:</strong> '
                f"{self._escape_html(mapping['error'])}</div>"
            )
        results = mapping.get("results")
        if isinstance(results, list):
            return self._render_results_table_html(
                self._coerce_rows(cast(list[object], results))
            )
        if mapping.get("success"):
            return '<p class="result-count">Query completed successfully</p>'
        return None

    def _parse_result(self, result: object) -> object:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": result}
        return {"error": str(result)}

    def _render_results_table(self, console: Console, results: list[dict]) -> None:
        if not results:
            return

        if not console.is_terminal:
            console.print(f"\n**Results ({len(results)} rows):**\n")
            console.print(tabulate(results, headers="keys", tablefmt="github"))
            console.print()
            return

        tm = get_theme_manager()
        console.print(f"\n[section]Results ({len(results)} rows):[/section]")
        all_columns = list(results[0].keys())
        display_columns = all_columns[:15]
        if len(all_columns) > 15:
            console.print(
                f"[warning]Note: Showing first 15 of {len(all_columns)} columns[/warning]"
            )
        table = Table(show_header=True, header_style=tm.style("table.header"))
        for col in display_columns:
            table.add_column(col)
        for row in results[:20]:
            table.add_row(*[str(row.get(key, "")) for key in display_columns])
        console.print(table)
        if len(results) > 20:
            console.print(f"[warning]... and {len(results) - 20} more rows[/warning]")

    def _render_results_table_html(self, results: list[dict]) -> str:
        if not results:
            return '<p class="result-count">0 rows returned</p>'
        first = results[0]
        headers = list(first.keys()) if isinstance(first, dict) else []
        row_html = []
        for row in results[:100]:
            cells = "".join(
                f"<td>{self._escape_html(row.get(h, ''))}</td>" for h in headers
            )
            row_html.append(f"<tr>{cells}</tr>")
        header_html = "".join(f"<th>{self._escape_html(h)}</th>" for h in headers)
        count_note = f'<p class="result-count">{len(results)} row(s) returned</p>'
        if len(results) > 100:
            count_note = (
                f'<p class="result-count">Showing 100 of {len(results)} rows</p>'
            )
        return (
            f'{count_note}<div class="table-wrapper">'
            f'<table class="sql-results"><thead><tr>{header_html}</tr></thead>'
            f"<tbody>{''.join(row_html)}</tbody></table></div>"
        )

    def _escape_html(self, value: object) -> str:
        if value is None:
            return ""
        return escape(str(value))

    def _coerce_mapping(self, data: object) -> dict[str, object] | None:
        if not isinstance(data, dict):
            return None
        return {str(key): value for key, value in data.items()}

    def _coerce_rows(self, rows: list[object]) -> list[dict]:
        coerced: list[dict] = []
        for row in rows:
            if isinstance(row, dict):
                coerced.append({str(key): value for key, value in row.items()})
            else:
                coerced.append({"value": row})
        return coerced

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
