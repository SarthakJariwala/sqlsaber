"""SQL-related tools for database operations."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import RunContext, ToolReturn

from sqlsaber.database import BaseDatabaseConnection
from sqlsaber.database.registry import DatabaseRegistry, UnknownDatabaseError
from sqlsaber.database.schema import SchemaManager
from sqlsaber.query_results import (
    MAX_CANONICAL_QUERY_RESULT_BYTES,
    QueryResultContext,
    QueryResultData,
    QueryResultStore,
    build_model_projection,
    descriptor_for_data,
    logical_result_file,
    new_query_result_id,
    query_result_columns,
)
from sqlsaber.render.blocks import Block, Cell, code, error, md, table
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
from .sql_guard import add_limit, validate_sql


def _as_cell(value: object) -> Cell:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


@dataclass
class _ResolvedTarget:
    """Connection + schema manager + dialect resolved for a single tool call."""

    connection: BaseDatabaseConnection
    schema_manager: SchemaManager
    dialect: str
    database_name: str | None = None


class ToolDatabaseError(Exception):
    """Raised when a tool cannot resolve which database to operate against."""


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
        self.registry: DatabaseRegistry | None = None
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

    def set_registry(self, registry: DatabaseRegistry) -> None:
        """Attach a multi-database registry. Replaces any single-DB binding."""
        self.registry = registry

    def _resolve(self, db_name: str | None) -> _ResolvedTarget:
        """Resolve which DB to act on for this tool call.

        - Registry attached: `db_name` must be a registered name.
        - No registry: fall back to the single-DB connection set via
          `set_connection`. `db_name` is ignored.
        Raises `ToolDatabaseError` with a user-facing message on failure.
        """
        if self.registry is not None:
            if not db_name:
                raise ToolDatabaseError(
                    "Multiple databases are connected; you must pass `db_name`. "
                    f"Valid names: {', '.join(self.registry.names())}."
                )
            try:
                entry = self.registry.get(db_name)
            except UnknownDatabaseError as exc:
                raise ToolDatabaseError(str(exc)) from exc
            return _ResolvedTarget(
                connection=entry.connection,
                schema_manager=entry.schema_manager,
                dialect=entry.dialect,
                database_name=entry.name,
            )

        if not self.db:
            raise ToolDatabaseError("No database connection available")

        schema_manager = self.schema_manager or SchemaManager(self.db)
        return _ResolvedTarget(
            connection=self.db,
            schema_manager=schema_manager,
            dialect=self.db.sqlglot_dialect,
            database_name=self.db.display_name,
        )


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

    async def execute(self, db_name: str | None = None) -> str:
        """List all tables in the database."""
        try:
            target = self._resolve(db_name)
        except ToolDatabaseError as exc:
            return json_dumps({"error": str(exc)})

        try:
            tables_info = await target.schema_manager.list_tables()
            return json_dumps(tables_info)
        except Exception as e:
            return json_dumps({"error": f"Error listing tables: {str(e)}"})


class IntrospectSchemaTool(SQLTool):
    """Tool for introspecting database schema."""

    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(message="Examining schema", icon="⚙️"),
        metadata=DisplayMetadata(display_name="Introspect Schema"),
    )

    def render_result(
        self,
        result: object,
        *,
        context: object = None,
    ) -> Sequence[Block] | None:
        del context
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None

        if "error" in mapping and mapping["error"]:
            return (error(str(mapping["error"])),)

        if not mapping:
            return (md("*No schema information found.*"),)

        blocks: list[Block] = [
            md(f"**Schema Information ({len(mapping)} tables):**"),
        ]
        for table_name, table_info in mapping.items():
            table_mapping = self._coerce_mapping(table_info)
            if table_mapping is None:
                continue
            blocks.append(md(f"**Table: {table_name}**"))
            table_comment = table_mapping.get("comment")
            if table_comment:
                blocks.append(md(f"*Comment: {table_comment}*"))
            table_columns = self._coerce_mapping(table_mapping.get("columns")) or {}
            if table_columns:
                include_comments = any(
                    (col_mapping := self._coerce_mapping(col_info))
                    and col_mapping.get("comment")
                    for col_info in table_columns.values()
                )
                rows: list[dict[str, Cell]] = []
                for col_name, col_info in table_columns.items():
                    col_mapping = self._coerce_mapping(col_info)
                    if col_mapping is None:
                        continue
                    row: dict[str, Cell] = {
                        "Column": col_name,
                        "Type": _as_cell(col_mapping.get("type", "")),
                        "Nullable": _as_cell(col_mapping.get("nullable")),
                        "Default": _as_cell(col_mapping.get("default")),
                    }
                    if include_comments:
                        row["Comments"] = _as_cell(col_mapping.get("comment", ""))
                    rows.append(row)
                blocks.append(table(rows, max_rows=200, max_columns=8))
            primary_keys = table_mapping.get("primary_keys") or []
            if isinstance(primary_keys, list) and primary_keys:
                blocks.append(
                    md(
                        f"**Primary Keys:** {', '.join(self._stringify_list(primary_keys))}"
                    )
                )
            foreign_keys = table_mapping.get("foreign_keys") or []
            if isinstance(foreign_keys, list) and foreign_keys:
                blocks.append(
                    md(
                        f"**Foreign Keys:** {', '.join(self._stringify_list(foreign_keys))}"
                    )
                )
            indexes = table_mapping.get("indexes") or []
            if isinstance(indexes, list) and indexes:
                blocks.append(
                    md(f"**Indexes:** {', '.join(self._stringify_list(indexes))}")
                )
        return tuple(blocks)

    def _parse_result(self, result: object) -> object:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": result}
        return {"error": str(result)}

    def _coerce_mapping(self, data: object) -> dict[str, object] | None:
        if not isinstance(data, dict):
            return None
        return {str(key): value for key, value in data.items()}

    def _stringify_list(self, items: list[object] | list[Any]) -> list[str]:
        return [str(item) for item in items]

    @property
    def name(self) -> str:
        return "introspect_schema"

    async def execute(
        self, table_pattern: str | None = None, db_name: str | None = None
    ) -> str:
        """
        Introspect database schema.

        Args:
            table_pattern: Optional pattern to filter tables (e.g., 'public.users', 'user%', '%order%')
        """
        try:
            target = self._resolve(db_name)
        except ToolDatabaseError as exc:
            return json_dumps({"error": str(exc)})

        try:
            schema_info = await target.schema_manager.get_schema_info(table_pattern)

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


class ExecuteSQLTool(SQLTool):
    """Tool for executing SQL queries."""

    def __init__(
        self,
        db_connection: BaseDatabaseConnection | None = None,
        schema_manager: SchemaManager | None = None,
        *,
        query_result_store: QueryResultStore | None = None,
    ) -> None:
        super().__init__(db_connection, schema_manager)
        self.query_result_store = query_result_store

    display_spec = ToolDisplaySpec(
        metadata=DisplayMetadata(display_name="Execute SQL"),
    )

    def render_executing(self, args: Mapping[str, Any]) -> Sequence[Block] | None:
        query = args.get("query") or args.get("sql") or ""
        if not isinstance(query, str) or not query.strip():
            return None
        return (md("**Executing SQL:**"), code(query, "sql"))

    def render_result(
        self,
        result: object,
        *,
        context: object = None,
    ) -> Sequence[Block] | None:
        del context
        data = self._parse_result(result)
        mapping = self._coerce_mapping(data)
        if mapping is None:
            return None

        if "error" in mapping and mapping["error"]:
            return (error(str(mapping["error"]), label="SQL error"),)

        results = mapping.get("results")
        if isinstance(results, list) and results:
            rows = self._coerce_rows(cast(list[object], results))
            keys = list(dict.fromkeys(key for row in rows for key in row))
            if not keys:
                return (md(f"*{len(rows)} rows returned with no columns.*"),)
            return (
                table(
                    rows,
                    caption=f"Results ({len(rows)} rows):",
                    max_columns=15,
                    max_rows=20,
                ),
            )
        if isinstance(results, list):
            return (md("*0 rows returned*"),)
        if mapping.get("success"):
            return (md("✓ Query completed successfully"),)
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

    async def execute(
        self, ctx: RunContext, query: str, db_name: str | None = None
    ) -> str | ToolReturn:
        """
        Execute a SQL query against the database.

        Args:
            query: SQL query to execute
        """
        try:
            target = self._resolve(db_name)
        except ToolDatabaseError as exc:
            return json_dumps({"error": str(exc)})

        if not query:
            return json_dumps({"error": "No query provided"})

        max_rows = self.MAX_ROWS

        try:
            # Get the dialect for this database
            dialect = target.dialect

            # Security check using sqlglot AST analysis
            validation_result = validate_sql(
                query, dialect, allow_dangerous=self.allow_dangerous
            )
            if not validation_result.allowed:
                return json_dumps({"error": validation_result.reason})

            # Add LIMIT if not present and it's a SELECT query
            auto_limit_applied = bool(
                validation_result.is_select
                and max_rows
                and not validation_result.has_limit
            )
            if auto_limit_applied:
                query = add_limit(query, dialect, max_rows)

            query_type = validation_result.query_type or "other"

            # Commit only for DML/DDL statements in dangerous mode
            commit = bool(self.allow_dangerous and query_type in {"dml", "ddl"})

            # Execute the query
            results = await target.connection.execute_query(
                query,
                commit=commit,
                read_only=not self.allow_dangerous,
            )

            # Format response based on query type. Directly constructed legacy
            # ExecuteSQLTool instances retain their old string return; managed and
            # standalone SqlTools always inject a canonical result store.
            tool_call_id = ctx.tool_call_id
            if query_type in {"dml", "ddl"}:
                payload: dict[str, Any] = {"success": True}
                if tool_call_id:
                    payload["file"] = f"result_{tool_call_id}.json"
                return json_dumps(payload)

            row_count = len(results)
            if self.query_result_store is None:
                payload: dict[str, Any] = {
                    "success": True,
                    "row_count": row_count,
                    "results": results,
                }
                if tool_call_id:
                    payload["file"] = f"result_{tool_call_id}.json"
                if auto_limit_applied:
                    payload["auto_limit_applied"] = True
                return json_dumps(payload)

            result_id = new_query_result_id()
            file = logical_result_file(tool_call_id, result_id)
            canonical_payload: dict[str, Any] = {
                "success": True,
                "row_count": row_count,
                "results": results,
                "file": file,
            }
            if auto_limit_applied:
                canonical_payload["auto_limit_applied"] = True
            try:
                canonical_data = json_dumps(
                    canonical_payload, ensure_ascii=False
                ).encode("utf-8")
            except (TypeError, ValueError):
                return json_dumps(
                    {
                        "error": (
                            "Query completed but its result could not be serialized "
                            "for retention."
                        )
                    }
                )
            if len(canonical_data) > MAX_CANONICAL_QUERY_RESULT_BYTES:
                return json_dumps(
                    {
                        "error": (
                            "Query completed but its result exceeds the retention "
                            "size limit."
                        )
                    }
                )
            descriptor = descriptor_for_data(
                canonical_data,
                result_id=result_id,
                file=file,
                row_count=row_count,
                columns=query_result_columns(cast(list[object], results)),
                database_name=target.database_name,
            )
            try:
                descriptor = await self.query_result_store.put(
                    QueryResultData(canonical_data),
                    descriptor=descriptor,
                    context=QueryResultContext(
                        run_id=getattr(ctx, "run_id", None),
                        conversation_id=getattr(ctx, "conversation_id", None),
                        tool_call_id=tool_call_id,
                        metadata=getattr(ctx, "metadata", None) or {},
                    ),
                )
                projection = build_model_projection(canonical_payload, descriptor)
            except Exception:
                return json_dumps(
                    {
                        "error": (
                            "Query completed but its complete result could not be "
                            "retained."
                        )
                    }
                )
            return ToolReturn(
                return_value=json_dumps(
                    projection,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                metadata={"query_result": descriptor.to_dict()},
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

            return json_dumps({"error": error_msg})


class ListDatabasesTool(SQLTool):
    """List databases connected to the current SQLSaber session.

    Registered only when more than one database is connected. The tool
    exposes the name, dialect, and optional description of each, so the
    agent can choose which database to target with subsequent tool calls.
    """

    multi_db_only = True

    display_spec = ToolDisplaySpec(
        executing=ExecutingConfig(message="Listing connected databases", icon="📚"),
        result=ResultConfig(
            format="table",
            title="Connected Databases ({total_databases} total)",
            fields=FieldMappings(items="databases", error="error"),
            table=TableConfig(
                columns=[
                    ColumnDef(field="name", header="Name", style="column.name"),
                    ColumnDef(field="dialect", header="Dialect", style="column.type"),
                    ColumnDef(field="description", header="Description", style="muted"),
                ],
                max_rows=20,
            ),
        ),
        metadata=DisplayMetadata(display_name="List Databases"),
    )

    @property
    def name(self) -> str:
        return "list_dbs"

    async def execute(self) -> str:
        """List all databases connected to this session."""
        if self.registry is None:
            return json_dumps(
                {"error": "Multi-database registry is not configured for this session."}
            )

        entries = self.registry.catalog()
        return json_dumps({"total_databases": len(entries), "databases": entries})
