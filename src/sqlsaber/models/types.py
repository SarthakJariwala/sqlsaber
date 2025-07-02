"""Type definitions for SQLSaber."""

from typing import Any, TypedDict


class ColumnInfo(TypedDict):
    """Type definition for column information."""

    data_type: str
    nullable: bool
    default: str | None
    max_length: int | None
    precision: int | None
    scale: int | None


class ForeignKeyInfo(TypedDict):
    """Type definition for foreign key information."""

    column: str
    references: dict[str, str]  # {"table": "schema.table", "column": "column_name"}


class SchemaInfo(TypedDict):
    """Type definition for schema information."""

    schema: str
    name: str
    type: str
    columns: dict[str, ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]


class ToolDefinition(TypedDict):
    """Type definition for tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]
