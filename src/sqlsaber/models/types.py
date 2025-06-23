"""Type definitions for SQLSaber."""

from typing import Any, Dict, List, Optional, TypedDict


class ColumnInfo(TypedDict):
    """Type definition for column information."""

    data_type: str
    nullable: bool
    default: Optional[str]
    max_length: Optional[int]
    precision: Optional[int]
    scale: Optional[int]


class ForeignKeyInfo(TypedDict):
    """Type definition for foreign key information."""

    column: str
    references: Dict[str, str]  # {"table": "schema.table", "column": "column_name"}


class SchemaInfo(TypedDict):
    """Type definition for schema information."""

    schema: str
    name: str
    type: str
    columns: Dict[str, ColumnInfo]
    primary_keys: List[str]
    foreign_keys: List[ForeignKeyInfo]


class ToolDefinition(TypedDict):
    """Type definition for tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any]
