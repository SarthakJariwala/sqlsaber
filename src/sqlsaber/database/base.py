"""Base classes and type definitions for database connections and schema introspection."""

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, TypedDict

# Default query timeout to prevent runaway queries
DEFAULT_QUERY_TIMEOUT = 30.0  # seconds


class QueryTimeoutError(RuntimeError):
    """Exception raised when a query exceeds its timeout."""

    def __init__(self, seconds: float):
        self.timeout = seconds
        super().__init__(f"Query exceeded timeout of {seconds}s")


class ColumnInfo(TypedDict):
    """Type definition for column information."""

    data_type: str
    nullable: bool
    default: str | None
    max_length: int | None
    precision: int | None
    scale: int | None
    comment: str | None
    type: str


class ForeignKeyInfo(TypedDict):
    """Type definition for foreign key information."""

    column: str
    references: dict[str, str]  # {"table": "schema.table", "column": "column_name"}


class IndexInfo(TypedDict):
    """Type definition for index information."""

    name: str
    columns: list[str]  # ordered
    unique: bool
    type: str | None  # btree, gin, FULLTEXT, etc. None if unknown


class SchemaInfo(TypedDict):
    """Type definition for schema information."""

    schema: str
    name: str
    type: str
    comment: str | None
    columns: dict[str, ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]
    indexes: list[IndexInfo]


class BaseDatabaseConnection(ABC):
    """Abstract base class for database connections."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None
        self._excluded_schemas: list[str] = []

    @property
    @abstractmethod
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name for this database."""
        pass

    @property
    def display_name(self) -> str:
        """Return the human-readable name for this database type."""
        return "database"

    @abstractmethod
    async def get_pool(self):
        """Get or create connection pool."""
        pass

    @abstractmethod
    async def close(self):
        """Close the connection pool."""
        pass

    @abstractmethod
    async def execute_query(
        self, query: str, *args, timeout: float | None = None, commit: bool = False
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        By default, all queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.

        If commit=True, the transaction will be committed on success (for DML/DDL
        in dangerous mode).

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default_timeout)
            commit: If True, commit the transaction on success instead of rolling back
        """
        pass

    def set_excluded_schemas(self, schemas: Iterable[str] | None) -> None:
        """Set schemas to exclude from introspection for this connection."""
        self._excluded_schemas = []
        if not schemas:
            return

        seen: set[str] = set()
        for schema in schemas:
            clean = schema.strip()
            if not clean:
                continue
            if clean in seen:
                continue
            seen.add(clean)
            self._excluded_schemas.append(clean)

    @property
    def excluded_schemas(self) -> list[str]:
        """Return list of excluded schemas for this connection."""
        return list(self._excluded_schemas)


class BaseSchemaIntrospector(ABC):
    """Abstract base class for database-specific schema introspection."""

    @abstractmethod
    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """Get tables information for the specific database type."""
        pass

    @abstractmethod
    async def get_columns_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get columns information for the specific database type."""
        pass

    @abstractmethod
    async def get_foreign_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get foreign keys information for the specific database type."""
        pass

    @abstractmethod
    async def get_primary_keys_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get primary keys information for the specific database type."""
        pass

    @abstractmethod
    async def get_indexes_info(
        self, connection, tables: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Get indexes information for the specific database type."""
        pass

    @abstractmethod
    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information."""
        pass

    @staticmethod
    def _merge_excluded_schemas(
        connection: BaseDatabaseConnection,
        defaults: Iterable[str],
        env_var: str | None = None,
    ) -> list[str]:
        """Combine default, connection, and environment schema exclusions."""

        combined: list[str] = []
        seen: set[str] = set()

        def _add(items: Iterable[str]) -> None:
            for item in items:
                name = item.strip()
                if not name:
                    continue
                if name in seen:
                    continue
                seen.add(name)
                combined.append(name)

        _add(defaults)
        _add(getattr(connection, "excluded_schemas", []) or [])

        if env_var:
            raw = os.getenv(env_var, "")
            if raw:
                # Support comma-separated values
                values = [part.strip() for part in raw.split(",")]
                _add(values)

        return combined
