"""Base classes and type definitions for database connections and schema introspection."""

from abc import ABC, abstractmethod
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
    columns: dict[str, ColumnInfo]
    primary_keys: list[str]
    foreign_keys: list[ForeignKeyInfo]
    indexes: list[IndexInfo]


class BaseDatabaseConnection(ABC):
    """Abstract base class for database connections."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None

    @property
    @abstractmethod
    def sqlglot_dialect(self) -> str:
        """Return the sqlglot dialect name for this database."""
        pass

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
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.

        Args:
            query: SQL query to execute
            *args: Query parameters
            timeout: Query timeout in seconds (overrides default_timeout)
        """
        pass


class BaseSchemaIntrospector(ABC):
    """Abstract base class for database-specific schema introspection."""

    @abstractmethod
    async def get_tables_info(
        self, connection, table_pattern: str | None = None
    ) -> dict[str, Any]:
        """Get tables information for the specific database type."""
        pass

    @abstractmethod
    async def get_columns_info(self, connection, tables: list) -> list:
        """Get columns information for the specific database type."""
        pass

    @abstractmethod
    async def get_foreign_keys_info(self, connection, tables: list) -> list:
        """Get foreign keys information for the specific database type."""
        pass

    @abstractmethod
    async def get_primary_keys_info(self, connection, tables: list) -> list:
        """Get primary keys information for the specific database type."""
        pass

    @abstractmethod
    async def get_indexes_info(self, connection, tables: list) -> list:
        """Get indexes information for the specific database type."""
        pass

    @abstractmethod
    async def list_tables_info(self, connection) -> list[dict[str, Any]]:
        """Get list of tables with basic information."""
        pass
