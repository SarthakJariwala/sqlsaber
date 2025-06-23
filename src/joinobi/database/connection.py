"""Database connection management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import aiomysql
import aiosqlite
import asyncpg


class BaseDatabaseConnection(ABC):
    """Abstract base class for database connections."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._pool = None

    @abstractmethod
    async def get_pool(self):
        """Get or create connection pool."""
        pass

    @abstractmethod
    async def close(self):
        """Close the connection pool."""
        pass

    @abstractmethod
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        pass


class PostgreSQLConnection(BaseDatabaseConnection):
    """PostgreSQL database connection using asyncpg."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool: Optional[asyncpg.Pool] = None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.connection_string, min_size=1, max_size=10
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            # Start a transaction that we'll always rollback
            transaction = conn.transaction()
            await transaction.start()

            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            finally:
                # Always rollback to ensure no changes are committed
                await transaction.rollback()


class MySQLConnection(BaseDatabaseConnection):
    """MySQL database connection using aiomysql."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool: Optional[aiomysql.Pool] = None
        self._parse_connection_string()

    def _parse_connection_string(self):
        """Parse MySQL connection string into components."""
        parsed = urlparse(self.connection_string)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3306
        self.database = parsed.path.lstrip("/") if parsed.path else ""
        self.user = parsed.username or ""
        self.password = parsed.password or ""

    async def get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.database,
                minsize=1,
                maxsize=10,
                autocommit=False,
            )
        return self._pool

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # Start transaction
                await conn.begin()
                try:
                    await cursor.execute(query, args if args else None)
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
                finally:
                    # Always rollback to ensure no changes are committed
                    await conn.rollback()


class SQLiteConnection(BaseDatabaseConnection):
    """SQLite database connection using aiosqlite."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        # Extract database path from sqlite:///path format
        self.database_path = connection_string.replace("sqlite:///", "")

    async def get_pool(self):
        """SQLite doesn't use connection pooling, return database path."""
        return self.database_path

    async def close(self):
        """SQLite connections are created per query, no persistent pool to close."""
        pass

    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        async with aiosqlite.connect(self.database_path) as conn:
            # Enable row factory for dict-like access
            conn.row_factory = aiosqlite.Row

            # Start transaction
            await conn.execute("BEGIN")
            try:
                cursor = await conn.execute(query, args if args else ())
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
            finally:
                # Always rollback to ensure no changes are committed
                await conn.rollback()


def DatabaseConnection(connection_string: str) -> BaseDatabaseConnection:
    """Factory function to create appropriate database connection based on connection string."""
    if connection_string.startswith("postgresql://"):
        return PostgreSQLConnection(connection_string)
    elif connection_string.startswith("mysql://"):
        return MySQLConnection(connection_string)
    elif connection_string.startswith("sqlite:///"):
        return SQLiteConnection(connection_string)
    else:
        raise ValueError(
            f"Unsupported database type in connection string: {connection_string}"
        )
