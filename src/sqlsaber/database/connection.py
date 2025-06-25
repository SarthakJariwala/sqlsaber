"""Database connection management."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs
import ssl

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
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context from connection string parameters."""
        parsed = urlparse(self.connection_string)
        if not parsed.query:
            return None

        params = parse_qs(parsed.query)
        ssl_mode = params.get("sslmode", [None])[0]

        if not ssl_mode or ssl_mode == "disable":
            return None

        # Create SSL context based on mode
        if ssl_mode in ["require", "verify-ca", "verify-full"]:
            ssl_context = ssl.create_default_context()

            # Configure certificate verification
            if ssl_mode == "require":
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            elif ssl_mode == "verify-ca":
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_REQUIRED
            elif ssl_mode == "verify-full":
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED

            # Load certificates if provided
            ssl_ca = params.get("sslrootcert", [None])[0]
            ssl_cert = params.get("sslcert", [None])[0]
            ssl_key = params.get("sslkey", [None])[0]

            if ssl_ca:
                ssl_context.load_verify_locations(ssl_ca)

            if ssl_cert and ssl_key:
                ssl_context.load_cert_chain(ssl_cert, ssl_key)

            return ssl_context

        return None

    async def get_pool(self) -> asyncpg.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            # Create pool with SSL context if configured
            if self._ssl_context:
                self._pool = await asyncpg.create_pool(
                    self.connection_string,
                    min_size=1,
                    max_size=10,
                    ssl=self._ssl_context,
                )
            else:
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

        # Parse SSL parameters
        self.ssl_params = {}
        if parsed.query:
            params = parse_qs(parsed.query)

            ssl_mode = params.get("ssl_mode", [None])[0]
            if ssl_mode:
                # Map SSL modes to aiomysql SSL parameters
                if ssl_mode.upper() == "DISABLED":
                    self.ssl_params["ssl"] = None
                elif ssl_mode.upper() in [
                    "PREFERRED",
                    "REQUIRED",
                    "VERIFY_CA",
                    "VERIFY_IDENTITY",
                ]:
                    ssl_context = ssl.create_default_context()

                    if ssl_mode.upper() == "REQUIRED":
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_NONE
                    elif ssl_mode.upper() == "VERIFY_CA":
                        ssl_context.check_hostname = False
                        ssl_context.verify_mode = ssl.CERT_REQUIRED
                    elif ssl_mode.upper() == "VERIFY_IDENTITY":
                        ssl_context.check_hostname = True
                        ssl_context.verify_mode = ssl.CERT_REQUIRED

                    # Load certificates if provided
                    ssl_ca = params.get("ssl_ca", [None])[0]
                    ssl_cert = params.get("ssl_cert", [None])[0]
                    ssl_key = params.get("ssl_key", [None])[0]

                    if ssl_ca:
                        ssl_context.load_verify_locations(ssl_ca)

                    if ssl_cert and ssl_key:
                        ssl_context.load_cert_chain(ssl_cert, ssl_key)

                    self.ssl_params["ssl"] = ssl_context

    async def get_pool(self) -> aiomysql.Pool:
        """Get or create connection pool."""
        if self._pool is None:
            pool_kwargs = {
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "db": self.database,
                "minsize": 1,
                "maxsize": 10,
                "autocommit": False,
            }

            # Add SSL parameters if configured
            pool_kwargs.update(self.ssl_params)

            self._pool = await aiomysql.create_pool(**pool_kwargs)
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
