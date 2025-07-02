"""Database connection management."""

from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse, parse_qs
import ssl
from pathlib import Path

import aiomysql
import aiosqlite
import asyncpg
import pandas as pd


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
    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        pass


class PostgreSQLConnection(BaseDatabaseConnection):
    """PostgreSQL database connection using asyncpg."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        self._pool: asyncpg.Pool | None = None
        self._ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext | None:
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

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
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
        self._pool: aiomysql.Pool | None = None
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

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
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

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
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


class CSVConnection(BaseDatabaseConnection):
    """CSV file connection using in-memory SQLite database."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

        # Parse CSV file path from connection string
        self.csv_path = connection_string.replace("csv:///", "")

        # CSV parsing options
        self.delimiter = ","
        self.encoding = "utf-8"
        self.has_header = True

        # Parse additional options from connection string
        parsed = urlparse(connection_string)
        if parsed.query:
            params = parse_qs(parsed.query)
            self.delimiter = params.get("delimiter", [","])[0]
            self.encoding = params.get("encoding", ["utf-8"])[0]
            self.has_header = params.get("header", ["true"])[0].lower() == "true"

        # Table name derived from filename
        self.table_name = Path(self.csv_path).stem

        # Initialize connection and flag to track if CSV is loaded
        self._conn = None
        self._csv_loaded = False

    async def get_pool(self):
        """Get or create the in-memory database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(":memory:")
            self._conn.row_factory = aiosqlite.Row
            await self._load_csv_data()
        return self._conn

    async def close(self):
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            self._csv_loaded = False

    async def _load_csv_data(self):
        """Load CSV data into the in-memory SQLite database."""
        if self._csv_loaded or not self._conn:
            return

        try:
            # Read CSV file using pandas
            df = pd.read_csv(
                self.csv_path,
                delimiter=self.delimiter,
                encoding=self.encoding,
                header=0 if self.has_header else None,
            )

            # If no header, create column names
            if not self.has_header:
                df.columns = [f"column_{i}" for i in range(len(df.columns))]

            # Create table with proper column types
            columns_sql = []
            for col in df.columns:
                # Infer SQLite type from pandas dtype
                dtype = df[col].dtype
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "REAL"
                elif pd.api.types.is_bool_dtype(dtype):
                    sql_type = "INTEGER"  # SQLite doesn't have BOOLEAN
                else:
                    sql_type = "TEXT"

                columns_sql.append(f'"{col}" {sql_type}')

            create_table_sql = (
                f'CREATE TABLE "{self.table_name}" ({", ".join(columns_sql)})'
            )
            await self._conn.execute(create_table_sql)

            # Insert data row by row
            placeholders = ", ".join(["?" for _ in df.columns])
            insert_sql = f'INSERT INTO "{self.table_name}" VALUES ({placeholders})'

            for _, row in df.iterrows():
                # Convert pandas values to Python native types
                values = []
                for val in row:
                    if pd.isna(val):
                        values.append(None)
                    elif isinstance(val, (pd.Timestamp, pd.Timedelta)):
                        values.append(str(val))
                    else:
                        values.append(val)

                await self._conn.execute(insert_sql, values)

            await self._conn.commit()
            self._csv_loaded = True

        except Exception as e:
            raise ValueError(f"Error loading CSV file '{self.csv_path}': {str(e)}")

    async def execute_query(self, query: str, *args) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        conn = await self.get_pool()

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
    elif connection_string.startswith("csv:///"):
        return CSVConnection(connection_string)
    else:
        raise ValueError(
            f"Unsupported database type in connection string: {connection_string}"
        )
