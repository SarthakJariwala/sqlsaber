"""Database connection management."""

import asyncio
import ssl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import aiomysql
import aiosqlite
import asyncpg
import duckdb

# Default query timeout to prevent runaway queries
DEFAULT_QUERY_TIMEOUT = 30.0  # seconds


class QueryTimeoutError(RuntimeError):
    """Exception raised when a query exceeds its timeout."""

    def __init__(self, seconds: float):
        self.timeout = seconds
        super().__init__(f"Query exceeded timeout of {seconds}s")


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

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            # Start a transaction that we'll always rollback
            transaction = conn.transaction()
            await transaction.start()

            try:
                # Set server-side timeout if specified
                if effective_timeout:
                    await conn.execute(
                        f"SET LOCAL statement_timeout = {int(effective_timeout * 1000)}"
                    )

                # Execute query with client-side timeout
                if effective_timeout:
                    rows = await asyncio.wait_for(
                        conn.fetch(query, *args), timeout=effective_timeout
                    )
                else:
                    rows = await conn.fetch(query, *args)

                return [dict(row) for row in rows]
            except asyncio.TimeoutError as exc:
                raise QueryTimeoutError(effective_timeout or 0) from exc
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

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        pool = await self.get_pool()

        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cursor:
                # Start transaction
                await conn.begin()
                try:
                    # Set server-side timeout if specified
                    if effective_timeout:
                        await cursor.execute(
                            f"SET SESSION MAX_EXECUTION_TIME = {int(effective_timeout * 1000)}"
                        )

                    # Execute query with client-side timeout
                    if effective_timeout:
                        await asyncio.wait_for(
                            cursor.execute(query, args if args else None),
                            timeout=effective_timeout,
                        )
                        rows = await asyncio.wait_for(
                            cursor.fetchall(), timeout=effective_timeout
                        )
                    else:
                        await cursor.execute(query, args if args else None)
                        rows = await cursor.fetchall()

                    return [dict(row) for row in rows]
                except asyncio.TimeoutError as exc:
                    raise QueryTimeoutError(effective_timeout or 0) from exc
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

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT

        async with aiosqlite.connect(self.database_path) as conn:
            # Enable row factory for dict-like access
            conn.row_factory = aiosqlite.Row

            # Start transaction
            await conn.execute("BEGIN")
            try:
                # Execute query with client-side timeout (SQLite has no server-side timeout)
                if effective_timeout:
                    cursor = await asyncio.wait_for(
                        conn.execute(query, args if args else ()),
                        timeout=effective_timeout,
                    )
                    rows = await asyncio.wait_for(
                        cursor.fetchall(), timeout=effective_timeout
                    )
                else:
                    cursor = await conn.execute(query, args if args else ())
                    rows = await cursor.fetchall()

                return [dict(row) for row in rows]
            except asyncio.TimeoutError as exc:
                raise QueryTimeoutError(effective_timeout or 0) from exc
            finally:
                # Always rollback to ensure no changes are committed
                await conn.rollback()


def _execute_duckdb_transaction(
    conn: duckdb.DuckDBPyConnection, query: str, args: tuple[Any, ...]
) -> list[dict[str, Any]]:
    """Run a DuckDB query inside a transaction and return list of dicts."""
    conn.execute("BEGIN TRANSACTION")
    try:
        if args:
            conn.execute(query, args)
        else:
            conn.execute(query)

        if conn.description is None:
            rows: list[dict[str, Any]] = []
        else:
            columns = [col[0] for col in conn.description]
            data = conn.fetchall()
            rows = [dict(zip(columns, row)) for row in data]

        conn.execute("ROLLBACK")
        return rows
    except Exception:
        conn.execute("ROLLBACK")
        raise


class CSVConnection(BaseDatabaseConnection):
    """CSV file connection using DuckDB per query."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)

        raw_path = connection_string.replace("csv:///", "", 1)
        self.csv_path = raw_path.split("?", 1)[0]

        self.delimiter = ","
        self.encoding = "utf-8"
        self.has_header = True

        parsed = urlparse(connection_string)
        if parsed.query:
            params = parse_qs(parsed.query)
            self.delimiter = params.get("delimiter", [self.delimiter])[0]
            self.encoding = params.get("encoding", [self.encoding])[0]
            self.has_header = params.get("header", ["true"])[0].lower() == "true"

        self.table_name = Path(self.csv_path).stem or "csv_table"

    async def get_pool(self):
        """CSV connections do not maintain a pool."""
        return None

    async def close(self):
        """No persistent resources to close for CSV connections."""
        pass

    def _quote_identifier(self, identifier: str) -> str:
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _quote_literal(self, value: str) -> str:
        escaped = value.replace("'", "''")
        return f"'{escaped}'"

    def _normalized_encoding(self) -> str | None:
        encoding = (self.encoding or "").strip()
        if not encoding or encoding.lower() == "utf-8":
            return None
        return encoding.replace("-", "").replace("_", "").upper()

    def _create_view(self, conn: duckdb.DuckDBPyConnection) -> None:
        header_literal = "TRUE" if self.has_header else "FALSE"
        option_parts = [f"HEADER={header_literal}"]

        if self.delimiter:
            option_parts.append(f"DELIM={self._quote_literal(self.delimiter)}")

        encoding = self._normalized_encoding()
        if encoding:
            option_parts.append(f"ENCODING={self._quote_literal(encoding)}")

        options_sql = ""
        if option_parts:
            options_sql = ", " + ", ".join(option_parts)

        base_relation_sql = (
            f"read_csv_auto({self._quote_literal(self.csv_path)}{options_sql})"
        )

        create_view_sql = (
            f"CREATE VIEW {self._quote_identifier(self.table_name)} AS "
            f"SELECT * FROM {base_relation_sql}"
        )
        conn.execute(create_view_sql)

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT
        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(":memory:")
            try:
                self._create_view(conn)
                return _execute_duckdb_transaction(conn, query, args_tuple)
            finally:
                conn.close()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc


class DuckDBConnection(BaseDatabaseConnection):
    """DuckDB database connection using duckdb Python API."""

    def __init__(self, connection_string: str):
        super().__init__(connection_string)
        if connection_string.startswith("duckdb:///"):
            db_path = connection_string.replace("duckdb:///", "", 1)
        elif connection_string.startswith("duckdb://"):
            db_path = connection_string.replace("duckdb://", "", 1)
        else:
            db_path = connection_string

        self.database_path = db_path or ":memory:"

    async def get_pool(self):
        """DuckDB creates connections per query, return database path."""
        return self.database_path

    async def close(self):
        """DuckDB connections are created per query, no persistent pool to close."""
        pass

    async def execute_query(
        self, query: str, *args, timeout: float | None = None
    ) -> list[dict[str, Any]]:
        """Execute a query and return results as list of dicts.

        All queries run in a transaction that is rolled back at the end,
        ensuring no changes are persisted to the database.
        """
        effective_timeout = timeout or DEFAULT_QUERY_TIMEOUT

        args_tuple = tuple(args) if args else tuple()

        def _run_query() -> list[dict[str, Any]]:
            conn = duckdb.connect(self.database_path)
            try:
                return _execute_duckdb_transaction(conn, query, args_tuple)
            finally:
                conn.close()

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_query), timeout=effective_timeout
            )
        except asyncio.TimeoutError as exc:
            raise QueryTimeoutError(effective_timeout or 0) from exc


def DatabaseConnection(connection_string: str) -> BaseDatabaseConnection:
    """Factory function to create appropriate database connection based on connection string."""
    if connection_string.startswith("postgresql://"):
        return PostgreSQLConnection(connection_string)
    elif connection_string.startswith("mysql://"):
        return MySQLConnection(connection_string)
    elif connection_string.startswith("sqlite:///"):
        return SQLiteConnection(connection_string)
    elif connection_string.startswith("duckdb://"):
        return DuckDBConnection(connection_string)
    elif connection_string.startswith("csv:///"):
        return CSVConnection(connection_string)
    else:
        raise ValueError(
            f"Unsupported database type in connection string: {connection_string}"
        )
