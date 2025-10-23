"""Tests for PostgreSQL schema introspection behavior."""

import pytest

from sqlsaber.database.postgresql import PostgreSQLSchemaIntrospector


class _FakeConn:
    def __init__(self):
        self.last_query: str | None = None
        self.last_params: list | None = None

    async def fetch(self, query, *args):
        # Record the query and params for assertions
        self.last_query = query
        self.last_params = list(args)
        return []


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


class _FakePGConnection:
    def __init__(self):
        self.pool = _FakePool(_FakeConn())
        self._excluded_schemas: list[str] = []

    async def get_pool(self):
        return self.pool

    def set_excluded_schemas(self, schemas: list[str]) -> None:
        self._excluded_schemas = schemas

    @property
    def excluded_schemas(self) -> list[str]:
        return self._excluded_schemas


@pytest.mark.asyncio
async def test_pg_excluded_schemas_defaults(monkeypatch):
    """Ensure default excluded schemas include TimescaleDB internals."""
    # Ensure env var not set
    monkeypatch.delenv("SQLSABER_PG_EXCLUDE_SCHEMAS", raising=False)
    conn = _FakePGConnection()
    insp = PostgreSQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(conn)
    # Defaults
    assert "pg_catalog" in excluded
    assert "information_schema" in excluded
    assert "_timescaledb_internal" in excluded


@pytest.mark.asyncio
async def test_pg_excluded_schemas_env_extension(monkeypatch):
    """Environment variable should extend excluded schema list."""
    monkeypatch.setenv("SQLSABER_PG_EXCLUDE_SCHEMAS", "myschema , other_schema")
    conn = _FakePGConnection()
    insp = PostgreSQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(conn)
    assert "myschema" in excluded
    assert "other_schema" in excluded


@pytest.mark.asyncio
async def test_pg_excluded_schemas_connection_extension(monkeypatch):
    """Connection-level excludes should be merged with defaults."""
    monkeypatch.delenv("SQLSABER_PG_EXCLUDE_SCHEMAS", raising=False)
    conn = _FakePGConnection()
    conn.set_excluded_schemas(["custom_schema"])
    insp = PostgreSQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(conn)
    assert "custom_schema" in excluded
    assert "_timescaledb_internal" in excluded


@pytest.mark.asyncio
async def test_pg_excluded_schemas_preserve_case(monkeypatch):
    """Ensure case-sensitive schema names are preserved in exclusions."""
    monkeypatch.delenv("SQLSABER_PG_EXCLUDE_SCHEMAS", raising=False)
    conn = _FakePGConnection()
    conn.set_excluded_schemas(["Sales", "sales"])
    insp = PostgreSQLSchemaIntrospector()
    excluded = insp._get_excluded_schemas(conn)
    assert "Sales" in excluded
    assert "sales" in excluded


@pytest.mark.asyncio
async def test_get_tables_info_excludes_and_patterns(monkeypatch):
    """Verify parameter order: exclusions first, then pattern parts."""
    monkeypatch.delenv("SQLSABER_PG_EXCLUDE_SCHEMAS", raising=False)
    conn = _FakePGConnection()
    insp = PostgreSQLSchemaIntrospector()

    # With explicit schema.table pattern
    await insp.get_tables_info(conn, table_pattern="public.users")
    params = conn.pool.conn.last_params or []
    assert params[-2:] == ["public", "users"]
    assert "table_schema NOT IN (" in (conn.pool.conn.last_query or "")


@pytest.mark.asyncio
async def test_list_tables_info_excludes(monkeypatch):
    """Ensure list_tables_info also uses excluded schemas."""
    monkeypatch.setenv("SQLSABER_PG_EXCLUDE_SCHEMAS", "custom_exclude")
    conn = _FakePGConnection()
    insp = PostgreSQLSchemaIntrospector()

    await insp.list_tables_info(conn)
    params = conn.pool.conn.last_params or []
    # Expect default three + custom_exclude
    assert params[-1] == "custom_exclude"
    assert "table_schema NOT IN (" in (conn.pool.conn.last_query or "")
