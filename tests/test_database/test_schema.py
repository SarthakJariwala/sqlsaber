"""Tests for schema introspection."""

import duckdb
import pytest

from sqlsaber.database import DuckDBConnection
from sqlsaber.database.schema import (
    DuckDBSchemaIntrospector,
    SchemaManager,
)


@pytest.mark.asyncio
async def test_duckdb_schema_manager(tmp_path):
    """Ensure DuckDB schema introspection surfaces tables and relationships."""
    db_path = tmp_path / "introspection.duckdb"

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
        conn.execute(
            "CREATE TABLE orders (id INTEGER, user_id INTEGER, FOREIGN KEY(user_id) REFERENCES users(id));"
        )
        conn.execute("CREATE UNIQUE INDEX idx_users_name ON users(name);")
    finally:
        conn.close()

    db_conn = DuckDBConnection(f"duckdb:///{db_path}")
    schema_manager = SchemaManager(db_conn)

    assert isinstance(schema_manager.introspector, DuckDBSchemaIntrospector)

    tables = await schema_manager.list_tables()
    table_names = {table["full_name"] for table in tables["tables"]}
    assert "main.users" in table_names
    assert "main.orders" in table_names

    schema_info = await schema_manager.get_schema_info()
    users_info = schema_info["main.users"]
    orders_info = schema_info["main.orders"]

    assert "id" in users_info["columns"]
    assert "INTEGER" in users_info["columns"]["id"]["data_type"].upper()
    assert "id" in users_info["primary_keys"]
    assert any(idx["name"] == "idx_users_name" for idx in users_info["indexes"])

    assert any(fk["column"] == "user_id" for fk in orders_info["foreign_keys"])
