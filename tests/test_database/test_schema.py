"""Tests for schema introspection."""

import aiosqlite
import duckdb
import pytest

from sqlsaber.database import DuckDBConnection, SQLiteConnection
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


@pytest.mark.asyncio
async def test_duckdb_comments(tmp_path):
    """Test that DuckDB table and column comments are retrieved correctly."""
    db_path = tmp_path / "comments.duckdb"

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price DECIMAL);"
        )
        conn.execute("COMMENT ON TABLE products IS 'Product catalog';")
        conn.execute("COMMENT ON COLUMN products.id IS 'Unique product identifier';")
        conn.execute("COMMENT ON COLUMN products.name IS 'Product name';")
        conn.execute("COMMENT ON COLUMN products.price IS 'Product price in USD';")
    finally:
        conn.close()

    db_conn = DuckDBConnection(f"duckdb:///{db_path}")
    schema_manager = SchemaManager(db_conn)

    # Test list_tables includes comments
    tables = await schema_manager.list_tables()
    products_table = next(t for t in tables["tables"] if t["table_name"] == "products")
    assert products_table["table_comment"] == "Product catalog"

    # Test get_schema_info includes comments
    schema_info = await schema_manager.get_schema_info()
    products_info = schema_info["main.products"]

    assert products_info["comment"] == "Product catalog"
    assert products_info["columns"]["id"]["comment"] == "Unique product identifier"
    assert products_info["columns"]["name"]["comment"] == "Product name"
    assert products_info["columns"]["price"]["comment"] == "Product price in USD"


@pytest.mark.asyncio
async def test_sqlite_no_comments(tmp_path):
    """Verify that SQLite returns None for comments since it doesn't support them."""

    db_path = tmp_path / "no_comments.db"

    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT);")
        await conn.commit()

    db_conn = SQLiteConnection(f"sqlite:///{db_path}")
    schema_manager = SchemaManager(db_conn)

    # Test list_tables has None for comments
    tables = await schema_manager.list_tables()
    items_table = next(t for t in tables["tables"] if t["table_name"] == "items")
    assert items_table["table_comment"] is None

    # Test get_schema_info has None for comments
    schema_info = await schema_manager.get_schema_info()
    items_info = schema_info["main.items"]

    assert items_info["comment"] is None
    assert items_info["columns"]["id"]["comment"] is None
    assert items_info["columns"]["name"]["comment"] is None
