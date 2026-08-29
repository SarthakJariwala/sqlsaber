"""Tests for schema display functionality and field mappings."""

import aiosqlite
import duckdb
import pytest

from sqlsaber.database import DuckDBConnection, SQLiteConnection
from sqlsaber.database.csv import CSVConnection
from sqlsaber.database.duckdb import DuckDBSchemaIntrospector
from sqlsaber.database.schema import SchemaManager
from sqlsaber.database.sqlite import SQLiteSchemaIntrospector
from sqlsaber.render.markdown_text import md_of
from sqlsaber.tools.renderer import ToolRenderer, core_display_registry


class TestSchemaDisplayMappings:
    """Test schema display field mapping and backwards compatibility."""

    @pytest.mark.asyncio
    async def test_table_type_mapping_sqlite(self, tmp_path):
        """Test that table types are correctly mapped for display."""
        db_path = tmp_path / "test_types.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
            await conn.execute("CREATE VIEW test_view AS SELECT * FROM test_table")
            await conn.commit()

        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        tables = await schema_manager.list_tables()

        table_data = tables["tables"]
        assert len(table_data) == 2

        for table in table_data:
            assert "name" in table
            assert "schema" in table
            assert "full_name" in table
            assert "table_type" in table
            assert "type" in table

            assert table["type"] == table["table_type"]

        table_types = {table["name"]: table["type"] for table in table_data}
        assert "table" in table_types.values() or "view" in table_types.values()

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_column_type_mapping_sqlite(self, tmp_path):
        """Test that column types are correctly mapped for schema display."""
        db_path = tmp_path / "test_columns.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT DEFAULT 'unknown@example.com',
                    age INTEGER DEFAULT 18,
                    active BOOLEAN DEFAULT TRUE,
                    score REAL DEFAULT 0.0
                )
            """)
            await conn.commit()

        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        schema_info = await schema_manager.get_schema_info()

        assert "main.test_table" in schema_info
        table_info = schema_info["main.test_table"]

        columns = table_info["columns"]
        for col_name, col_info in columns.items():
            assert "data_type" in col_info
            assert "type" in col_info

            assert col_info["type"] == col_info["data_type"]

        assert columns["name"]["type"] == "TEXT"
        assert columns["id"]["type"] == "INTEGER"
        assert columns["active"]["type"] == "BOOLEAN"
        assert columns["score"]["type"] == "REAL"

        assert columns["email"]["default"] == "'unknown@example.com'"
        assert columns["age"]["default"] == "18"

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_schema_display_integration(self, tmp_path):
        """Test end-to-end schema display with field mappings."""
        db_path = tmp_path / "test_display.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("""
                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    price DECIMAL(10,2) DEFAULT 0.00,
                    active BOOLEAN DEFAULT TRUE
                )
            """)
            await conn.commit()

        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)
        schema_info = await schema_manager.get_schema_info()

        output = md_of(
            ToolRenderer(core_display_registry()).result(
                "introspect_schema", schema_info
            )
        )

        assert "TEXT" in output
        assert "INTEGER" in output
        assert "BOOLEAN" in output

        assert "0.00" in output
        assert "TRUE" in output

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_schema_display_includes_comments(self, tmp_path):
        """Ensure schema display renders table and column comments when present."""
        db_path = tmp_path / "test_comments.duckdb"

        conn = duckdb.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price DECIMAL);"
            )
            conn.execute("COMMENT ON TABLE products IS 'Product catalog';")
            conn.execute(
                "COMMENT ON COLUMN products.id IS 'Unique product identifier';"
            )
            conn.execute("COMMENT ON COLUMN products.name IS 'Product name';")
        finally:
            conn.close()

        db_conn = DuckDBConnection(f"duckdb:///{db_path}")
        schema_manager = SchemaManager(db_conn)
        schema_info = await schema_manager.get_schema_info()

        output = md_of(
            ToolRenderer(core_display_registry()).result(
                "introspect_schema", schema_info
            )
        )

        assert "Unique product identifier" in output
        assert "Product name" in output

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_table_list_display_integration(self, tmp_path):
        """Test end-to-end table list display with type mappings."""
        db_path = tmp_path / "test_table_display.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            await conn.execute(
                "CREATE VIEW active_users AS SELECT * FROM users WHERE id > 0"
            )
            await conn.commit()

        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)
        tables = await schema_manager.list_tables()

        output = md_of(
            ToolRenderer(core_display_registry()).result("list_tables", tables)
        )

        lines = output.split("\n")
        type_lines = [
            line
            for line in lines
            if "|" in line and ("users" in line or "active_users" in line)
        ]

        has_table_type = any("table" in line for line in type_lines)
        has_view_type = any("view" in line for line in type_lines)

        assert has_table_type or has_view_type

        await db_conn.close()


class TestDatabaseTypeSchemaIntrospection:
    """Test schema introspection across different database types."""

    @pytest.mark.asyncio
    async def test_duckdb_schema_introspection(self, tmp_path):
        """Test DuckDB-specific schema introspection features."""
        db_path = tmp_path / "test_duckdb.db"

        conn = duckdb.connect(str(db_path))
        try:
            conn.execute("""
                CREATE TABLE products (
                    id INTEGER PRIMARY KEY,
                    name VARCHAR(100) DEFAULT 'Unknown Product',
                    price DECIMAL(10,2) DEFAULT 0.00,
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE VIEW expensive_products AS SELECT * FROM products WHERE price > 100"
            )
            conn.execute("CREATE INDEX idx_product_name ON products(name)")
        finally:
            conn.close()

        db_conn = DuckDBConnection(f"duckdb:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        tables = await schema_manager.list_tables()
        table_names = {table["name"] for table in tables["tables"]}
        assert "products" in table_names
        assert "expensive_products" in table_names

        schema_info = await schema_manager.get_schema_info()
        products_info = schema_info["main.products"]

        columns = products_info["columns"]
        assert columns["name"]["data_type"] == "VARCHAR"
        assert columns["price"]["data_type"] == "DECIMAL(10,2)"
        assert columns["active"]["data_type"] == "BOOLEAN"

        assert columns["name"]["default"] is not None
        assert columns["price"]["default"] is not None

        assert "id" in products_info["primary_keys"]

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_csv_schema_introspection(self, tmp_path):
        """Test CSV schema introspection (using DuckDB backend)."""
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text(
            "id,name,age,active\n1,Alice,25,true\n2,Bob,30,false\n", encoding="utf-8"
        )

        csv_conn = CSVConnection(f"csv:///{csv_path}")
        schema_manager = SchemaManager(csv_conn)

        tables = await schema_manager.list_tables()
        table_names = {table["name"] for table in tables["tables"]}
        assert "test_data" in table_names

        schema_info = await schema_manager.get_schema_info()
        assert "main.test_data" in schema_info

        columns = schema_info["main.test_data"]["columns"]
        assert "id" in columns
        assert "name" in columns
        assert "age" in columns
        assert "active" in columns

        await csv_conn.close()


class TestSchemaIntrospectorMapping:
    """Test that schema introspectors are correctly mapped to connection types."""

    @pytest.mark.asyncio
    async def test_sqlite_introspector_mapping(self, tmp_path):
        """Test SQLite connection gets correct introspector."""

        db_path = tmp_path / "test.db"
        db_path.touch()

        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        assert isinstance(schema_manager.introspector, SQLiteSchemaIntrospector)
        await db_conn.close()

    @pytest.mark.asyncio
    async def test_duckdb_introspector_mapping(self, tmp_path):
        """Test DuckDB connection gets correct introspector."""

        db_path = tmp_path / "test.duckdb"

        db_conn = DuckDBConnection(f"duckdb:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        assert isinstance(schema_manager.introspector, DuckDBSchemaIntrospector)
        await db_conn.close()

    @pytest.mark.asyncio
    async def test_csv_introspector_mapping(self, tmp_path):
        """Test CSV connection gets DuckDB introspector."""

        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n", encoding="utf-8")

        csv_conn = CSVConnection(f"csv:///{csv_path}")
        schema_manager = SchemaManager(csv_conn)

        assert isinstance(schema_manager.introspector, DuckDBSchemaIntrospector)
        await csv_conn.close()
