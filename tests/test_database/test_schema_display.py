"""Tests for schema display functionality and field mappings."""

from io import StringIO

import aiosqlite
import duckdb
import pytest

from sqlsaber.cli.display import DisplayManager
from sqlsaber.database import CSVConnection, DuckDBConnection, SQLiteConnection
from sqlsaber.database.duckdb import DuckDBSchemaIntrospector
from sqlsaber.database.schema import SchemaManager
from sqlsaber.database.sqlite import SQLiteSchemaIntrospector
from sqlsaber.theme.manager import create_console


class TestSchemaDisplayMappings:
    """Test schema display field mapping and backwards compatibility."""

    @pytest.mark.asyncio
    async def test_table_type_mapping_sqlite(self, tmp_path):
        """Test that table types are correctly mapped for display."""
        # Create test database with table and view
        db_path = tmp_path / "test_types.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
            await conn.execute("CREATE VIEW test_view AS SELECT * FROM test_table")
            await conn.commit()

        # Test schema manager
        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        tables = await schema_manager.list_tables()

        # Verify backwards compatibility fields exist
        table_data = tables["tables"]
        assert len(table_data) == 2

        for table in table_data:
            # Check all expected fields exist
            assert "name" in table
            assert "schema" in table
            assert "full_name" in table
            assert "table_type" in table  # Original field
            assert "type" in table  # Mapped field for display

            # Verify mapping is correct
            assert table["type"] == table["table_type"]

        # Check specific types
        table_types = {table["name"]: table["type"] for table in table_data}
        assert "table" in table_types.values() or "view" in table_types.values()

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_column_type_mapping_sqlite(self, tmp_path):
        """Test that column types are correctly mapped for schema display."""
        # Create test database with various column types and defaults
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

        # Test schema introspection
        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        schema_info = await schema_manager.get_schema_info()

        # Check schema structure
        assert "main.test_table" in schema_info
        table_info = schema_info["main.test_table"]

        # Verify column info has both fields
        columns = table_info["columns"]
        for col_name, col_info in columns.items():
            assert "data_type" in col_info  # Original field
            assert "type" in col_info  # Mapped field for display

            # Verify mapping is correct
            assert col_info["type"] == col_info["data_type"]

        # Check specific columns
        assert columns["name"]["type"] == "TEXT"
        assert columns["id"]["type"] == "INTEGER"
        assert columns["active"]["type"] == "BOOLEAN"
        assert columns["score"]["type"] == "REAL"

        # Check defaults are preserved
        assert columns["email"]["default"] == "'unknown@example.com'"
        assert columns["age"]["default"] == "18"

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_schema_display_integration(self, tmp_path):
        """Test end-to-end schema display with field mappings."""
        # Create test database
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

        # Test complete flow from schema to display
        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)
        schema_info = await schema_manager.get_schema_info()

        # Test display manager
        string_io = StringIO()
        console = create_console(file=string_io, width=120, legacy_windows=False)
        display_manager = DisplayManager(console)

        # This should not raise an error and should populate type information
        display_manager.show_schema_info(schema_info)
        output = string_io.getvalue()

        # Verify types are displayed (not empty)
        assert "TEXT" in output  # name column type
        assert "INTEGER" in output  # id column type
        assert "BOOLEAN" in output  # active column type

        # Verify defaults are displayed
        assert "0.00" in output  # price default
        assert "TRUE" in output  # active default

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

        string_io = StringIO()
        console = create_console(file=string_io, width=120, legacy_windows=False)
        display_manager = DisplayManager(console)

        display_manager.show_schema_info(schema_info)
        output = string_io.getvalue()

        assert "Unique product identifier" in output
        assert "Product name" in output

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_table_list_display_integration(self, tmp_path):
        """Test end-to-end table list display with type mappings."""
        # Create test database with different object types
        db_path = tmp_path / "test_table_display.db"

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("CREATE TABLE users (id INTEGER, name TEXT)")
            await conn.execute(
                "CREATE VIEW active_users AS SELECT * FROM users WHERE id > 0"
            )
            await conn.commit()

        # Test complete flow from tables to display
        db_conn = SQLiteConnection(f"sqlite:///{db_path}")
        schema_manager = SchemaManager(db_conn)
        tables = await schema_manager.list_tables()

        # Test display manager
        string_io = StringIO()
        console = create_console(file=string_io, width=120, legacy_windows=False)
        display_manager = DisplayManager(console)

        # This should not raise an error and should populate type information
        display_manager.show_table_list(tables)
        output = string_io.getvalue()

        # Verify types are displayed in the Type column (not empty)
        lines = output.split("\n")
        type_lines = [
            line
            for line in lines
            if "|" in line and ("users" in line or "active_users" in line)
        ]

        # Should have at least one table and one view
        has_table_type = any("table" in line for line in type_lines)
        has_view_type = any("view" in line for line in type_lines)

        assert has_table_type or has_view_type  # At least one should have a type

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

        # Test schema introspection
        db_conn = DuckDBConnection(f"duckdb:///{db_path}")
        schema_manager = SchemaManager(db_conn)

        # Test table listing
        tables = await schema_manager.list_tables()
        table_names = {table["name"] for table in tables["tables"]}
        assert "products" in table_names
        assert "expensive_products" in table_names

        # Test schema info
        schema_info = await schema_manager.get_schema_info()
        products_info = schema_info["main.products"]

        # Check column types (DuckDB uses different type names)
        columns = products_info["columns"]
        assert columns["name"]["data_type"] == "VARCHAR"  # DuckDB specific
        assert columns["price"]["data_type"] == "DECIMAL(10,2)"  # DuckDB specific
        assert columns["active"]["data_type"] == "BOOLEAN"

        # Check defaults (DuckDB may format differently)
        assert columns["name"]["default"] is not None
        assert columns["price"]["default"] is not None

        # Check primary key
        assert "id" in products_info["primary_keys"]

        await db_conn.close()

    @pytest.mark.asyncio
    async def test_csv_schema_introspection(self, tmp_path):
        """Test CSV schema introspection (using DuckDB backend)."""
        # Create CSV file
        csv_path = tmp_path / "test_data.csv"
        csv_path.write_text(
            "id,name,age,active\n1,Alice,25,true\n2,Bob,30,false\n", encoding="utf-8"
        )

        # Test CSV connection schema introspection
        csv_conn = CSVConnection(f"csv:///{csv_path}")
        schema_manager = SchemaManager(csv_conn)

        # Test table listing
        tables = await schema_manager.list_tables()
        table_names = {table["name"] for table in tables["tables"]}
        assert "test_data" in table_names  # Should match CSV filename

        # Test schema info
        schema_info = await schema_manager.get_schema_info()
        assert "main.test_data" in schema_info

        # CSV columns should have inferred types
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
        db_path.touch()  # Create empty file

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

        # CSV should use DuckDB introspector
        assert isinstance(schema_manager.introspector, DuckDBSchemaIntrospector)
        await csv_conn.close()
