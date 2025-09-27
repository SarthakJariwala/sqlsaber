"""Comprehensive tests for SQLite database module."""

import sqlite3

import pytest

from sqlsaber.database.sqlite import SQLiteConnection, SQLiteSchemaIntrospector


class TestSQLiteConnection:
    """Test SQLite connection functionality."""

    def test_connection_string_parsing(self):
        """Test SQLite connection string parsing."""
        conn = SQLiteConnection("sqlite:///path/to/database.db")
        assert conn.database_path == "path/to/database.db"
        assert conn.connection_string == "sqlite:///path/to/database.db"

    @pytest.mark.asyncio
    async def test_memory_database_connection(self):
        """Test connection to in-memory SQLite database."""
        conn = SQLiteConnection("sqlite:///:memory:")

        # Test basic query execution
        result = await conn.execute_query("SELECT 1 as test")
        assert len(result) == 1
        assert result[0]["test"] == 1

        await conn.close()

    @pytest.mark.asyncio
    async def test_query_error_handling(self):
        """Test SQLite query error handling."""
        conn = SQLiteConnection("sqlite:///:memory:")

        # Test invalid SQL
        with pytest.raises(Exception):  # Should raise some SQL-related exception
            await conn.execute_query("INVALID SQL STATEMENT")

        # Test reference to non-existent table
        with pytest.raises(Exception):
            await conn.execute_query("SELECT * FROM non_existent_table")

        await conn.close()


class TestSQLiteSchemaIntrospector:
    """Test SQLite schema introspection functionality."""

    @pytest.mark.asyncio
    async def test_basic_table_listing(self, tmp_path):
        """Test basic table listing functionality."""
        db_path = tmp_path / "introspect.db"

        with sqlite3.connect(db_path) as db_conn:
            db_conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            db_conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER)")
            db_conn.execute(
                "CREATE VIEW user_orders AS SELECT * FROM users JOIN orders ON users.id = orders.user_id"
            )
            db_conn.commit()

        # Test introspection
        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        # Test list_tables_info
        tables = await introspector.list_tables_info(conn)

        table_names = {table["table_name"] for table in tables}
        table_types = {table["table_name"]: table["table_type"] for table in tables}

        assert "users" in table_names
        assert "orders" in table_names
        assert "user_orders" in table_names

        # Check table types
        assert table_types["users"] == "table"
        assert table_types["orders"] == "table"
        assert table_types["user_orders"] == "view"

        await conn.close()

    @pytest.mark.asyncio
    async def test_column_introspection(self, tmp_path):
        """Test column introspection with various data types."""
        db_path = tmp_path / "columns.db"

        with sqlite3.connect(db_path) as db_conn:
            db_conn.execute("""
                CREATE TABLE data_types (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    age INTEGER DEFAULT 0,
                    salary REAL DEFAULT 0.0,
                    active BOOLEAN DEFAULT TRUE,
                    data BLOB,
                    created_date DATE,
                    created_time DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            db_conn.commit()

        # Test column introspection
        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        # Get table info first
        tables = await introspector.get_tables_info(conn)

        # Get columns info
        columns = await introspector.get_columns_info(conn, tables)

        # Organize columns by name for easier testing
        column_info = {
            col["column_name"]: col
            for col in columns
            if col["table_name"] == "data_types"
        }

        # Check column types
        assert column_info["id"]["data_type"] == "INTEGER"
        assert column_info["name"]["data_type"] == "TEXT"
        assert column_info["age"]["data_type"] == "INTEGER"
        assert column_info["salary"]["data_type"] == "REAL"
        assert column_info["active"]["data_type"] == "BOOLEAN"
        assert column_info["data"]["data_type"] == "BLOB"

        # Check nullability (SQLite returns "NO"/"YES" strings)
        assert column_info["name"]["is_nullable"] == "NO"  # NOT NULL
        assert column_info["age"]["is_nullable"] == "YES"  # Nullable by default

        # Check defaults
        assert column_info["age"]["column_default"] == "0"
        assert column_info["salary"]["column_default"] == "0.0"
        assert column_info["active"]["column_default"] == "TRUE"
        assert column_info["created_time"]["column_default"] == "CURRENT_TIMESTAMP"

        await conn.close()

    @pytest.mark.asyncio
    async def test_primary_key_detection(self, tmp_path):
        """Test primary key detection."""
        db_path = tmp_path / "pk_test.db"

        with sqlite3.connect(db_path) as db_conn:
            db_conn.execute(
                "CREATE TABLE single_pk (id INTEGER PRIMARY KEY, name TEXT)"
            )
            db_conn.execute(
                "CREATE TABLE composite_pk (user_id INTEGER, role_id INTEGER, PRIMARY KEY (user_id, role_id))"
            )
            db_conn.commit()

        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        primary_keys = await introspector.get_primary_keys_info(conn, tables)

        # Organize by table
        pk_by_table = {}
        for pk in primary_keys:
            table_name = pk["table_name"]
            if table_name not in pk_by_table:
                pk_by_table[table_name] = []
            pk_by_table[table_name].append(pk["column_name"])

        # Check primary keys
        assert "id" in pk_by_table["single_pk"]
        assert len(pk_by_table["single_pk"]) == 1

        assert "user_id" in pk_by_table["composite_pk"]
        assert "role_id" in pk_by_table["composite_pk"]
        assert len(pk_by_table["composite_pk"]) == 2

        await conn.close()

    @pytest.mark.asyncio
    async def test_foreign_key_detection(self, tmp_path):
        """Test foreign key detection."""
        db_path = tmp_path / "fk_test.db"

        with sqlite3.connect(db_path) as db_conn:
            # Enable foreign keys for this connection
            db_conn.execute("PRAGMA foreign_keys = ON")

            db_conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
            db_conn.execute("""
                CREATE TABLE orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            db_conn.commit()

        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        foreign_keys = await introspector.get_foreign_keys_info(conn, tables)

        # Check foreign key relationships
        fk_found = False
        for fk in foreign_keys:
            if fk["table_name"] == "orders" and fk["column_name"] == "user_id":
                assert fk["foreign_table_name"] == "users"
                assert fk["foreign_column_name"] == "id"
                fk_found = True

        assert fk_found, "Foreign key relationship not detected"

        await conn.close()

    @pytest.mark.asyncio
    async def test_index_detection(self, tmp_path):
        """Test index detection."""
        db_path = tmp_path / "index_test.db"

        with sqlite3.connect(db_path) as db_conn:
            db_conn.execute(
                "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT)"
            )
            db_conn.execute("CREATE INDEX idx_product_name ON products(name)")
            db_conn.execute(
                "CREATE UNIQUE INDEX idx_product_category ON products(category)"
            )
            db_conn.execute(
                "CREATE INDEX idx_product_composite ON products(name, category)"
            )
            db_conn.commit()

        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        indexes = await introspector.get_indexes_info(conn, tables)

        # Check indexes exist (excluding auto-generated primary key indexes)
        index_names = [
            idx["index_name"]
            for idx in indexes
            if not idx["index_name"].startswith("sqlite_autoindex_")
        ]

        assert "idx_product_name" in index_names
        assert "idx_product_category" in index_names
        assert "idx_product_composite" in index_names

        # Check index properties
        for idx in indexes:
            if idx["index_name"] == "idx_product_name":
                assert not idx["is_unique"]
                assert "name" in idx["column_names"]
            elif idx["index_name"] == "idx_product_category":
                assert idx["is_unique"]
                assert "category" in idx["column_names"]
            elif idx["index_name"] == "idx_product_composite":
                assert not idx["is_unique"]
                assert "name" in idx["column_names"]
                assert "category" in idx["column_names"]

        await conn.close()

    @pytest.mark.asyncio
    async def test_table_pattern_filtering(self, tmp_path):
        """Test table pattern filtering functionality."""
        db_path = tmp_path / "pattern_test.db"

        with sqlite3.connect(db_path) as db_conn:
            db_conn.execute("CREATE TABLE user_profiles (id INTEGER)")
            db_conn.execute("CREATE TABLE user_settings (id INTEGER)")
            db_conn.execute("CREATE TABLE order_items (id INTEGER)")
            db_conn.execute("CREATE TABLE products (id INTEGER)")
            db_conn.commit()

        conn = SQLiteConnection(f"sqlite:///{db_path}")
        introspector = SQLiteSchemaIntrospector()

        # Test pattern matching
        user_tables = await introspector.get_tables_info(conn, "user_%")
        user_table_names = {table["table_name"] for table in user_tables}

        assert "user_profiles" in user_table_names
        assert "user_settings" in user_table_names
        assert "order_items" not in user_table_names
        assert "products" not in user_table_names

        await conn.close()

    @pytest.mark.asyncio
    async def test_empty_database_introspection(self):
        """Test introspection of empty database."""
        conn = SQLiteConnection("sqlite:///:memory:")
        introspector = SQLiteSchemaIntrospector()

        tables = await introspector.list_tables_info(conn)
        assert len(tables) == 0

        schema_tables = await introspector.get_tables_info(conn)
        assert len(schema_tables) == 0

        columns = await introspector.get_columns_info(conn, [])
        assert len(columns) == 0

        await conn.close()
