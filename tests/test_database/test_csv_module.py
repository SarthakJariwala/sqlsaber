"""Comprehensive tests for CSV database module."""

import pytest

from sqlsaber.database.csv import CSVConnection, CSVSchemaIntrospector


class TestCSVConnection:
    """Test CSV connection functionality."""

    def test_connection_string_parsing(self, tmp_path):
        """Test CSV connection string parsing."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("col1,col2\n1,2\n")

        # Basic CSV connection
        conn = CSVConnection(f"csv:///{csv_path}")
        assert conn.csv_path == str(csv_path)
        assert conn.table_name == "test"
        assert conn.delimiter == ","
        assert conn.encoding == "utf-8"
        assert conn.has_header

    def test_connection_string_with_parameters(self, tmp_path):
        """Test CSV connection string with query parameters."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("1|2\n3|4\n")

        conn = CSVConnection(
            f"csv:///{csv_path}?delimiter=|&header=false&encoding=utf-8"
        )
        assert conn.delimiter == "|"
        assert not conn.has_header
        assert conn.encoding == "utf-8"

    @pytest.mark.asyncio
    async def test_basic_csv_reading(self, tmp_path):
        """Test basic CSV reading functionality."""
        csv_path = tmp_path / "simple.csv"
        csv_path.write_text("name,age,city\nAlice,25,NYC\nBob,30,SF\n")

        conn = CSVConnection(f"csv:///{csv_path}")

        # Query the CSV data
        result = await conn.execute_query('SELECT * FROM "simple" ORDER BY name')

        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["age"] == 25
        assert result[0]["city"] == "NYC"
        assert result[1]["name"] == "Bob"

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_without_header(self, tmp_path):
        """Test CSV files without headers."""
        csv_path = tmp_path / "no_header.csv"
        csv_path.write_text("Alice,25,NYC\nBob,30,SF\n")

        conn = CSVConnection(f"csv:///{csv_path}?header=false")

        # Query with auto-generated column names
        result = await conn.execute_query('SELECT * FROM "no_header" ORDER BY column0')

        assert len(result) == 2
        assert result[0]["column0"] == "Alice"
        assert result[0]["column1"] == 25
        assert result[0]["column2"] == "NYC"

        await conn.close()

    @pytest.mark.asyncio
    async def test_custom_delimiter(self, tmp_path):
        """Test CSV with custom delimiter."""
        csv_path = tmp_path / "pipe_delimited.csv"
        csv_path.write_text("name|age|city\nAlice|25|NYC\nBob|30|SF\n")

        conn = CSVConnection(f"csv:///{csv_path}?delimiter=|")

        result = await conn.execute_query(
            'SELECT name, age FROM "pipe_delimited" WHERE age > 25'
        )

        assert len(result) == 1
        assert result[0]["name"] == "Bob"
        assert result[0]["age"] == 30

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_type_inference(self, tmp_path):
        """Test CSV automatic type inference."""
        csv_path = tmp_path / "types.csv"
        csv_path.write_text(
            "id,name,price,active,created\n1,Product A,19.99,true,2023-01-01\n2,Product B,29.99,false,2023-01-02\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")

        result = await conn.execute_query('SELECT * FROM "types" WHERE active = true')

        assert len(result) == 1
        assert result[0]["id"] == 1
        assert result[0]["name"] == "Product A"
        assert result[0]["price"] == 19.99
        assert result[0]["active"]

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_aggregation_queries(self, tmp_path):
        """Test aggregation queries on CSV data."""
        csv_path = tmp_path / "sales.csv"
        csv_path.write_text(
            "product,category,amount\nLaptop,Electronics,1200\nMouse,Electronics,25\nDesk,Furniture,300\nChair,Furniture,150\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")

        # Test aggregation
        result = await conn.execute_query("""
            SELECT category, COUNT(*) as count, SUM(amount) as total
            FROM "sales"
            GROUP BY category
            ORDER BY category
        """)

        assert len(result) == 2

        electronics = next(r for r in result if r["category"] == "Electronics")
        assert electronics["count"] == 2
        assert electronics["total"] == 1225

        furniture = next(r for r in result if r["category"] == "Furniture")
        assert furniture["count"] == 2
        assert furniture["total"] == 450

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_with_quotes_and_escapes(self, tmp_path):
        """Test CSV with quoted fields and escapes."""
        csv_path = tmp_path / "complex.csv"
        csv_path.write_text(
            'name,description,tags\n"John Doe","A ""great"" person","tag1,tag2"\n"Jane Smith","Simple person","tag3"\n'
        )

        conn = CSVConnection(f"csv:///{csv_path}")

        result = await conn.execute_query('SELECT * FROM "complex" ORDER BY name')

        assert len(result) == 2
        assert result[1]["name"] == "John Doe"  # After ordering
        assert "great" in result[1]["description"]
        assert "tag1,tag2" in result[1]["tags"]

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_encoding_handling(self, tmp_path):
        """Test CSV with different encodings."""
        csv_path = tmp_path / "encoded.csv"
        # Write file with UTF-8 content
        csv_path.write_text(
            "name,description\nCafé,Délicieux café\nNaïve,Très naïf\n", encoding="utf-8"
        )

        conn = CSVConnection(f"csv:///{csv_path}?encoding=utf-8")

        result = await conn.execute_query('SELECT * FROM "encoded" ORDER BY name')

        assert len(result) == 2
        assert "Café" in [r["name"] for r in result]
        assert "Naïve" in [r["name"] for r in result]

        await conn.close()

    @pytest.mark.asyncio
    async def test_empty_csv_file(self, tmp_path):
        """Test handling of empty CSV file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("col1,col2\n")  # Header only

        conn = CSVConnection(f"csv:///{csv_path}")

        result = await conn.execute_query('SELECT * FROM "empty"')
        assert len(result) == 0

        await conn.close()


class TestCSVSchemaIntrospector:
    """Test CSV schema introspection (uses DuckDB backend)."""

    @pytest.mark.asyncio
    async def test_csv_table_listing(self, tmp_path):
        """Test CSV table listing."""
        csv_path = tmp_path / "products.csv"
        csv_path.write_text("id,name,price\n1,Laptop,999.99\n2,Mouse,29.99\n")

        conn = CSVConnection(f"csv:///{csv_path}")
        introspector = CSVSchemaIntrospector()

        tables = await introspector.list_tables_info(conn)

        assert len(tables) == 1
        assert tables[0]["table_name"] == "products"
        assert tables[0]["table_schema"] == "main"

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_column_introspection(self, tmp_path):
        """Test CSV column type inference and introspection."""
        csv_path = tmp_path / "data_types.csv"
        csv_path.write_text(
            "id,name,price,active,created_date\n1,Product A,19.99,true,2023-01-01\n2,Product B,29.99,false,2023-01-02\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")
        introspector = CSVSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        columns = await introspector.get_columns_info(conn, tables)

        # Organize columns by name
        column_info = {col["column_name"]: col for col in columns}

        # DuckDB should infer appropriate types
        assert "id" in column_info
        assert "name" in column_info
        assert "price" in column_info
        assert "active" in column_info
        assert "created_date" in column_info

        # Check that some type inference occurred (exact types depend on DuckDB version)
        for col_name, col_data in column_info.items():
            assert col_data["data_type"] is not None
            assert len(col_data["data_type"]) > 0

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_with_null_values(self, tmp_path):
        """Test CSV handling of null/empty values."""
        csv_path = tmp_path / "with_nulls.csv"
        csv_path.write_text(
            "id,name,optional_field\n1,Alice,\n2,Bob,some_value\n3,,another_value\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")
        introspector = CSVSchemaIntrospector()

        # Test that we can introspect tables with null values
        tables = await introspector.get_tables_info(conn)
        assert len(tables) == 1

        columns = await introspector.get_columns_info(conn, tables)
        column_names = [col["column_name"] for col in columns]

        assert "id" in column_names
        assert "name" in column_names
        assert "optional_field" in column_names

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_complex_data_types(self, tmp_path):
        """Test CSV with complex data that might challenge type inference."""
        csv_path = tmp_path / "complex_types.csv"
        csv_path.write_text(
            "mixed_numbers,dates,booleans\n1,2023-01-01,true\n2.5,2023-12-31,false\n0,2023-06-15,1\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")
        introspector = CSVSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        columns = await introspector.get_columns_info(conn, tables)

        # Should handle mixed types gracefully
        assert len(columns) == 3

        column_names = [col["column_name"] for col in columns]
        assert "mixed_numbers" in column_names
        assert "dates" in column_names
        assert "booleans" in column_names

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_schema_manager_integration(self, tmp_path):
        """Test CSV with SchemaManager integration."""
        from sqlsaber.database.schema import SchemaManager

        csv_path = tmp_path / "integration_test.csv"
        csv_path.write_text(
            "id,product_name,category,price\n1,Laptop,Electronics,999.99\n2,Desk,Furniture,299.99\n"
        )

        conn = CSVConnection(f"csv:///{csv_path}")
        schema_manager = SchemaManager(conn)

        # Test list_tables
        tables = await schema_manager.list_tables()
        assert len(tables["tables"]) == 1
        assert tables["tables"][0]["name"] == "integration_test"

        # Test get_schema_info
        schema_info = await schema_manager.get_schema_info()
        assert "main.integration_test" in schema_info

        table_info = schema_info["main.integration_test"]
        assert "columns" in table_info
        assert "id" in table_info["columns"]
        assert "product_name" in table_info["columns"]
        assert "price" in table_info["columns"]

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_introspection_edge_cases(self, tmp_path):
        """Test CSV introspection edge cases."""
        # Test single column CSV
        single_col_path = tmp_path / "single_col.csv"
        single_col_path.write_text("value\n1\n2\n3\n")

        conn = CSVConnection(f"csv:///{single_col_path}")
        introspector = CSVSchemaIntrospector()

        tables = await introspector.get_tables_info(conn)
        columns = await introspector.get_columns_info(conn, tables)

        assert len(columns) == 1
        assert columns[0]["column_name"] == "value"

        await conn.close()

    @pytest.mark.asyncio
    async def test_csv_file_path_edge_cases(self, tmp_path):
        """Test CSV connection with various file path formats."""
        # Test CSV with spaces in name
        csv_path = tmp_path / "file with spaces.csv"
        csv_path.write_text("col1,col2\nval1,val2\n")

        conn = CSVConnection(f"csv:///{csv_path}")

        # Should handle file names with spaces
        assert conn.table_name == "file with spaces"

        # Test basic query works
        result = await conn.execute_query('SELECT * FROM "file with spaces"')
        assert len(result) == 1
        assert result[0]["col1"] == "val1"

        await conn.close()
