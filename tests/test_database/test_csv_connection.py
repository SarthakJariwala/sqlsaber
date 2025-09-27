"""Tests for the CSVConnection backed by DuckDB."""

import pytest

from sqlsaber.database import CSVConnection
from sqlsaber.database.schema import SchemaManager


@pytest.mark.asyncio
async def test_csv_connection_reads_csv(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    conn = CSVConnection(f"csv:///{csv_path}")
    try:
        table_name = csv_path.stem
        rows = await conn.execute_query(f'SELECT * FROM "{table_name}" ORDER BY a')
        assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

        schema_manager = SchemaManager(conn)
        tables = await schema_manager.list_tables()
        assert any(table["name"] == table_name for table in tables["tables"])
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_csv_connection_without_header(tmp_path):
    csv_path = tmp_path / "no_header.csv"
    csv_path.write_text("1,2\n3,4\n", encoding="utf-8")

    conn = CSVConnection(f"csv:///{csv_path}?header=false")
    try:
        table_name = csv_path.stem
        rows = await conn.execute_query(
            f'SELECT column0, column1 FROM "{table_name}" ORDER BY column0'
        )
        assert rows == [{"column0": 1, "column1": 2}, {"column0": 3, "column1": 4}]
    finally:
        await conn.close()
