"""Tests for the CSVsConnection backed by DuckDB."""

from urllib.parse import urlencode

import pytest

from sqlsaber.database import CSVsConnection
from sqlsaber.database.schema import SchemaManager


@pytest.mark.asyncio
async def test_csvs_connection_exposes_multiple_views(tmp_path):
    users = tmp_path / "users.csv"
    orders = tmp_path / "orders.csv"

    users.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    orders.write_text("id,user_id,total\n10,1,9.99\n11,2,20.00\n", encoding="utf-8")

    specs = [f"csv:///{users}", f"csv:///{orders}"]
    conn = CSVsConnection(f"csvs:///?{urlencode({'spec': specs}, doseq=True)}")

    try:
        rows = await conn.execute_query(
            'SELECT u.name, o.total FROM "users" u JOIN "orders" o ON u.id = o.user_id ORDER BY o.id'
        )
        assert rows == [
            {"name": "Alice", "total": 9.99},
            {"name": "Bob", "total": 20.0},
        ]

        schema_manager = SchemaManager(conn)
        tables = await schema_manager.list_tables()
        names = {t["name"] for t in tables["tables"]}
        assert {"users", "orders"}.issubset(names)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_csvs_connection_dedupes_view_names(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    csv_a = dir_a / "data.csv"
    csv_b = dir_b / "data.csv"

    csv_a.write_text("x\n1\n", encoding="utf-8")
    csv_b.write_text("x\n2\n", encoding="utf-8")

    specs = [f"csv:///{csv_a}", f"csv:///{csv_b}"]
    conn = CSVsConnection(f"csvs:///?{urlencode({'spec': specs}, doseq=True)}")

    try:
        rows = await conn.execute_query(
            'SELECT x FROM "data" UNION ALL SELECT x FROM "data_2" ORDER BY x'
        )
        assert rows == [{"x": 1}, {"x": 2}]

        schema_manager = SchemaManager(conn)
        tables = await schema_manager.list_tables()
        names = {t["name"] for t in tables["tables"]}
        assert {"data", "data_2"}.issubset(names)
    finally:
        await conn.close()
