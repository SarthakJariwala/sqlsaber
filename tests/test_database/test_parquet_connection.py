"""Parquet inputs use the CSV DuckDB lifecycle, including its lockdown."""

from datetime import date
from decimal import Decimal

import duckdb
import pytest

from sqlsaber.database import DatabaseConnection, QueryTimeoutError, SchemaManager
from sqlsaber.database.resolver import (
    DatabaseResolutionError,
    resolve_database,
    resolve_databases,
)


@pytest.fixture
def parquet_path(tmp_path):
    path = tmp_path / "orders.parquet"
    with duckdb.connect() as db:
        db.execute(
            "COPY (SELECT 1 AS id, 12.50::DECIMAL(10,2) AS total, "
            "DATE '2026-01-02' AS placed, NULL::VARCHAR AS note) "
            "TO ? (FORMAT PARQUET)",
            [str(path)],
        )
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("as_url", [False, True])
async def test_parquet_types_and_schema(parquet_path, as_url):
    spec = f"parquet:///{parquet_path}" if as_url else str(parquet_path)
    resolved = resolve_database(spec)
    assert resolved.name == "orders"
    conn = DatabaseConnection(resolved.connection_string)
    assert conn.sqlglot_dialect == "duckdb"
    try:
        assert await conn.execute_query("SELECT * FROM orders", read_only=True) == [
            {
                "id": 1,
                "total": Decimal("12.50"),
                "placed": date(2026, 1, 2),
                "note": None,
            }
        ]
        schema = await SchemaManager(conn).get_schema_info()
        assert schema["main.orders"]["columns"]["total"]["data_type"] == "DECIMAL(10,2)"
        assert schema["main.orders"]["columns"]["placed"]["data_type"] == "DATE"
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("other_format", ["csv", "parquet"])
async def test_join_files(parquet_path, other_format):
    other = parquet_path.with_name(f"customers.{other_format}")
    with duckdb.connect() as db:
        db.execute(
            f"COPY (SELECT 1 AS id, 'Alice' AS name) TO ? (FORMAT {other_format})",
            [str(other)],
        )
    resolved = resolve_databases([str(parquet_path), f"{other_format}:///{other}"])
    assert len(resolved) == 1
    conn = DatabaseConnection(resolved[0].connection_string)
    try:
        assert await conn.execute_query(
            "SELECT name, total FROM customers JOIN orders USING (id)", read_only=True
        ) == [{"name": "Alice", "total": Decimal("12.50")}]
        tables = await SchemaManager(conn).list_tables()
        assert {t["name"] for t in tables["tables"]} == {"customers", "orders"}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_same_stem_across_formats(parquet_path):
    csv = parquet_path.with_suffix(".csv")
    csv.write_text("id\n2\n")
    conn = DatabaseConnection(
        resolve_database([str(parquet_path), str(csv)]).connection_string
    )
    assert await conn.execute_query(
        "SELECT id FROM orders UNION ALL SELECT id FROM orders_2 ORDER BY id",
        read_only=True,
    ) == [{"id": 1}, {"id": 2}]


def test_parquet_paths_and_missing_files(parquet_path, monkeypatch):
    monkeypatch.chdir(parquet_path.parent)
    uppercase = parquet_path.rename(parquet_path.with_suffix(".PARQUET"))
    assert resolve_database(uppercase.name).name == "orders"
    for spec in ["missing.parquet", [uppercase.name, "missing.parquet"]]:
        with pytest.raises(DatabaseResolutionError, match="PARQUET file .* not found"):
            resolve_database(spec)
    assert len(resolve_databases([str(uppercase), "sqlite:///other.db"])) == 2


@pytest.mark.asyncio
async def test_quoted_file_and_table_name(parquet_path, monkeypatch):
    path = parquet_path.rename(parquet_path.with_name("order's data.parquet"))
    conn = DatabaseConnection(resolve_database(str(path)).connection_string)
    assert await conn.execute_query(
        'SELECT id FROM "order\'s data"', read_only=True
    ) == [{"id": 1}]

    # Double quotes are valid SQL identifiers, but not Windows filenames.
    monkeypatch.setattr(conn, "table_name", 'order\'s "data"')
    assert await conn.execute_query(
        'SELECT id FROM "order\'s ""data"""', read_only=True
    ) == [{"id": 1}]


@pytest.mark.asyncio
async def test_invalid_parquet(tmp_path):
    path = tmp_path / "bad.parquet"
    path.write_text("not a parquet file")
    conn = DatabaseConnection(resolve_database(str(path)).connection_string)
    with pytest.raises(duckdb.InvalidInputException):
        await conn.execute_query("SELECT * FROM bad", read_only=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["read_parquet", "read_blob", "replacement"])
@pytest.mark.parametrize("selected", [True, False])
async def test_lockdown_blocks_file_reads(parquet_path, reader, selected):
    target = parquet_path
    if not selected:
        target = parquet_path.with_name("sibling.parquet")
        target.write_bytes(parquet_path.read_bytes())
    relation = f"'{target}'" if reader == "replacement" else f"{reader}('{target}')"
    conn = DatabaseConnection(resolve_database(str(parquet_path)).connection_string)
    with pytest.raises(duckdb.PermissionException, match="disabled|permission"):
        await conn.execute_query(f"SELECT * FROM {relation}", read_only=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("overwrite", [True, False])
async def test_lockdown_blocks_file_writes(parquet_path, overwrite):
    before = parquet_path.read_bytes()
    target = parquet_path if overwrite else parquet_path.with_name("output.parquet")
    conn = DatabaseConnection(resolve_database(str(parquet_path)).connection_string)
    with pytest.raises(duckdb.PermissionException, match="disabled|permission"):
        await conn.execute_query(
            f"COPY orders TO '{target}' (FORMAT PARQUET)", read_only=True
        )
    assert parquet_path.read_bytes() == before
    if not overwrite:
        assert not target.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("during_load", [False, True])
async def test_parquet_timeout(parquet_path, monkeypatch, during_load):
    conn = DatabaseConnection(resolve_database(str(parquet_path)).connection_string)
    query = "SELECT sum(a.i * b.i) FROM range(10000000) a(i), range(10000000) b(i)"
    if during_load:
        monkeypatch.setattr(conn, "_create_table", lambda db: db.execute(query))
    with pytest.raises(QueryTimeoutError):
        await conn.execute_query(query, timeout=0.2, read_only=True)
