"""Engine-level read-only hardening tests.

These exercise the database connections directly (bypassing the AST guard) to
verify defense-in-depth: arbitrary file reads must fail at the DuckDB engine in
read-only mode, and runaway queries must be cancelled at the engine when the
timeout elapses.
"""

import time

import pytest

from sqlsaber.database.base import QueryTimeoutError
from sqlsaber.database.csv import CSVConnection
from sqlsaber.database.csvs import CSVsConnection
from sqlsaber.database.duckdb import DuckDBConnection
from sqlsaber.database.sqlite import SQLiteConnection

RECURSIVE_BOMB = (
    "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) "
    "SELECT count(*) FROM t"
)


def _write_valid_csv(path) -> None:
    """Write a small, valid CSV that DuckDB can read via a replacement scan."""
    path.write_text("x\n1\n2\n", encoding="utf-8")


class TestDuckDBFileReadLockdown:
    @pytest.mark.asyncio
    async def test_memory_read_only_blocks_readable_file(self, tmp_path):
        evil = tmp_path / "evil.csv"
        _write_valid_csv(evil)
        conn = DuckDBConnection(":memory:")
        try:
            # Without lockdown the file is genuinely readable...
            rows = await conn.execute_query(f"SELECT * FROM '{evil}'", read_only=False)
            assert rows == [{"x": 1}, {"x": 2}]
            # ...but read-only mode must block the read at the engine.
            with pytest.raises(Exception):
                await conn.execute_query(f"SELECT * FROM '{evil}'", read_only=True)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_memory_read_only_allows_normal_query(self):
        conn = DuckDBConnection(":memory:")
        try:
            rows = await conn.execute_query("SELECT 42 AS answer", read_only=True)
            assert rows == [{"answer": 42}]
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_file_read_only_blocks_readable_file(self, tmp_path):
        evil = tmp_path / "evil.csv"
        _write_valid_csv(evil)
        db_path = tmp_path / "t.duckdb"
        conn = DuckDBConnection(str(db_path))
        # Materialize a table so the file DB exists for the read-only reopen.
        await conn.execute_query("CREATE TABLE t AS SELECT 1 AS x")
        try:
            with pytest.raises(Exception):
                await conn.execute_query(f"SELECT * FROM '{evil}'", read_only=True)
        finally:
            await conn.close()


class TestCSVFileReadLockdown:
    @pytest.mark.asyncio
    async def test_csv_legit_query_still_works(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        conn = CSVConnection(f"csv:///{csv_path}")
        try:
            rows = await conn.execute_query(
                'SELECT * FROM "data" ORDER BY a', read_only=True
            )
            assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_csv_read_only_blocks_other_readable_file(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
        evil = tmp_path / "evil.csv"
        _write_valid_csv(evil)
        conn = CSVConnection(f"csv:///{csv_path}")
        try:
            with pytest.raises(Exception):
                await conn.execute_query(f"SELECT * FROM '{evil}'", read_only=True)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_csvs_legit_query_still_works(self, tmp_path):
        from urllib.parse import urlencode

        users = tmp_path / "users.csv"
        orders = tmp_path / "orders.csv"
        users.write_text("id,name\n1,Alice\n", encoding="utf-8")
        orders.write_text("id,user_id,total\n10,1,9.99\n", encoding="utf-8")
        specs = [f"csv:///{users}", f"csv:///{orders}"]
        conn = CSVsConnection(f"csvs:///?{urlencode({'spec': specs}, doseq=True)}")
        try:
            rows = await conn.execute_query(
                'SELECT u.name, o.total FROM "users" u '
                'JOIN "orders" o ON u.id = o.user_id',
                read_only=True,
            )
            assert rows == [{"name": "Alice", "total": 9.99}]
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_csvs_read_only_blocks_other_readable_file(self, tmp_path):
        from urllib.parse import urlencode

        users = tmp_path / "users.csv"
        users.write_text("id,name\n1,Alice\n", encoding="utf-8")
        evil = tmp_path / "evil.csv"
        _write_valid_csv(evil)
        specs = [f"csv:///{users}"]
        conn = CSVsConnection(f"csvs:///?{urlencode({'spec': specs}, doseq=True)}")
        try:
            with pytest.raises(Exception):
                await conn.execute_query(f"SELECT * FROM '{evil}'", read_only=True)
        finally:
            await conn.close()


class TestQueryCancellation:
    @pytest.mark.asyncio
    async def test_csv_materialization_times_out_promptly(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        conn = CSVConnection(f"csv:///{csv_path}")

        def slow_materialization(duck_conn) -> None:
            duck_conn.execute(RECURSIVE_BOMB)

        monkeypatch.setattr(conn, "_create_table", slow_materialization)
        started = time.monotonic()
        try:
            with pytest.raises(QueryTimeoutError):
                await conn.execute_query("SELECT 1", timeout=0.2)
        finally:
            await conn.close()

        assert time.monotonic() - started < 2

    @pytest.mark.asyncio
    async def test_csvs_materialization_times_out_promptly(self, tmp_path, monkeypatch):
        from urllib.parse import urlencode

        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a\n1\n", encoding="utf-8")
        specs = [f"csv:///{csv_path}"]
        conn = CSVsConnection(f"csvs:///?{urlencode({'spec': specs}, doseq=True)}")

        def slow_materialization(duck_conn) -> None:
            duck_conn.execute(RECURSIVE_BOMB)

        monkeypatch.setattr(conn.csv_sources[0], "_create_table", slow_materialization)
        started = time.monotonic()
        try:
            with pytest.raises(QueryTimeoutError):
                await conn.execute_query("SELECT 1", timeout=0.2)
        finally:
            await conn.close()

        assert time.monotonic() - started < 2

    @pytest.mark.asyncio
    async def test_duckdb_recursive_bomb_times_out(self):
        conn = DuckDBConnection(":memory:")
        try:
            with pytest.raises(QueryTimeoutError):
                await conn.execute_query(RECURSIVE_BOMB, timeout=0.5)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_sqlite_recursive_bomb_times_out(self):
        conn = SQLiteConnection("sqlite:///:memory:")
        try:
            with pytest.raises(QueryTimeoutError):
                await conn.execute_query(RECURSIVE_BOMB, timeout=0.5)
        finally:
            await conn.close()
