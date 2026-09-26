"""MCP contract tests using real databases and protocol clients."""

import asyncio
import datetime
from decimal import Decimal
from pathlib import Path
import sqlite3
import sys
from unittest.mock import AsyncMock

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from fastmcp.exceptions import ToolError
from fastmcp.utilities.tests import run_server_async
import httpx
import pytest

from sqlsaber.database.registry import DatabaseRegistry
from sqlsaber.mcp import _response, create_server


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "analytics.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, amount INTEGER)")
        conn.executemany(
            "INSERT INTO items VALUES (?, ?)", [(i, i * 3) for i in range(1, 1006)]
        )
    return path


async def test_discovery_and_structured_results(database):
    async with Client(create_server(str(database))) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert set(tools) == {
            "list_dbs",
            "list_tables",
            "introspect_schema",
            "execute_sql",
        }
        for tool in tools.values():
            assert "ctx" not in tool.input_schema["properties"]
            assert tool.annotations.read_only_hint is True
            assert tool.output_schema["type"] == "object"
        catalog = (await client.call_tool("list_dbs")).data
        assert catalog["databases"][0]["name"] == "analytics"
        tables = (await client.call_tool("list_tables")).data
        assert [table["name"] for table in tables["tables"]] == ["items"]
        schema = (
            await client.call_tool("introspect_schema", {"table_pattern": "%items"})
        ).data
        assert set(schema["tables"]["main.items"]["columns"]) == {"id", "amount"}
        result = await client.call_tool(
            "execute_sql",
            {"query": "SELECT SUM(amount) AS total FROM items WHERE id <= 3"},
        )
        assert result.data == {
            "db_name": "analytics",
            "results": [{"total": 18}],
            "row_count": 1,
            "row_limit": 1000,
            "truncated": False,
        }


@pytest.mark.parametrize(
    "query,count,first,last,truncated",
    [
        ("SELECT id FROM items ORDER BY id DESC", 1000, 1005, 6, True),
        ("SELECT id FROM items ORDER BY id LIMIT 1000000", 1000, 1, 1000, True),
        (
            "WITH one AS (SELECT id FROM items LIMIT 1) SELECT items.id FROM items CROSS JOIN one ORDER BY items.id",
            1000,
            1,
            1000,
            True,
        ),
        ("SELECT id FROM items ORDER BY id LIMIT 2 OFFSET 3", 2, 4, 5, False),
        ("SELECT id FROM items ORDER BY id LIMIT 1000", 1000, 1, 1000, False),
        (
            "SELECT id FROM items WHERE id < 3 UNION ALL SELECT id FROM items WHERE id > 1003 ORDER BY id",
            4,
            1,
            1005,
            False,
        ),
    ],
)
async def test_outer_row_bound(database, query, count, first, last, truncated):
    async with Client(create_server(str(database))) as client:
        result = (await client.call_tool("execute_sql", {"query": query})).data
        assert result["row_count"] == count
        assert result["results"][0] == {"id": first}
        assert result["results"][-1] == {"id": last}
        assert result["truncated"] is truncated


async def test_limit_and_read_only_reach_driver_before_fetch(database, monkeypatch):
    import sqlglot
    from sqlsaber.database.sqlite import SQLiteConnection

    original = SQLiteConnection.execute_query
    calls = []

    async def execute(self, query, *args, **kwargs):
        calls.append(query)
        assert sqlglot.parse_one(query).args["limit"].expression.this == "1001"
        assert kwargs == {"commit": False, "read_only": True}
        return await original(self, query, *args, **kwargs)

    monkeypatch.setattr(SQLiteConnection, "execute_query", execute)
    async with Client(create_server(str(database))) as client:
        await client.call_tool(
            "execute_sql", {"query": "SELECT id FROM items LIMIT 900000"}
        )
        denied = await client.call_tool(
            "execute_sql", {"query": "DELETE FROM items"}, raise_on_error=False
        )
        assert denied.is_error
    assert len(calls) == 1


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM items",
        "DROP TABLE items",
        "SELECT 1; DELETE FROM items",
        "SELECT load_extension('/secret')",
        "",
        "SELECT FROM 'private-secret'",
    ],
)
async def test_rejected_queries_are_protocol_errors(database, query):
    async with Client(create_server(str(database))) as client:
        result = await client.call_tool(
            "execute_sql", {"query": query}, raise_on_error=False
        )
        assert result.is_error
        assert "private-secret" not in str(result.content)
        remaining = (
            await client.call_tool(
                "execute_sql", {"query": "SELECT COUNT(*) AS n FROM items"}
            )
        ).data
        assert remaining["results"] == [{"n": 1005}]


async def test_database_routing_and_error_masking(database, tmp_path):
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE items (id INTEGER)")
        conn.execute("INSERT INTO items VALUES (99)")
    async with Client(create_server([str(database), str(other)])) as client:
        missing = await client.call_tool("list_tables", raise_on_error=False)
        assert missing.is_error
        result = (
            await client.call_tool(
                "execute_sql", {"query": "SELECT id FROM items", "db_name": "other"}
            )
        ).data
        assert result["results"] == [{"id": 99}]
    async with Client(create_server(str(database))) as client:
        unknown = await client.call_tool(
            "list_tables", {"db_name": "other"}, raise_on_error=False
        )
        assert unknown.is_error
        failed = await client.call_tool(
            "execute_sql",
            {"query": "SELECT * FROM secret_table_name"},
            raise_on_error=False,
        )
        assert failed.is_error
        assert "secret_table_name" not in str(failed.content)


def test_result_serialization_and_byte_budget():
    assert _response(
        {
            "values": [
                Decimal("123456789.123456789"),
                datetime.date(2026, 9, 23),
                b"\x00\xff",
            ]
        }
    ) == {"values": ["123456789.123456789", "2026-09-23", "AP8="]}
    with pytest.raises(ToolError, match="unsupported"):
        _response({"value": float("inf")})
    with pytest.raises(ToolError, match="exceeds"):
        _response({"value": "é" * 500_000})


async def test_lifespan_closes_registry(database, monkeypatch):
    original = DatabaseRegistry.close
    closed = []

    async def close(self):
        closed.append(self.names())
        await original(self)

    monkeypatch.setattr(DatabaseRegistry, "close", close)
    async with Client(create_server(str(database))) as client:
        await client.call_tool("list_dbs")
    assert closed == [["analytics"]]


async def test_partial_startup_closes_registry(database, monkeypatch):
    from sqlsaber.database.sqlite import SQLiteConnection

    close = AsyncMock()
    monkeypatch.setattr(DatabaseRegistry, "close", close)
    monkeypatch.setattr(
        SQLiteConnection,
        "get_pool",
        AsyncMock(side_effect=RuntimeError("startup failure")),
    )
    with pytest.raises(Exception):
        async with Client(create_server(str(database))):
            pass
    close.assert_awaited_once()


async def test_http_transport(database):
    async with run_server_async(create_server(str(database))) as url:
        async with Client(url) as client:
            responses = await asyncio.gather(
                *[
                    client.call_tool("execute_sql", {"query": f"SELECT {value} AS n"})
                    for value in [7, 19, 31]
                ]
            )
            assert [response.data["results"] for response in responses] == [
                [{"n": 7}],
                [{"n": 19}],
                [{"n": 31}],
            ]


async def test_http_rejects_untrusted_host_and_origin(database):
    app = create_server(str(database)).http_app(host_origin_protection=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000"
    ) as client:
        for headers in [{"host": "evil.example"}, {"origin": "https://evil.example"}]:
            response = await client.post("/mcp", headers=headers, json={})
            assert response.status_code in {400, 403, 421}


async def test_stdio_cli_transport(database, tmp_path):
    # Explicit isolated config; no API keys forwarded to the server subprocess.
    env = {
        key: str(tmp_path / key.lower())
        for key in [
            "HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "XDG_CACHE_HOME",
        ]
    }
    env["KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "sqlsaber", "mcp", "-d", str(database)],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        log_file=tmp_path / "server-stderr.log",
    )
    async with Client(transport) as client:
        assert len(await client.list_tools()) == 4
        result = await client.call_tool(
            "execute_sql", {"query": "SELECT amount FROM items WHERE id = 7"}
        )
        assert result.data["results"] == [{"amount": 21}]


async def test_csv_and_duckdb_queries(tmp_path):
    import duckdb

    csv = tmp_path / "orders.csv"
    csv.write_text("id,amount\n1,7\n2,13\n")
    db = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE items AS SELECT 23 AS amount")
    for path, table, expected in [(csv, "orders", 20), (db, "items", 23)]:
        async with Client(create_server(str(path))) as client:
            # Mix introspection and query calls to exercise driver connection modes.
            results = await asyncio.gather(
                client.call_tool("introspect_schema"),
                client.call_tool(
                    "execute_sql",
                    {"query": f"SELECT SUM(amount) AS total FROM {table}"},
                ),
            )
            assert results[0].data["tables"]
            assert results[1].data["results"] == [{"total": expected}]


async def test_cancellation_drains_operation_before_next_call(database, monkeypatch):
    import anyio
    from sqlsaber.database.sqlite import SQLiteConnection

    original = SQLiteConnection.execute_query
    started = asyncio.Event()
    released = asyncio.Event()
    finished = asyncio.Event()

    async def execute(self, query, *args, **kwargs):
        if "777" in query:
            started.set()
            await released.wait()
            result = await original(self, query, *args, **kwargs)
            finished.set()
            return result
        assert finished.is_set(), "A cancelled operation released its slot too early"
        return await original(self, query, *args, **kwargs)

    monkeypatch.setattr(SQLiteConnection, "execute_query", execute)
    async with Client(create_server(str(database))) as client:
        first = asyncio.create_task(
            client.call_tool("execute_sql", {"query": "SELECT 777 AS n"})
        )
        await asyncio.wait_for(started.wait(), 5)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(
            client.call_tool("execute_sql", {"query": "SELECT 19 AS n"})
        )
        try:
            await anyio.sleep(0.05)
            assert not second.done()
        finally:
            released.set()
        result = await asyncio.wait_for(second, 5)
        assert result.data["results"] == [{"n": 19}]
