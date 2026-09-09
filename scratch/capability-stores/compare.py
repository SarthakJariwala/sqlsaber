"""Throwaway comparison of capability-mode store wiring.

This is not production code. Run:

    uv run python scratch/capability-stores/compare.py
    uv run python scratch/capability-stores/compare.py a
    uv run python scratch/capability-stores/compare.py b
    uv run python scratch/capability-stores/compare.py c
    uv run python scratch/capability-stores/compare.py split
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from sqlsaber.artifacts import InMemoryArtifactStore
from sqlsaber.capabilities.base import SqlSaberCapability
from sqlsaber.capabilities.plugins import PluginContext, discover_capabilities
from sqlsaber.capabilities.sql_tools import SqlTools
from sqlsaber.config.settings import Config
from sqlsaber.knowledge.manager import KnowledgeManager
from sqlsaber.query_result_resolution import query_result_references_from_messages
from sqlsaber.query_results import (
    InMemoryQueryResultStore,
    QueryResultContext,
    QueryResultStore,
    QueryResultUnavailable,
    StoredQueryResult,
)

ROW_COUNT = 250
SEED_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    revenue INTEGER NOT NULL
);
"""


def seed_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SEED_SQL)
        connection.executemany(
            "INSERT INTO customers (name, revenue) VALUES (?, ?)",
            [
                (f"customer-{index:03d}-" + ("x" * 24), 1000 + index)
                for index in range(ROW_COUNT)
            ],
        )
        connection.commit()
    finally:
        connection.close()


class PeekResults(SqlSaberCapability):
    """Stand-in for notebook/sandbox/viz: resolve complete SQL rows from history."""

    id = "peek-results"
    description = "Load complete stashed SQL results from the current conversation."

    def __init__(self, query_result_store: QueryResultStore | None = None) -> None:
        self.query_result_store = query_result_store or InMemoryQueryResultStore()
        self._toolset = FunctionToolset[Any](id=self.id)
        self._toolset.add_function(
            self.peek_sql_results,
            name="peek_sql_results",
            takes_ctx=True,
        )

    def get_toolset(self) -> FunctionToolset[Any]:
        return self._toolset

    async def peek_sql_results(self, ctx: RunContext[Any]) -> str:
        references = query_result_references_from_messages(ctx.messages)
        ok: list[dict[str, Any]] = []
        missing: list[str] = []
        for reference in references:
            descriptor = reference.descriptor
            if descriptor is None:
                missing.append(reference.file)
                continue
            try:
                loaded = await self.query_result_store.get(
                    descriptor.id,
                    context=QueryResultContext(),
                )
            except QueryResultUnavailable:
                missing.append(reference.file)
                continue
            ok.append(
                {
                    "file": reference.file,
                    "result_id": descriptor.id,
                    "rows": len(loaded.rows()),
                }
            )
        return json.dumps(
            {
                "ok": ok,
                "missing": missing,
                "store_id": id(self.query_result_store),
            }
        )


def _messages(run: Any) -> list[Any]:
    method = getattr(run, "new_messages", None)
    if callable(method):
        return list(method())
    method = getattr(run, "all_messages", None)
    if callable(method):
        return list(method())
    return []


def _tool_return(run: Any, name: str) -> ToolReturnPart | None:
    for message in _messages(run):
        for part in getattr(message, "parts", ()):
            if isinstance(part, ToolReturnPart) and part.tool_name == name:
                return part
    return None


def respond_execute_then_peek(messages: list[Any], info: Any) -> ModelResponse:
    del info
    parts = [part for message in messages for part in getattr(message, "parts", ())]
    if any(
        isinstance(part, ToolReturnPart) and part.tool_name == "peek_sql_results"
        for part in parts
    ):
        return ModelResponse(parts=[TextPart("done")])
    if any(
        isinstance(part, ToolReturnPart) and part.tool_name == "execute_sql"
        for part in parts
    ):
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "peek_sql_results",
                    {},
                    tool_call_id="peek1",
                )
            ]
        )
    return ModelResponse(
        parts=[
            ToolCallPart(
                "execute_sql",
                {"query": "SELECT id, name, revenue FROM customers ORDER BY id"},
                tool_call_id="sql1",
            )
        ]
    )


async def load_from_store(store: QueryResultStore, run: Any) -> list[dict[str, Any]]:
    loaded_rows: list[dict[str, Any]] = []
    for reference in query_result_references_from_messages(_messages(run)):
        descriptor = reference.descriptor
        if descriptor is None:
            continue
        loaded = await store.get(descriptor.id, context=QueryResultContext())
        loaded_rows.append(
            {
                "file": reference.file,
                "result_id": descriptor.id,
                "rows": len(loaded.rows()),
                "truncated_in_model": _execute_sql_truncated(run),
            }
        )
    return loaded_rows


def _execute_sql_truncated(run: Any) -> bool | None:
    part = _tool_return(run, "execute_sql")
    if part is None or not isinstance(part.content, str):
        return None
    try:
        payload = json.loads(part.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    truncated = payload.get("results_truncated")
    return truncated if isinstance(truncated, bool) else None


def _plugin_store_ids(capabilities: Sequence[Any]) -> dict[str, int | None]:
    found: dict[str, int | None] = {}
    for capability in capabilities:
        name = type(capability).__name__
        store = getattr(capability, "query_result_store", None)
        if store is None:
            tool = getattr(capability, "tool", None)
            store = getattr(tool, "query_result_store", None)
        if store is None:
            context = getattr(getattr(capability, "tool", None), "_context", None)
            store = getattr(context, "query_result_store", None)
        found[name] = id(store) if store is not None else None
    return found


@dataclass
class VariantReport:
    name: str
    host_rows: int
    peek_rows: int
    peek_missing: int
    model_truncated: bool | None
    stores_shared: bool
    plugin_constructors: dict[str, str]
    store_ids: dict[str, int | None]
    host_glue: str


async def run_agent(capabilities: list[Any]) -> Any:
    agent = Agent(
        FunctionModel(respond_execute_then_peek),
        instructions="Query customers, then peek at the stashed result.",
        capabilities=capabilities,
    )
    async with agent:
        return await agent.run("Top customers")


async def variant_a_thread_store(database: Path) -> VariantReport:
    store = InMemoryQueryResultStore()
    sql = SqlTools(database=str(database), query_result_store=store)
    peek = PeekResults(store)
    capabilities = [sql, peek]
    run = await run_agent(capabilities)
    host = await load_from_store(store, run)
    peek_part = _tool_return(run, "peek_sql_results")
    peek_payload = (
        json.loads(peek_part.content) if peek_part else {"ok": [], "missing": []}
    )
    return VariantReport(
        name="A thread-the-store",
        host_rows=host[0]["rows"] if host else 0,
        peek_rows=peek_payload["ok"][0]["rows"] if peek_payload.get("ok") else 0,
        peek_missing=len(peek_payload.get("missing") or []),
        model_truncated=_execute_sql_truncated(run),
        stores_shared=id(store)
        == id(sql.query_result_store)
        == id(peek.query_result_store),
        plugin_constructors={
            "SqlTools": "SqlTools(database=..., query_result_store=store)",
            "PeekResults": "PeekResults(store)",
            "Notebook": "Notebook(PluginContext(... query_result_store=store))",
        },
        store_ids=_plugin_store_ids(capabilities),
        host_glue=(
            "store = InMemoryQueryResultStore()\n"
            "sql = SqlTools(database=dsn, query_result_store=store)\n"
            "peek = PeekResults(store)\n"
            "agent = Agent(..., capabilities=[sql, peek])\n"
            "refs = query_result_references_from_messages(result.new_messages())\n"
            "loaded = await store.get(refs[0].descriptor.id, context=QueryResultContext())"
        ),
    )


async def variant_split_default_stores(database: Path) -> VariantReport:
    sql = SqlTools(database=str(database))
    peek = PeekResults()
    capabilities = [sql, peek]
    run = await run_agent(capabilities)
    host = await load_from_store(sql.query_result_store, run)
    peek_part = _tool_return(run, "peek_sql_results")
    peek_payload = (
        json.loads(peek_part.content) if peek_part else {"ok": [], "missing": []}
    )
    return VariantReport(
        name="split default stores (the trap)",
        host_rows=host[0]["rows"] if host else 0,
        peek_rows=peek_payload["ok"][0]["rows"] if peek_payload.get("ok") else 0,
        peek_missing=len(peek_payload.get("missing") or []),
        model_truncated=_execute_sql_truncated(run),
        stores_shared=id(sql.query_result_store) == id(peek.query_result_store),
        plugin_constructors={
            "SqlTools": "SqlTools(database=...)",
            "PeekResults": "PeekResults()",
            "Sandbox": "Sandbox() uses its own InMemoryQueryResultStore",
        },
        store_ids=_plugin_store_ids(capabilities),
        host_glue=(
            "sql = SqlTools(database=dsn)\n"
            "peek = PeekResults()\n"
            "agent = Agent(..., capabilities=[sql, peek])"
        ),
    )


class SqlSaberKit:
    """Proposed session object: stores plus capabilities, not the Agent."""

    def __init__(self, sql: SqlTools, *, model_name: str = "test:model") -> None:
        self.sql = sql
        self.query_result_store = sql.query_result_store
        self.artifact_store = InMemoryArtifactStore()
        self.knowledge_manager = KnowledgeManager()
        self.config = Config.in_memory(
            model_name=model_name,
            api_keys={"openai": "test-key"},
        )
        self._peek = PeekResults(self.query_result_store)

    @classmethod
    def for_database(cls, database: str) -> SqlSaberKit:
        store = InMemoryQueryResultStore()
        return cls(SqlTools(database=database, query_result_store=store))

    def plugin_context(self) -> PluginContext:
        return PluginContext(
            registry=self.sql.registry,
            knowledge_manager=self.knowledge_manager,
            allow_dangerous=self.sql.allow_dangerous,
            tool_overrides={},
            config=self.config,
            main_model_name="test:model",
            query_result_store=self.query_result_store,
            artifact_store=self.artifact_store,
        )

    def capabilities(self) -> list[Any]:
        discovered = discover_capabilities(self.plugin_context())
        return [self.sql, self._peek, *discovered]

    async def get_query_result(self, result: str | StoredQueryResult) -> Any:
        result_id = result.id if isinstance(result, StoredQueryResult) else result
        return await self.query_result_store.get(
            result_id,
            context=QueryResultContext(),
        )

    def query_results_from(self, run: Any) -> list[StoredQueryResult]:
        return [
            reference.descriptor
            for reference in query_result_references_from_messages(_messages(run))
            if reference.descriptor is not None
        ]


async def variant_b_kit(database: Path) -> VariantReport:
    kit = SqlSaberKit.for_database(str(database))
    capabilities = kit.capabilities()
    run = await run_agent(capabilities)
    descriptors = kit.query_results_from(run)
    host_rows = 0
    if descriptors:
        loaded = await kit.get_query_result(descriptors[0])
        host_rows = len(loaded.rows())
    peek_part = _tool_return(run, "peek_sql_results")
    peek_payload = (
        json.loads(peek_part.content) if peek_part else {"ok": [], "missing": []}
    )
    constructors = {
        type(capability).__name__: "kit.capabilities()" for capability in capabilities
    }
    return VariantReport(
        name="B SqlSaberKit session",
        host_rows=host_rows,
        peek_rows=peek_payload["ok"][0]["rows"] if peek_payload.get("ok") else 0,
        peek_missing=len(peek_payload.get("missing") or []),
        model_truncated=_execute_sql_truncated(run),
        stores_shared=len(
            {value for value in _plugin_store_ids(capabilities).values() if value}
        )
        == 1,
        plugin_constructors=constructors,
        store_ids=_plugin_store_ids(capabilities),
        host_glue=(
            "kit = SqlSaberKit.for_database(dsn)\n"
            "agent = Agent(..., capabilities=kit.capabilities())\n"
            "loaded = await kit.get_query_result(kit.query_results_from(result)[0])"
        ),
    )


@dataclass
class CapabilityRun:
    """Proposed adapter: SQLSaberResult affordances over Agent.run()."""

    run: Any
    store: QueryResultStore

    @classmethod
    def adopt(cls, run: Any, sql: SqlTools) -> CapabilityRun:
        return cls(run, sql.query_result_store)

    @property
    def query_results(self) -> list[StoredQueryResult]:
        return [
            reference.descriptor
            for reference in query_result_references_from_messages(_messages(self.run))
            if reference.descriptor is not None
        ]

    async def get_query_result(self, result: str | StoredQueryResult) -> Any:
        result_id = result.id if isinstance(result, StoredQueryResult) else result
        return await self.store.get(result_id, context=QueryResultContext())


async def variant_c_run_view(database: Path) -> VariantReport:
    store = InMemoryQueryResultStore()
    sql = SqlTools(database=str(database), query_result_store=store)
    peek = PeekResults(store)
    run = await run_agent([sql, peek])
    view = CapabilityRun.adopt(run, sql)
    host_rows = 0
    if view.query_results:
        loaded = await view.get_query_result(view.query_results[0])
        host_rows = len(loaded.rows())
    peek_part = _tool_return(run, "peek_sql_results")
    peek_payload = (
        json.loads(peek_part.content) if peek_part else {"ok": [], "missing": []}
    )
    return VariantReport(
        name="C adopt Agent.run()",
        host_rows=host_rows,
        peek_rows=peek_payload["ok"][0]["rows"] if peek_payload.get("ok") else 0,
        peek_missing=len(peek_payload.get("missing") or []),
        model_truncated=_execute_sql_truncated(run),
        stores_shared=id(store)
        == id(sql.query_result_store)
        == id(peek.query_result_store),
        plugin_constructors={
            "SqlTools": "still pass query_result_store into every capability",
            "CapabilityRun": "CapabilityRun.adopt(result, sql)",
        },
        store_ids=_plugin_store_ids([sql, peek]),
        host_glue=(
            "sql = SqlTools(database=dsn, query_result_store=store)\n"
            "result = await agent.run(...)\n"
            "view = CapabilityRun.adopt(result, sql)\n"
            "loaded = await view.get_query_result(view.query_results[0])"
        ),
    )


def print_report(report: VariantReport) -> None:
    print(f"\n== {report.name} ==")
    print(
        f"host_rows={report.host_rows} peek_rows={report.peek_rows} peek_missing={report.peek_missing}"
    )
    print(
        f"model_truncated={report.model_truncated} stores_shared={report.stores_shared}"
    )
    print(f"store_ids={report.store_ids}")
    print("plugin constructors:")
    for name, ctor in report.plugin_constructors.items():
        print(f"  {name}: {ctor}")
    print("host glue:")
    print(report.host_glue)


VARIANTS = {
    "a": variant_a_thread_store,
    "split": variant_split_default_stores,
    "b": variant_b_kit,
    "c": variant_c_run_view,
}


async def main(selected: Sequence[str]) -> None:
    names = list(selected) if selected else ["split", "a", "b", "c"]
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "customers.sqlite"
        seed_database(database)
        reports: list[VariantReport] = []
        for name in names:
            runner = VARIANTS[name]
            report = await runner(database)
            print_report(report)
            reports.append(report)
        print("\n== summary ==")
        print(
            f"{'variant':<28} {'host':>5} {'peek':>5} {'miss':>5} {'trunc':>6} {'shared':>6}"
        )
        for report in reports:
            print(
                f"{report.name:<28} {report.host_rows:>5} {report.peek_rows:>5} "
                f"{report.peek_missing:>5} {str(report.model_truncated):>6} "
                f"{str(report.stores_shared):>6}"
            )


if __name__ == "__main__":
    choice = [argument.lower() for argument in sys.argv[1:]]
    unknown = [name for name in choice if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    asyncio.run(main(choice))
