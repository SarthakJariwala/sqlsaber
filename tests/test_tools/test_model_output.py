"""Model CSV presentation stays separate from structured storage and rendering."""

import csv
import datetime
import io
import json
from decimal import Decimal

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from sqlsaber.capabilities import SqlTools
from sqlsaber.database.registry import DatabaseEntry, DatabaseRegistry
from sqlsaber.database.sqlite import SQLiteConnection
from sqlsaber.query_result_resolution import query_result_references_from_messages
from sqlsaber.query_results import (
    QueryResultContext,
    build_model_projection,
    descriptor_for_data,
)
from sqlsaber.tools.model_output import format_sql_output
from sqlsaber.tools.renderer import ToolRenderContext, ToolRenderer


def test_csv_cells_and_database_types() -> None:
    payload = {
        "results": [
            {"value": None},
            {"value": ""},
            {"value": r"\N"},
            {"value": 'a,b"c\r\nd\re'},
            {"value": "東京"},
            {"value": Decimal("12.5")},
            {"value": datetime.date(2026, 9, 6)},
            {"value": b"abc"},
            {"value": {"nested": [True, None]}},
            {"other": 42},
        ]
    }
    output = format_sql_output("execute_sql", payload)
    rows = list(csv.reader(io.StringIO(output.split("\n", 2)[2], newline="")))
    assert rows[0] == ["value", "other"]
    assert [row[0] for row in rows[1:]] == [
        r"\N",
        "",
        r"\\N",
        'a,b"c\r\nd\re',
        "東京",
        "12.5",
        "2026-09-06",
        "YWJj",
        '{"nested":[true,null]}',
        r"\N",
    ]
    assert rows[-1][1] == "42"


def test_schema_keeps_constraints_comments_and_input() -> None:
    payload = {
        "main.users": {
            "comment": "Users, including archived",
            "columns": {"id": {"type": "INTEGER", "nullable": False, "default": None}},
            "primary_keys": ["id"],
            "foreign_keys": ["id -> accounts.id"],
            "indexes": ["users_pk (id) UNIQUE"],
        }
    }
    before = json.dumps(payload)
    output = format_sql_output("introspect_schema", payload)
    metadata = json.loads(output.splitlines()[0])
    assert metadata == {
        "table": "main.users",
        **{k: v for k, v in payload["main.users"].items() if k != "columns"},
    }
    assert "name,type,nullable,default\r\nid,INTEGER,false,\\N" in output
    assert json.dumps(payload) == before


@pytest.mark.parametrize(
    "payload", [{"error": "bad query"}, {"success": True}, {}, {"results": []}]
)
def test_non_tabular_output_stays_json(payload) -> None:
    assert json.loads(format_sql_output("execute_sql", payload)) == payload


def test_csv_preview_obeys_byte_budget() -> None:
    payload = {"results": [{'"' * 80: "東京," * 100}] * 20}
    descriptor = descriptor_for_data(
        b"{}",
        result_id="qr_" + "a" * 32,
        file="result_test.json",
        row_count=20,
        columns=('"' * 80,),
    )
    projection = build_model_projection(
        payload, descriptor, max_bytes=1024, csv_tool_results=True
    )
    assert projection["results_truncated"] is True
    assert len(format_sql_output("execute_sql", projection).encode()) <= 1024
    assert projection["result_id"] == descriptor.id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name", ["list_tables", "introspect_schema", "execute_sql", "list_dbs"]
)
@pytest.mark.parametrize("csv_tool_results", [None, False, True])
async def test_agent_output_format_and_replay_preserve_structured_results(
    name, tmp_path, csv_tool_results
) -> None:
    connection = SQLiteConnection(f"sqlite:///{tmp_path / 'test.db'}")
    registry = DatabaseRegistry(
        [
            DatabaseEntry.from_connection(
                name="test",
                connection=connection,
                description=None,
                excluded_schemas=[],
            )
        ]
    )
    if name == "list_dbs":
        registry = DatabaseRegistry(
            [
                *registry,
                DatabaseEntry.from_connection(
                    name="second",
                    connection=connection,
                    description="Another DB",
                    excluded_schemas=[],
                ),
            ]
        )
    options = {} if csv_tool_results is None else {"csv_tool_results": csv_tool_results}
    capability = SqlTools(registry=registry, **options)
    await connection.execute_query(
        "CREATE TABLE users (id INTEGER, name TEXT)", commit=True
    )
    await connection.execute_query("INSERT INTO users VALUES (1, 'Ada')", commit=True)

    def respond(messages, info):
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart("Done")])
        args = {"query": "SELECT * FROM users"} if name == "execute_sql" else {}
        return ModelResponse(parts=[ToolCallPart(name, args, tool_call_id="test_call")])

    model = FunctionModel(respond)
    try:
        agent = Agent(model, capabilities=[capability])
        result = await agent.run("Inspect users")
        messages = ModelMessagesTypeAdapter.validate_json(
            ModelMessagesTypeAdapter.dump_json(result.all_messages())
        )
        part = next(
            p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)
        )
        if csv_tool_results:
            assert "CSV;" in part.content
            assert isinstance(part.metadata, dict)
            payload = part.metadata["sqlsaber_structured_result"]
        else:
            payload = json.loads(part.content)
            assert "sqlsaber_structured_result" not in (part.metadata or {})
        renderer = ToolRenderer(capability.display_specs)
        assert renderer.result(
            name, part.content, context=ToolRenderContext(metadata=part.metadata)
        ) == renderer.result(name, payload)
        if name == "execute_sql":
            references = query_result_references_from_messages(messages)
            assert len(references) == 1
            stored = await capability.query_result_store.get(
                references[0].descriptor.id,
                context=QueryResultContext(),
            )
            assert stored.rows() == [{"id": 1, "name": "Ada"}]
    finally:
        await registry.close()
