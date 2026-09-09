"""Runnable walkthrough for the capability-mode tutorial.

Uses only the current public SqlTools API plus query_result_resolution.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from sqlsaber import InMemoryQueryResultStore, SqlTools
from sqlsaber.query_result_resolution import query_result_references_from_messages
from sqlsaber.query_results import QueryResultContext


def seed(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, revenue INTEGER)"
    )
    connection.executemany(
        "INSERT INTO customers (name, revenue) VALUES (?, ?)",
        [(f"customer-{index:03d}-" + ("x" * 24), 1000 + index) for index in range(250)],
    )
    connection.commit()
    connection.close()


def respond(messages, info):
    del info
    parts = [part for message in messages for part in message.parts]
    if any(isinstance(part, ToolReturnPart) for part in parts):
        return ModelResponse(parts=[TextPart("Listed the customers.")])
    return ModelResponse(
        parts=[
            ToolCallPart(
                "execute_sql",
                {"query": "SELECT id, name, revenue FROM customers ORDER BY id"},
                tool_call_id="sql1",
            )
        ]
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "customers.sqlite"
        seed(database)
        store = InMemoryQueryResultStore()
        sql = SqlTools(database=str(database), query_result_store=store)
        agent = Agent(
            FunctionModel(respond),
            instructions="You query the customers table.",
            capabilities=[sql],
        )
        async with agent:
            result = await agent.run("Show every customer")
        execute_sql = next(
            part
            for message in result.new_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == "execute_sql"
        )
        preview = json.loads(execute_sql.content)
        print("model_truncated", preview.get("results_truncated"))
        print(
            "preview_row_count",
            len(preview.get("preview_rows") or preview.get("results") or []),
        )
        references = query_result_references_from_messages(result.new_messages())
        loaded = await store.get(
            references[0].descriptor.id,
            context=QueryResultContext(),
        )
        print("stored_row_count", len(loaded.rows()))
        print("first_row", loaded.rows()[0])
        print("last_row", loaded.rows()[-1])


if __name__ == "__main__":
    asyncio.run(main())
