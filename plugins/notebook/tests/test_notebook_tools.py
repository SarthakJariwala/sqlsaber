from __future__ import annotations

import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic_ai import RunContext

from sqlsaber_notebook.execution.base import NotebookInfrastructureError
from sqlsaber_notebook.execution.fake import FakeNotebookBackend, FakeRunResult
from sqlsaber_notebook.result import Workspace
from sqlsaber_notebook.session import NotebookSession
from sqlsaber_notebook.tools import _coerce_idx, edit_cell, list_workspace


def _ctx(session: NotebookSession) -> RunContext[NotebookSession]:
    return cast(RunContext[NotebookSession], SimpleNamespace(deps=session))


def _echo_execution(
    notebook: bytes,
    inputs: Mapping[str, bytes],
    run: int,
) -> FakeRunResult:
    del inputs, run
    return FakeRunResult(notebook)


@pytest.mark.parametrize("value", [-1, True, False, "0", 1.5])
def test_cell_index_rejects_negative_boolean_and_malformed(value: object) -> None:
    with pytest.raises(ValueError):
        _coerce_idx(value, 1)


def test_cell_index_rejects_out_of_range_and_only_none_appends() -> None:
    assert _coerce_idx(None, 0) is None
    assert _coerce_idx(0, 1) == 0
    with pytest.raises(ValueError, match="omit idx to append"):
        _coerce_idx(1, 1)


async def test_edit_cell_appends_edits_and_returns_current_snapshot() -> None:
    backend = FakeNotebookBackend(_echo_execution)
    session = NotebookSession(workspace=Workspace(()), backend=backend, image="unused")

    appended = await edit_cell(_ctx(session), "x = 1")
    edited = await edit_cell(_ctx(session), "x = 2", 0)

    assert appended.return_value == "Appended cell #0."
    assert edited.return_value == "Edited cell #0."
    assert session.cells == ["x = 2"]
    assert session.run_count == 2
    assert isinstance(edited.content, list)
    snapshot = edited.content[0]
    assert isinstance(snapshot, str)
    assert "Current notebook state:" in snapshot
    assert "x = 2" in snapshot
    await session.close()


async def test_invalid_index_does_not_execute() -> None:
    backend = FakeNotebookBackend(_echo_execution)
    session = NotebookSession(
        workspace=Workspace(()),
        backend=backend,
        image="unused",
        cells=["x = 1"],
    )
    result = await edit_cell(_ctx(session), "bad", -1)
    assert "not applied" in str(result.return_value)
    assert session.cells == ["x = 1"]
    assert session.run_count == 0
    assert backend.environments == []


async def test_execution_failure_rolls_back_source_outputs_and_artifacts() -> None:
    def fail(notebook: bytes, inputs: Mapping[str, bytes], run: int) -> FakeRunResult:
        del notebook, inputs, run
        raise NotebookInfrastructureError(
            "executor unavailable",
            backend="fake",
            phase="notebook-execution",
        )

    session = NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(fail),
        image="unused",
        cells=["x = 1"],
        outputs=[[{"output_type": "stream", "name": "stdout", "text": "old"}]],
    )
    result = await edit_cell(_ctx(session), "x = 2", 0)
    assert "not applied" in str(result.return_value)
    assert session.cells == ["x = 1"]
    assert session.outputs[0][0]["text"] == "old"
    await session.close()


async def test_session_lock_serializes_parallel_edits() -> None:
    active = 0
    max_active = 0

    async def execute(
        notebook: bytes, inputs: Mapping[str, bytes], run: int
    ) -> FakeRunResult:
        nonlocal active, max_active
        del inputs, run
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return FakeRunResult(notebook)

    session = NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(execute),
        image="unused",
    )
    await asyncio.gather(
        edit_cell(_ctx(session), "a = 1"),
        edit_cell(_ctx(session), "b = 2"),
    )
    assert max_active == 1
    assert session.cells == ["a = 1", "b = 2"]
    assert session.run_count == 2
    await session.close()


async def test_list_workspace_uses_session_metadata() -> None:
    session = NotebookSession(
        workspace=Workspace(()), backend=FakeNotebookBackend(), image="unused"
    )
    payload = await list_workspace(_ctx(session))
    assert '"working_directory": "run/"' in payload
    await session.close()
