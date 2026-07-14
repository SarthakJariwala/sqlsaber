from __future__ import annotations

import os

import pytest

from sqlsaber_notebook.execution import (
    DEFAULT_NOTEBOOK_IMAGE,
    ExecutionLimits,
    NotebookInput,
)
from sqlsaber_notebook.execution.modal import ModalNotebookBackend
from sqlsaber_notebook.result import Workspace
from sqlsaber_notebook.session import NotebookSession

from _notebooks import assert_contract_result, contract_notebook, parse_notebook

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("SQLSABER_RUN_MODAL_INTEGRATION") != "1",
        reason="set SQLSABER_RUN_MODAL_INTEGRATION=1 for credentialed Modal tests",
    ),
]


async def test_live_modal_notebook_contract_and_termination() -> None:
    backend = ModalNotebookBackend()
    environment = await backend.open(
        [NotebookInput("data.json", b'{"values":[1,2,3]}')],
        image=DEFAULT_NOTEBOOK_IMAGE,
        limits=ExecutionLimits(),
    )
    sandbox = environment.sandbox
    session: NotebookSession | None = None
    try:
        first = await environment.execute(
            contract_notebook(),
            cell_timeout=120,
            command_timeout=600,
        )
        second = await environment.execute(
            contract_notebook(),
            cell_timeout=120,
            command_timeout=600,
        )
        assert_contract_result(first.notebook, expected_uid=1000)
        assert_contract_result(second.notebook, expected_uid=1000)
        assert [item.path for item in second.artifacts] == [
            "nested/summary.txt",
            "plot.png",
        ]
        summary = next(
            item for item in second.artifacts if item.path.endswith("summary.txt")
        )
        plot = next(item for item in second.artifacts if item.path == "plot.png")
        assert await environment.read_artifact(summary) == b"sum=6 counter=1"
        assert (await environment.read_artifact(plot)).startswith(b"\x89PNG\r\n\x1a\n")

        sources = [
            cell["source"] for cell in parse_notebook(contract_notebook())["cells"]
        ]
        session = NotebookSession(
            workspace=Workspace(()),
            backend=backend,
            image=DEFAULT_NOTEBOOK_IMAGE,
            cells=[
                "".join(source) if isinstance(source, list) else source
                for source in sources
            ],
        )
        session.environment = environment
        await session.run_notebook()
        assert session.run_count == 1
        assert_contract_result(session.notebook_bytes(), expected_uid=1000)
    finally:
        if session is not None:
            await session.close()
            await session.close()
        else:
            await environment.close()
            await environment.close()

    assert await sandbox.poll.aio() is not None
