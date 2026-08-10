from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import pytest

from sqlsaber_notebook.execution import (
    DEFAULT_NOTEBOOK_IMAGE,
    ExecutionLimits,
    NotebookInput,
    NotebookLimitExceeded,
)
from sqlsaber_notebook.execution.daytona import (
    DaytonaNotebookBackend,
    _load_daytona,
)

from _notebooks import assert_contract_result, contract_notebook


def _single_cell_notebook(source: str) -> bytes:
    notebook: dict[str, Any] = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "cell-1",
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook).encode()


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("SQLSABER_RUN_DAYTONA_INTEGRATION") != "1",
        reason="set SQLSABER_RUN_DAYTONA_INTEGRATION=1 for Daytona tests",
    ),
]


async def _assert_deleted(name: str) -> None:
    sdk = _load_daytona()
    client = sdk.AsyncDaytona()
    try:
        with pytest.raises(sdk.DaytonaNotFoundError):
            await client.get(name)
    finally:
        await client.close()


async def test_live_daytona_notebook_contract_and_deletion() -> None:
    backend = DaytonaNotebookBackend()
    environment = await backend.open(
        [NotebookInput("data.json", b'{"values":[1,2,3]}')],
        image=DEFAULT_NOTEBOOK_IMAGE,
        limits=ExecutionLimits(),
    )
    name = environment.sandbox_name
    assert environment.remote_root.startswith("/sqlsaber-notebook/")
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

        background = await environment.execute(
            _single_cell_notebook(
                """import subprocess
import sys
import time
from pathlib import Path
Path('background.txt').write_bytes(b'start')
subprocess.Popen([
    sys.executable,
    '-c',
    "import time; p='background.txt'; "
    "[(open(p, 'ab').write(b'x'), time.sleep(.02)) for _ in iter(int, 1)]",
])
time.sleep(.1)
"""
            ),
            cell_timeout=120,
            command_timeout=600,
        )
        artifact = next(
            item for item in background.artifacts if item.path == "background.txt"
        )
        await asyncio.sleep(0.2)
        assert len(await environment.read_artifact(artifact)) == artifact.size

        with pytest.raises(NotebookLimitExceeded, match="unsafe files"):
            await environment.execute(
                _single_cell_notebook(
                    """from pathlib import Path
Path('target.txt').write_text('target')
Path('unsafe-link').symlink_to('target.txt')
"""
                ),
                cell_timeout=120,
                command_timeout=600,
            )
    finally:
        await environment.close()
        await environment.close()

    await _assert_deleted(name)


async def test_live_daytona_cancellation_deletes_sandbox() -> None:
    environment = await DaytonaNotebookBackend().open(
        [],
        image=DEFAULT_NOTEBOOK_IMAGE,
        limits=ExecutionLimits(),
    )
    name = environment.sandbox_name
    task = asyncio.create_task(
        environment.execute(
            _single_cell_notebook("import time; time.sleep(300)"),
            cell_timeout=None,
            command_timeout=None,
        )
    )
    await asyncio.sleep(2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_deleted(name)
