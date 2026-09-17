from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from sqlsaber_notebook.execution import (
    DEFAULT_NOTEBOOK_IMAGE,
    ExecutionLimits,
    NotebookInput,
)
from sqlsaber_notebook.execution.e2b import E2BNotebookBackend
from sqlsaber_notebook.result import Workspace
from sqlsaber_notebook.session import NotebookSession

from _notebooks import (
    assert_contract_result,
    contract_notebook,
    parse_notebook,
    stream_text,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("SQLSABER_RUN_E2B_INTEGRATION") != "1",
        reason="set SQLSABER_RUN_E2B_INTEGRATION=1 for credentialed E2B tests",
    ),
]


async def test_live_e2b_notebook_contract_and_termination() -> None:
    from e2b import AsyncSandbox

    backend = E2BNotebookBackend()
    environment = await backend.open(
        [NotebookInput("data.json", b'{"values":[1,2,3]}')],
        image=DEFAULT_NOTEBOOK_IMAGE,
        limits=ExecutionLimits(),
    )
    sandbox = environment.sandbox
    print(f"E2B sandbox: {sandbox.sandbox_id}")
    try:
        # Positive control: the same probe must reach an HTTP responder when
        # egress is enabled, otherwise a broken probe could falsely pass denial.
        control = await AsyncSandbox.create(
            template=(await sandbox.get_info()).template_id,
            timeout=60,
            allow_internet_access=True,
        )
        try:
            probe = (
                "import socket\n"
                "with socket.create_connection(('1.1.1.1', 80), timeout=3) as connection:\n"
                "    connection.sendall(b'GET / HTTP/1.1\\r\\nHost: 1.1.1.1\\r\\nConnection: close\\r\\n\\r\\n')\n"
                "    print(connection.recv(1024).decode())\n"
            )
            response = await control.commands.run(
                shlex.join(["/usr/bin/python3", "-c", probe])
            )
            assert response.stdout.startswith("HTTP/")
            print("Positive control: HTTP response received with E2B network enabled")
        finally:
            await control.kill()
        assert not await control.is_running()
        for run in range(2):
            result = await environment.execute(
                contract_notebook(), cell_timeout=120, command_timeout=600
            )
            assert_contract_result(result.notebook, expected_uid=1000)
            assert [item.path for item in result.artifacts] == [
                "nested/summary.txt",
                "plot.png",
            ]
            summary, plot = result.artifacts
            assert await environment.read_artifact(summary) == b"sum=6 counter=1"
            png = await environment.read_artifact(plot)
            assert png.startswith(b"\x89PNG\r\n\x1a\n")
            cells = parse_notebook(result.notebook)["cells"]
            print(f"Run {run + 1}: {stream_text(cells[0]).strip()}")
            print(
                f"Run {run + 1}: intentional RuntimeError retained; {stream_text(cells[2]).strip()}"
            )
            print(f"Downloaded summary: sum=6 counter=1; PNG: {len(png)} bytes")
            if directory := os.getenv("SQLSABER_E2B_EVIDENCE_DIR"):
                evidence = Path(directory)
                evidence.mkdir(parents=True, exist_ok=True)
                (evidence / "notebook.ipynb").write_bytes(result.notebook)
                (evidence / "plot.png").write_bytes(png)
        session = NotebookSession(
            workspace=Workspace(()),
            backend=backend,
            image=DEFAULT_NOTEBOOK_IMAGE,
            cells=[
                "".join(cell["source"])
                for cell in parse_notebook(contract_notebook())["cells"]
            ],
        )
        # Reuse the real environment to exercise transactional session publication.
        session.environment = environment
        await session.run_notebook()
        assert session.run_count == 1
        assert_contract_result(session.notebook_bytes(), expected_uid=1000)
        print("NotebookSession: transactional notebook publication passed")
        await session.close()
    finally:
        await environment.close()
        await environment.close()
    assert not await sandbox.is_running()
    print("E2B sandbox is_running=False after close (verified with provider)")
