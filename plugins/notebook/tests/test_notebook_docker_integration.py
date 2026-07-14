from __future__ import annotations

import os
import shutil

import pytest

from sqlsaber_notebook.execution import (
    DEFAULT_NOTEBOOK_IMAGE,
    DockerNotebookBackend,
    ExecutionLimits,
    NotebookBackendUnavailable,
    NotebookInput,
)
from sqlsaber_notebook.result import ManifestEntry, Workspace
from sqlsaber_notebook.session import NotebookSession

from _notebooks import assert_contract_result, contract_notebook

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable")
async def test_live_docker_notebook_contract() -> None:
    backend = DockerNotebookBackend()
    try:
        environment = await backend.open(
            [NotebookInput("data.json", b'{"values":[1,2,3]}')],
            image=DEFAULT_NOTEBOOK_IMAGE,
            limits=ExecutionLimits(),
        )
    except NotebookBackendUnavailable as exc:
        pytest.skip(str(exc))

    root = environment.root
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
        expected_uid = os.getuid() if hasattr(os, "getuid") else 1000
        assert_contract_result(first.notebook, expected_uid=expected_uid)
        assert_contract_result(second.notebook, expected_uid=expected_uid)
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
    finally:
        await environment.close()
        await environment.close()

    assert not root.exists()


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable")
async def test_live_docker_session_uses_logical_input_paths() -> None:
    backend = DockerNotebookBackend()
    session = NotebookSession(
        workspace=Workspace(
            (NotebookInput("data.json", b'{"value":42}'),),
            (ManifestEntry("data.json", source="test"),),
        ),
        backend=backend,
        image=DEFAULT_NOTEBOOK_IMAGE,
        cells=[
            "from pathlib import Path\n"
            "import json\n"
            "value = json.loads(Path('../inputs/data.json').read_text())['value']\n"
            "Path('answer.txt').write_text(str(value))\n"
            "print(value)"
        ],
    )
    try:
        try:
            await session.run_notebook()
        except NotebookBackendUnavailable as exc:
            pytest.skip(str(exc))
        assert session.run_count == 1
        text = session.outputs[0][0]["text"]
        assert ("".join(text) if isinstance(text, list) else text) == "42\n"
        artifact = next(item for item in session.artifacts if item.path == "answer.txt")
        assert session.environment is not None
        assert await session.environment.read_artifact(artifact) == b"42"
    finally:
        await session.close()
