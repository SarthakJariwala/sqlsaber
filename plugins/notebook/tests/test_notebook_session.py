from __future__ import annotations

import json
from collections.abc import Mapping

import nbformat
import pytest

from sqlsaber_notebook import session as session_module
from sqlsaber_notebook.execution import NotebookInput, NotebookLimitExceeded
from sqlsaber_notebook.execution.base import NotebookInfrastructureError
from sqlsaber_notebook.execution.fake import FakeNotebookBackend, FakeRunResult
from sqlsaber_notebook.result import ManifestEntry, Workspace, WorkspaceFile
from sqlsaber_notebook.session import NotebookSession


def _executed(
    notebook: bytes,
    inputs: Mapping[str, bytes],
    run_count: int,
) -> FakeRunResult:
    assert json.loads(inputs["data.json"]) == {"values": [1, 2, 3]}
    assert "manifest.json" in inputs
    parsed = nbformat.reads(notebook.decode(), as_version=4)
    for index, cell in enumerate(parsed.cells):
        cell.execution_count = index + 1
        cell.outputs = [
            nbformat.v4.new_output(
                "stream",
                name="stdout",
                text=f"run={run_count} cell={index}\n",
            )
        ]
    return FakeRunResult(
        nbformat.writes(parsed).encode(),
        {"result.txt": f"run-{run_count}".encode()},
    )


async def test_session_stages_manifest_executes_and_installs_atomically() -> None:
    backend = FakeNotebookBackend(_executed)
    workspace = Workspace(
        (NotebookInput("data.json", b'{"values":[1,2,3]}'),),
        (ManifestEntry("data.json", sql="select 1"),),
    )
    session = NotebookSession(
        workspace=workspace,
        backend=backend,
        image="unused",
        cells=["print('one')", "print('two')"],
    )

    assert session.environment is None
    await session.run_notebook()
    assert session.run_count == 1
    assert session.outputs[0][0]["text"] == "run=1 cell=0\n"
    assert [item.path for item in session.artifacts] == ["result.txt"]
    assert set(backend.environments[0].inputs) == {"data.json", "manifest.json"}

    workspace_json = json.loads(await session.list_workspace())
    assert workspace_json["inputs"][0]["sql"] == "select 1"
    assert workspace_json["generated"][0]["path"] == "result.txt"

    await session.close()
    await session.close()
    assert backend.environments[0].closed is True


def test_workspace_file_rejects_mutable_bytes_and_invalid_metadata() -> None:
    with pytest.raises(TypeError, match="data must be bytes"):
        WorkspaceFile("data.csv", bytearray(b"data"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="media_type"):
        WorkspaceFile("data.csv", b"data", media_type=42)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="provenance"):
        WorkspaceFile(
            "data.csv",
            b"data",
            provenance={"attachment_id": 42},  # type: ignore[dict-item]
        )


async def test_workspace_files_preserve_legacy_tuples_and_stage_rich_metadata() -> None:
    legacy = Workspace.from_files([("legacy.csv", b"value\n1\n")], source="caller")
    assert legacy.files == (NotebookInput("legacy.csv", b"value\n1\n"),)
    assert legacy.manifest == (ManifestEntry("legacy.csv", source="caller"),)

    provenance = {"attachment_id": "attachment-1", "variant": "preview"}
    workspace = Workspace.from_files(
        [
            WorkspaceFile(
                "preview.jpeg",
                b"jpeg-bytes",
                media_type="image/jpeg",
                provenance=provenance,
            )
        ]
    )
    provenance["attachment_id"] = "changed"
    session = NotebookSession(
        workspace=workspace,
        backend=FakeNotebookBackend(),
        image="unused",
    )

    await session.ensure_environment()
    staged = session.environment
    assert staged is not None
    manifest = json.loads(staged.inputs["manifest.json"])
    assert manifest == [
        {
            "file": "../inputs/preview.jpeg",
            "media_type": "image/jpeg",
            "provenance": {
                "attachment_id": "attachment-1",
                "variant": "preview",
            },
            "source": None,
            "sql": None,
        }
    ]
    workspace_json = json.loads(await session.list_workspace())
    assert workspace_json["inputs"][0]["media_type"] == "image/jpeg"
    assert workspace_json["inputs"][0]["provenance"] == {
        "attachment_id": "attachment-1",
        "variant": "preview",
    }
    await session.close()


async def test_session_rejects_oversized_manifest_before_backend_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_module, "MAX_WORKSPACE_MANIFEST_BYTES", 100)
    backend = FakeNotebookBackend()
    session = NotebookSession(
        workspace=Workspace.from_files(
            [
                WorkspaceFile(
                    "data.bin",
                    b"data",
                    provenance={"description": "x" * 200},
                )
            ]
        ),
        backend=backend,
        image="unused",
    )

    with pytest.raises(NotebookLimitExceeded, match="manifest exceeds 100 bytes"):
        await session.ensure_environment()
    assert backend.environments == []
    await session.close()


async def test_session_rejects_backend_source_changes_without_installing_state() -> (
    None
):
    def mutate_source(
        notebook: bytes, inputs: Mapping[str, bytes], run_count: int
    ) -> FakeRunResult:
        del inputs, run_count
        parsed = nbformat.reads(notebook.decode(), as_version=4)
        parsed.cells[0].source = "tampered = True"
        return FakeRunResult(nbformat.writes(parsed).encode())

    session = NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(mutate_source),
        image="unused",
        cells=["original = True"],
        outputs=[[{"output_type": "stream", "name": "stdout", "text": "old"}]],
    )
    old_notebook = session.notebook_bytes()
    with pytest.raises(NotebookInfrastructureError, match="changed notebook source"):
        await session.run_notebook()
    assert session.outputs[0][0]["text"] == "old"
    assert session.notebook_bytes() == old_notebook
    assert session.run_count == 0
    await session.close()


def test_session_does_not_impose_a_cell_count_policy() -> None:
    session = NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(),
        image="unused",
        cells=[f"value_{index} = {index}" for index in range(75)],
    )
    assert len(session.cells) == 75


async def test_blank_session_returns_valid_notebook_without_execution() -> None:
    session = NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(),
        image="unused",
    )
    parsed = nbformat.reads(session.notebook_bytes().decode(), as_version=4)
    assert parsed.cells == []
    await session.ensure_environment()
    await session.close()
