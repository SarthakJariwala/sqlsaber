from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from sqlsaber_notebook.execution import (
    ExecutionLimits,
    NotebookBackendUnavailable,
    NotebookInput,
    NotebookLimitExceeded,
    resolve_notebook_backend,
    resolve_notebook_image,
)
from sqlsaber_notebook.execution.base import ArtifactInfo, NotebookInfrastructureError
from sqlsaber_notebook.execution.fake import FakeNotebookBackend, FakeRunResult


def notebook_bytes(label: str = "initial") -> bytes:
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "cell-1",
                    "metadata": {},
                    "outputs": [],
                    "source": [f"print({label!r})"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode()


def test_balanced_execution_defaults_support_classical_ml() -> None:
    limits = ExecutionLimits()
    assert limits.cpu_cores == 4.0
    assert limits.memory_mb == 8192
    assert limits.cell_seconds == 600
    assert limits.command_seconds is None
    assert limits.max_input_file_bytes == 100 * 1024 * 1024
    assert limits.max_total_input_bytes == 251 * 1024 * 1024
    assert limits.max_notebook_bytes == 50 * 1024 * 1024
    assert limits.max_total_artifact_bytes == 200 * 1024 * 1024


def test_backend_selection_is_explicit_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQLSABER_NOTEBOOK_BACKEND", "modal")
    assert resolve_notebook_backend().name == "modal"
    assert resolve_notebook_backend("docker").name == "docker"

    with pytest.raises(
        NotebookBackendUnavailable, match="expected 'docker' or 'modal'"
    ):
        resolve_notebook_backend("daytona")


def test_image_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQLSABER_NOTEBOOK_IMAGE", "example/image@sha256:abc")
    assert resolve_notebook_image() == "example/image@sha256:abc"


@pytest.mark.parametrize(
    "name", ["", ".hidden", "../secret", "nested/file", r"nested\file"]
)
async def test_fake_backend_rejects_unsafe_input_names(name: str) -> None:
    backend = FakeNotebookBackend()
    with pytest.raises(NotebookLimitExceeded):
        await backend.open(
            [NotebookInput(name, b"data")],
            image="unused",
            limits=ExecutionLimits(),
        )


async def test_fake_backend_has_clean_runs_immutable_inputs_and_lazy_artifacts() -> (
    None
):
    seen_inputs: list[dict[str, bytes]] = []

    def execute(
        notebook: bytes, inputs: Mapping[str, bytes], run: int
    ) -> FakeRunResult:
        seen_inputs.append(dict(inputs))
        mutable_inputs = dict(inputs)
        mutable_inputs["data.json"] = b"mutated"
        return FakeRunResult(
            notebook,
            {f"result-{run}.txt": f"run-{run}".encode()},
            stdout=f"run {run}",
        )

    backend = FakeNotebookBackend(execute)
    environment = await backend.open(
        [NotebookInput("data.json", b"original")],
        image="unused",
        limits=ExecutionLimits(),
    )

    first = await environment.execute(
        notebook_bytes(), cell_timeout=10, command_timeout=20
    )
    second = await environment.execute(
        notebook_bytes("second"), cell_timeout=10, command_timeout=20
    )

    assert seen_inputs == [
        {"data.json": b"original"},
        {"data.json": b"original"},
    ]
    # The executor receives a fresh mapping; mutating it never changes staged bytes.
    assert backend.environments[0].inputs == {"data.json": b"original"}
    assert first.stdout == "run 1"
    assert second.stdout == "run 2"
    assert first.artifacts == (ArtifactInfo("result-1.txt", 5, "text/plain"),)
    assert await environment.read_artifact(second.artifacts[0]) == b"run-2"
    with pytest.raises(NotebookInfrastructureError, match="Unknown artifact"):
        await environment.read_artifact(first.artifacts[0])

    await environment.close()
    await environment.close()
    with pytest.raises(NotebookInfrastructureError, match="closed"):
        await environment.list_workspace()


async def test_fake_backend_rejects_malformed_notebook() -> None:
    environment = await FakeNotebookBackend().open(
        [], image="unused", limits=ExecutionLimits()
    )
    with pytest.raises(NotebookInfrastructureError, match="malformed notebook"):
        await environment.execute(b"not json", cell_timeout=1, command_timeout=1)
    await environment.close()


async def test_fake_backend_enforces_artifact_limits() -> None:
    backend = FakeNotebookBackend(
        lambda notebook, inputs, run: FakeRunResult(
            notebook,
            {"large.bin": b"1234"},
        )
    )
    environment = await backend.open(
        [],
        image="unused",
        limits=ExecutionLimits(max_artifact_bytes=3),
    )
    with pytest.raises(NotebookLimitExceeded, match="Artifact exceeds"):
        await environment.execute(notebook_bytes(), cell_timeout=1, command_timeout=1)
    await environment.close()
