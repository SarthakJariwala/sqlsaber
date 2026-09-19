"""Configuration parity through real capability and analyst orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from sqlsaber.artifacts import InMemoryArtifactStore
from sqlsaber.capabilities.plugins import resolve_capability_specs
from sqlsaber.query_results import InMemoryQueryResultStore
from sqlsaber_notebook import ExecutionLimits, NotebookConfig, WorkspaceLimits, analyze
from sqlsaber_notebook import execution
from sqlsaber_notebook.capability import Notebook, capability
from sqlsaber_notebook.execution import NotebookInput, NotebookLimitExceeded
from sqlsaber_notebook.execution.fake import FakeNotebookBackend, FakeRunResult
from sqlsaber_notebook.result import Workspace, WorkspaceFile, workspace_manifest_bytes
from sqlsaber_notebook.session import NotebookSession


class RecordingBackend(FakeNotebookBackend):
    def __init__(self) -> None:
        super().__init__(
            lambda notebook, inputs, run: FakeRunResult(
                notebook, {"model.bin": b"trained-model"}
            )
        )
        self.images: list[str] = []
        self.executions: list[AsyncMock] = []

    async def open(
        self, inputs: Sequence[NotebookInput], *, image: str, limits: ExecutionLimits
    ):
        environment = await super().open(inputs, image=image, limits=limits)
        self.images.append(image)
        execute = AsyncMock(wraps=environment.execute)
        environment.execute = execute
        self.executions.append(execute)
        return environment


async def run_analysis(
    managed: bool, config: NotebookConfig, files: list[WorkspaceFile]
):
    model = TestModel(call_tools=["edit_cell"], custom_output_text="Analyzed")
    if not managed:
        return await analyze(
            "Analyze all inputs",
            Workspace.from_files(files),
            model=model,
            model_provider="test",
            config=config,
        )

    class Resolver:
        async def resolve(self, refs, *, context):
            return files

    context = SimpleNamespace(
        query_result_store=InMemoryQueryResultStore(),
        workspace_input_resolver=Resolver(),
        artifact_store=InMemoryArtifactStore(),
        resolve_subagent_model=lambda *args, **kwargs: SimpleNamespace(
            model=model, model_name="test", provider="test"
        ),
    )
    capabilities = resolve_capability_specs(
        [partial(capability, config=config)], cast(Any, context)
    )
    assert len(capabilities) == 1
    notebook = capabilities[0]
    assert isinstance(notebook, Notebook)
    ctx = SimpleNamespace(
        messages=[],
        usage=RunUsage(),
        tool_call_id="analysis",
        metadata={},
        run_id="run",
        conversation_id="conversation",
    )
    return await notebook.tool.execute_with_attachments(
        cast(Any, ctx), "Analyze all inputs", attachment_refs=["authorized-collection"]
    )


@pytest.mark.parametrize("managed", [False, True])
async def test_config_reaches_admission_execution_and_export(managed, monkeypatch):
    backend = RecordingBackend()
    monkeypatch.setenv("SQLSABER_NOTEBOOK_BACKEND", "must-not-be-used")
    monkeypatch.setenv("SQLSABER_NOTEBOOK_IMAGE", "environment-image")
    monkeypatch.setattr(execution, "DockerNotebookBackend", lambda: backend)
    config = NotebookConfig(
        backend="docker",
        image="custom-ml-image@sha256:abc",
        workspace=WorkspaceLimits(max_files=60, max_file_bytes=7, max_total_bytes=420),
        cell_seconds=3600,
        command_seconds=7200,
        cpu_cores=6.5,
        memory_mb=16384,
        max_artifacts=2,
        max_artifact_bytes=13,
        max_total_artifact_bytes=13,
    )
    files = [WorkspaceFile(f"image-{i}.bin", b"pixels!") for i in range(60)]
    result = await run_analysis(managed, config, files)
    if managed:
        assert isinstance(result, ToolReturn)
        assert result.return_value == "Analyzed"
        artifacts = result.metadata["artifact_publication"]["artifacts"]
        assert any(item["name"] == "files/model.bin" for item in artifacts)
    else:
        assert result.answer == "Analyzed"
        assert result.files[0].data == b"trained-model"
    assert backend.images == ["custom-ml-image@sha256:abc"]
    environment = backend.environments[0]
    assert len(environment.inputs) == 61
    assert environment.inputs["image-59.bin"] == b"pixels!"
    assert len(json.loads(environment.inputs["manifest.json"])) == 60
    assert environment.limits.max_total_input_bytes == 420 + 1024**2
    assert environment.limits.cpu_cores == 6.5
    assert environment.limits.memory_mb == 16384
    assert environment.limits.max_artifact_bytes == 13
    assert environment.closed
    assert backend.executions[0].call_args.kwargs == {
        "cell_timeout": 3600,
        "command_timeout": 7200,
    }


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("max_files", 1),
        ("max_file_bytes", 3),
        ("max_total_bytes", 6),
        ("max_manifest_bytes", 1),
    ],
)
async def test_workspace_rejection_is_identical_before_provisioning(
    managed, field, value
):
    backend = RecordingBackend()
    config = NotebookConfig(
        backend=backend,
        workspace=replace(WorkspaceLimits(), **{field: value}),
    )
    files = [WorkspaceFile("a.bin", b"123"), WorkspaceFile("b.bin", b"4567")]
    if managed:
        result = await run_analysis(True, config, files)
        assert isinstance(result, str)
        assert json.loads(result)["phase"] == "input-validation"
    else:
        with pytest.raises(NotebookLimitExceeded):
            await run_analysis(False, config, files)
    assert not backend.environments


@pytest.mark.parametrize("managed", [False, True])
async def test_custom_config_exceeds_old_workspace_and_manifest_caps(managed):
    backend = RecordingBackend()
    mib = 1024**2
    # Share immutable bytes across file entries to test real sizes without a
    # multi-gigabyte host allocation. The fake backend does not copy the bytes.
    data = b"x" * (101 * mib)
    files = [
        WorkspaceFile(
            f"image-{i}.bin",
            data,
            provenance={f"field-{j}": "x" * 2000 for j in range(10)},
        )
        for i in range(60)
    ]
    config = NotebookConfig(
        backend=backend,
        workspace=WorkspaceLimits(
            max_files=60,
            max_file_bytes=101 * mib,
            max_total_bytes=6060 * mib,
            max_manifest_bytes=2 * mib,
        ),
    )
    result = await run_analysis(managed, config, files)
    if managed:
        assert isinstance(result, ToolReturn)
    else:
        assert result.answer == "Analyzed"
    inputs = backend.environments[0].inputs
    assert (
        sum(len(value) for name, value in inputs.items() if name != "manifest.json")
        == 6060 * mib
    )
    assert mib < len(inputs["manifest.json"]) < 2 * mib


async def test_manifest_has_its_own_budget_without_lending_it_to_user_files():
    backend = RecordingBackend()
    files = [WorkspaceFile("a.bin", b"123"), WorkspaceFile("b.bin", b"4567")]
    manifest = workspace_manifest_bytes(Workspace.from_files(files))
    config = NotebookConfig(
        backend=backend,
        workspace=WorkspaceLimits(
            max_files=2,
            max_file_bytes=4,
            max_total_bytes=7,
            max_manifest_bytes=len(manifest),
        ),
    )
    await run_analysis(False, config, files)
    assert backend.environments[0].inputs["manifest.json"] == manifest
    assert backend.environments[0].limits.max_input_file_bytes == len(manifest)
    assert backend.environments[0].limits.max_total_input_bytes == 7 + len(manifest)
    files[1] = WorkspaceFile("b.bin", b"45678")
    with pytest.raises(NotebookLimitExceeded, match="exceeds 4 bytes"):
        await run_analysis(False, config, files)
    assert len(backend.environments) == 1


@pytest.mark.parametrize("managed", [False, True])
async def test_environment_fallback_and_disabled_cell_timer(managed, monkeypatch):
    backend = RecordingBackend()
    monkeypatch.setenv("SQLSABER_NOTEBOOK_BACKEND", "docker")
    monkeypatch.setenv("SQLSABER_NOTEBOOK_IMAGE", "environment-image")
    monkeypatch.setattr(execution, "DockerNotebookBackend", lambda: backend)
    await run_analysis(
        managed, NotebookConfig(cell_seconds=None), [WorkspaceFile("a", b"x")]
    )
    assert backend.images == ["environment-image"]
    assert backend.executions[0].call_args.kwargs == {
        "cell_timeout": None,
        "command_timeout": None,
    }


async def test_explicit_api_selectors_override_config():
    configured = RecordingBackend()
    explicit = RecordingBackend()
    await analyze(
        "Analyze",
        Workspace(()),
        model=TestModel(call_tools=[]),
        model_provider="test",
        config=NotebookConfig(backend=configured, image="configured-image"),
        backend=explicit,
        image="explicit-image",
    )
    assert not configured.environments
    assert explicit.images == ["explicit-image"]


def test_default_config_preserves_backend_defaults():
    assert NotebookConfig().execution_limits() == ExecutionLimits()


@pytest.mark.parametrize("selector", ["backend", "image"])
def test_empty_explicit_selectors_do_not_fall_back_to_environment(
    selector, monkeypatch
):
    monkeypatch.setenv("SQLSABER_NOTEBOOK_BACKEND", "docker")
    monkeypatch.setenv("SQLSABER_NOTEBOOK_IMAGE", "environment-image")
    resolve = (
        execution.resolve_notebook_backend
        if selector == "backend"
        else execution.resolve_notebook_image
    )
    with pytest.raises(execution.NotebookExecutionError):
        resolve("")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cell_seconds": 0},
        {"command_seconds": -1},
        {"memory_mb": True},
        {"cpu_cores": float("nan")},
        {"cpu_cores": float("inf")},
        {"max_artifacts": 1.5},
        {"image": " "},
        {"backend": ""},
    ],
)
def test_invalid_config_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        NotebookConfig(**kwargs)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_invalid_workspace_budget_fails_at_construction(value):
    with pytest.raises(ValueError):
        WorkspaceLimits(max_total_bytes=value)


async def test_workspace_listing_pages_inputs_and_generated_files_independently():
    backend = FakeNotebookBackend(
        lambda notebook, inputs, run: FakeRunResult(
            notebook, {f"out-{i:03}.txt": b"x" for i in range(103)}
        )
    )
    config = NotebookConfig(workspace=WorkspaceLimits(max_files=60), max_artifacts=103)
    session = NotebookSession(
        workspace=Workspace(tuple(NotebookInput(f"in-{i}", b"x") for i in range(60))),
        backend=backend,
        image="unused",
        workspace_limits=config.workspace,
        execution_limits=config.execution_limits(),
    )
    try:
        await session.run_notebook()
        first = json.loads(await session.list_workspace())
        second = json.loads(await session.list_workspace(offset=first["next_offset"]))
        last = json.loads(await session.list_workspace(offset=second["next_offset"]))
        assert [len(page["inputs"]) for page in (first, second, last)] == [50, 10, 0]
        assert [len(page["generated"]) for page in (first, second, last)] == [50, 50, 3]
        assert second["inputs"][0]["path"] == "../inputs/in-50"
        assert last["generated"][-1]["path"] == "out-102.txt"
        assert last["next_offset"] is None
        assert first["input_count"] == 60
        assert first["configured_workspace_limits"]["max_files"] == 60
        assert first["configured_execution_limits"]["cell_seconds"] == 600
    finally:
        await session.close()
