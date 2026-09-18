from __future__ import annotations

import asyncio
import json
import shlex
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sqlsaber_notebook.execution import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookExecutionTimeout,
    NotebookImageError,
    NotebookInfrastructureError,
    NotebookInput,
    NotebookLimitExceeded,
)
from sqlsaber_notebook.execution.e2b import E2BNotebookBackend, E2BNotebookEnvironment
from sqlsaber_notebook.settings import build_notebook_config, settings

from _notebooks import contract_notebook


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import e2b

    sandbox = SimpleNamespace(
        commands=SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(stdout=""))
        ),
        files=SimpleNamespace(write=AsyncMock(), read=AsyncMock()),
        kill=AsyncMock(),
    )
    create = AsyncMock(return_value=sandbox)
    build = AsyncMock(return_value=SimpleNamespace(template_id="built-template"))
    template = MagicMock()
    template.return_value.from_image.return_value.set_user.return_value = "definition"
    template.build = build
    template.exists = AsyncMock(return_value=False)
    monkeypatch.setattr(e2b, "AsyncTemplate", template)
    monkeypatch.setattr(e2b.AsyncSandbox, "create", create)
    return SimpleNamespace(
        sandbox=sandbox, create=create, build=build, template=template
    )


async def test_open_preserves_image_resources_credentials_and_isolation(sdk) -> None:
    backend = E2BNotebookBackend(api_key="test-key")
    env = await backend.open(
        [NotebookInput("quoted ' input.json", b"123")],
        image="registry/custom@sha256:abc",
        limits=ExecutionLimits(cpu_cores=2.5, memory_mb=4096),
    )
    sdk.template.return_value.from_image.assert_called_once_with(
        "registry/custom@sha256:abc"
    )
    assert sdk.build.call_args.kwargs["cpu_count"] == 3
    assert sdk.build.call_args.kwargs["memory_mb"] == 4096
    assert sdk.build.call_args.kwargs["api_key"] == "test-key"
    sdk.template.exists.assert_awaited_once_with(
        sdk.build.call_args.kwargs["name"], api_key="test-key"
    )
    sdk.create.assert_awaited_once_with(
        template="built-template",
        timeout=3600,
        allow_internet_access=False,
        api_key="test-key",
    )
    assert sdk.sandbox.files.write.call_args_list[0].args == (
        "/root/sqlsaber-notebook/inputs/quoted ' input.json",
        b"123",
    )
    assert sdk.sandbox.files.write.call_args_list[0].kwargs["user"] == "root"
    await env.close()
    await env.close()
    sdk.sandbox.kill.assert_awaited_once()


async def test_input_validation_precedes_remote_resources(sdk) -> None:
    with pytest.raises(NotebookLimitExceeded):
        await E2BNotebookBackend().open(
            [NotebookInput("../escape", b"x")], image="image", limits=ExecutionLimits()
        )
    sdk.build.assert_not_awaited()
    sdk.template.exists.assert_not_awaited()
    sdk.create.assert_not_awaited()


async def test_existing_template_skips_build_with_bound_credentials(sdk) -> None:
    sdk.template.exists.return_value = True
    env = await E2BNotebookBackend(api_key="customer-key").open(
        [], image="image", limits=ExecutionLimits()
    )
    reference = sdk.template.exists.call_args.args[0]
    assert reference.startswith("sqlsaber-notebook-")
    sdk.template.exists.assert_awaited_once_with(reference, api_key="customer-key")
    sdk.build.assert_not_awaited()
    sdk.template.assert_not_called()
    assert sdk.create.call_args.kwargs["template"] == reference
    assert sdk.create.call_args.kwargs["api_key"] == "customer-key"
    await env.close()


@pytest.mark.parametrize("failure", [RuntimeError("lookup failed"), TimeoutError()])
async def test_lookup_failure_does_not_build_or_create(sdk, failure) -> None:
    sdk.template.exists.side_effect = failure
    error = (
        NotebookExecutionTimeout
        if isinstance(failure, TimeoutError)
        else NotebookImageError
    )
    with pytest.raises(error):
        await E2BNotebookBackend().open([], image="image", limits=ExecutionLimits())
    sdk.build.assert_not_awaited()
    sdk.create.assert_not_awaited()


async def test_template_reference_tracks_image_and_effective_resources(sdk) -> None:
    sdk.template.exists.return_value = True
    references = []
    for image, cpu, memory in [
        ("image-a", 2, 4096),
        ("image-a", 2, 4096),
        ("image-b", 2, 4096),
        ("image-a", 3, 4096),
        ("image-a", 2, 8192),
    ]:
        env = await E2BNotebookBackend().open(
            [], image=image, limits=ExecutionLimits(cpu_cores=cpu, memory_mb=memory)
        )
        references.append(sdk.template.exists.call_args.args[0])
        await env.close()
    assert references[0] == references[1]
    assert len(set(references)) == 4
    sdk.build.assert_not_awaited()


async def test_open_upload_failure_cleans_up(sdk) -> None:
    sdk.sandbox.files.write.side_effect = RuntimeError("upload failed")
    with pytest.raises(NotebookInfrastructureError) as error:
        await E2BNotebookBackend().open(
            [NotebookInput("data", b"x")], image="image", limits=ExecutionLimits()
        )
    assert error.value.phase == "input-upload"
    sdk.sandbox.kill.assert_awaited_once()


async def test_template_failure_is_not_a_fallback(sdk) -> None:
    sdk.build.side_effect = RuntimeError("build failed")
    with pytest.raises(NotebookImageError):
        await E2BNotebookBackend().open([], image="bad-image", limits=ExecutionLimits())
    sdk.create.assert_not_awaited()


def successful_run(sdk, files=None) -> bytes:
    notebook = contract_notebook()
    sdk.sandbox.commands.run.side_effect = [
        SimpleNamespace(stdout=""),
        SimpleNamespace(stdout=""),
        SimpleNamespace(
            stdout=json.dumps(
                {
                    "code": 0,
                    "stdout": "done",
                    "stderr": "",
                    "files": {"notebook.ipynb": len(notebook), **(files or {})},
                }
            )
        ),
    ]
    sdk.sandbox.files.read.return_value = bytearray(notebook)
    return notebook


@pytest.mark.parametrize(
    "configured,requested,expected", [(40, 70, 40), (70, 40, 40), (None, None, 0)]
)
async def test_execute_uses_effective_timeout_and_binary_results(
    sdk, configured, requested, expected
) -> None:
    notebook = successful_run(sdk, {"summary.csv": 7})
    env = E2BNotebookEnvironment(
        sdk.sandbox,
        ExecutionLimits(command_seconds=configured, cell_seconds=configured),
    )
    result = await env.execute(
        notebook, cell_timeout=requested, command_timeout=requested
    )
    assert result.notebook == notebook
    assert [a.path for a in result.artifacts] == ["summary.csv"]
    call = sdk.sandbox.commands.run.call_args
    assert call.kwargs["timeout"] == expected
    assert json.loads(shlex.split(call.args[0])[-1])["cell"] == (expected or -1)
    assert await env.list_workspace() == result.artifacts
    sdk.sandbox.files.read.return_value = bytearray(b"a,b\n1,2")
    assert await env.read_artifact(result.artifacts[0]) == b"a,b\n1,2"
    sdk.sandbox.files.read.return_value = b"changed size"
    with pytest.raises(NotebookInfrastructureError, match="changed"):
        await env.read_artifact(result.artifacts[0])
    with pytest.raises(NotebookInfrastructureError, match="Unknown"):
        await env.read_artifact(ArtifactInfo("../secret", 2))
    await env.close()


@pytest.mark.parametrize("failure", [TimeoutError(), asyncio.CancelledError()])
async def test_interrupted_execution_terminates_and_invalidates(sdk, failure) -> None:
    env = E2BNotebookEnvironment(sdk.sandbox, ExecutionLimits())
    sdk.sandbox.commands.run.side_effect = failure
    expected = (
        asyncio.CancelledError
        if isinstance(failure, asyncio.CancelledError)
        else NotebookExecutionTimeout
    )
    with pytest.raises(expected):
        await env.execute(contract_notebook(), cell_timeout=10, command_timeout=20)
    sdk.sandbox.kill.assert_awaited_once()
    with pytest.raises(NotebookInfrastructureError, match="closed"):
        await env.list_workspace()


@pytest.mark.parametrize("files", [{"../escape": 1}, {"too-large": 11}])
async def test_invalid_inventory_never_published(sdk, files) -> None:
    notebook = successful_run(sdk, files)
    env = E2BNotebookEnvironment(sdk.sandbox, ExecutionLimits(max_artifact_bytes=10))
    with pytest.raises((NotebookLimitExceeded, NotebookInfrastructureError)):
        await env.execute(notebook, cell_timeout=10, command_timeout=20)
    sdk.sandbox.files.read.assert_not_awaited()
    sdk.sandbox.kill.assert_awaited_once()


async def test_failed_cleanup_is_reported_and_retryable(sdk) -> None:
    env = E2BNotebookEnvironment(sdk.sandbox, ExecutionLimits())
    sdk.sandbox.kill.side_effect = [RuntimeError("unreachable"), True]
    with pytest.raises(NotebookInfrastructureError) as error:
        await env.close()
    assert error.value.phase == "cleanup"
    await env.close()
    assert sdk.sandbox.kill.await_count == 2


def test_settings_bind_key_without_mutating_environment(monkeypatch) -> None:
    monkeypatch.setenv("E2B_API_KEY", "native-key")
    config = build_notebook_config({"backend": "e2b"}, {"e2b_api_key": "saved-key"})
    assert isinstance(config.backend, E2BNotebookBackend)
    assert config.backend._api_key == "saved-key"
    assert build_notebook_config({"backend": "e2b"}, {}).backend == "e2b"
    field = settings.field("e2b_api_key")
    assert field.credential == "e2b.api_key"
    assert field.env == "E2B_API_KEY"
    assert field.kind == "secret"
    assert field.active({"backend": "e2b"})
    assert not field.active({"backend": "docker"})
    assert "uploaded to E2B" in settings.notice({"backend": "e2b"})
    import os

    assert os.environ["E2B_API_KEY"] == "native-key"
