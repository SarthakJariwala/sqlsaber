"""Tests for sandbox tools."""

import json
import sys
from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai import RunContext
import pytest

pytest.importorskip("sqlsaber_sandbox")

from sqlsaber_sandbox.capability import Sandbox, capability
from sqlsaber_sandbox.tools import MAX_CODE_CHARS, MAX_REQUIREMENTS, RunPythonTool

PROVIDER_ENV_VARS = (
    "E2B_API_KEY",
    "DAYTONA_API_KEY",
    "SPRITES_TOKEN",
    "HOPX_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_CONFIG_PATH",
    "CLOUDFLARE_SANDBOX_BASE_URL",
    "CLOUDFLARE_API_TOKEN",
)


def _build_result(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    duration_ms: int | None = 5,
    truncated: bool = False,
    timed_out: bool = False,
    success: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        truncated=truncated,
        timed_out=timed_out,
        success=success,
    )


class DummySandbox:
    """Minimal sandbox stand-in for unit tests."""

    def __init__(self, results: list[SimpleNamespace]):
        self._results = list(results)
        self.executed: list[str] = []

    async def __aenter__(self) -> "DummySandbox":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def execute(self, command: str) -> SimpleNamespace:
        self.executed.append(command)
        return self._results.pop(0)


class DummySandboxFactory:
    def __init__(self, sandbox: DummySandbox):
        self._sandbox = sandbox
        self.created_with: list[int] = []

    def create(self, timeout: int) -> DummySandbox:
        self.created_with.append(timeout)
        return self._sandbox


def _patch_sandbox(monkeypatch: pytest.MonkeyPatch, sandbox: DummySandbox) -> None:
    factory = DummySandboxFactory(sandbox)
    module = SimpleNamespace(Sandbox=factory)
    monkeypatch.setitem(sys.modules, "sandboxes", module)


def _parse_result(payload: str) -> dict:
    return json.loads(payload)


def _make_ctx() -> RunContext[None]:
    return cast(RunContext[None], SimpleNamespace(messages=[]))


def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


def _isolate_modal_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)


def test_sandbox_capability_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_env(monkeypatch)
    _isolate_modal_home(monkeypatch, tmp_path)

    assert capability(cast(Any, None)) == ()


def test_sandbox_capability_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("E2B_API_KEY", "test-key")

    assert isinstance(capability(cast(Any, None)), Sandbox)


def test_sandbox_capability_disabled_with_modal_id_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("MODAL_TOKEN_ID", "modal-id-only")
    _isolate_modal_home(monkeypatch, tmp_path)

    assert capability(cast(Any, None)) == ()


def test_sandbox_capability_enabled_with_modal_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_env(monkeypatch)
    _isolate_modal_home(monkeypatch, tmp_path)
    modal_config = tmp_path / ".modal.toml"
    modal_config.write_text('token_id = "test"\n', encoding="utf-8")

    assert isinstance(capability(cast(Any, None)), Sandbox)


def test_sandbox_capability_enabled_with_modal_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_env(monkeypatch)
    config_path = tmp_path / "custom-modal.toml"
    config_path.write_text('token_id = "test"\n', encoding="utf-8")
    monkeypatch.setenv("MODAL_CONFIG_PATH", str(config_path))

    assert isinstance(capability(cast(Any, None)), Sandbox)


@pytest.mark.asyncio
async def test_run_python_requires_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _clear_provider_env(monkeypatch)
    _isolate_modal_home(monkeypatch, tmp_path)
    tool = RunPythonTool()

    result = _parse_result(await tool.execute(_make_ctx(), code="print('hi')"))

    assert "error" in result


@pytest.mark.asyncio
async def test_run_python_rejects_large_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    tool = RunPythonTool()

    result = _parse_result(
        await tool.execute(_make_ctx(), code="x" * (MAX_CODE_CHARS + 1))
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_run_python_rejects_many_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    tool = RunPythonTool()

    requirements = [f"pkg{i}" for i in range(MAX_REQUIREMENTS + 1)]
    result = _parse_result(
        await tool.execute(_make_ctx(), code="print('ok')", requirements=requirements)
    )

    assert "error" in result


@pytest.mark.asyncio
async def test_run_python_executes_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    sandbox = DummySandbox([_build_result(stdout="hello")])
    _patch_sandbox(monkeypatch, sandbox)

    tool = RunPythonTool()
    result = _parse_result(await tool.execute(_make_ctx(), code="print('hello')"))

    assert result["success"] is True
    assert result["stdout"] == "hello"
    assert sandbox.executed


@pytest.mark.asyncio
async def test_run_python_installs_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    sandbox = DummySandbox(
        [
            _build_result(stdout="installed"),
            _build_result(stdout="done"),
        ]
    )
    _patch_sandbox(monkeypatch, sandbox)

    tool = RunPythonTool()
    result = _parse_result(
        await tool.execute(_make_ctx(), code="print('done')", requirements=["requests"])
    )

    assert result["success"] is True
    assert len(sandbox.executed) == 2
    assert "pip install" in sandbox.executed[0]


@pytest.mark.asyncio
async def test_run_python_handles_install_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    sandbox = DummySandbox(
        [_build_result(exit_code=1, stdout="", stderr="boom", success=False)]
    )
    _patch_sandbox(monkeypatch, sandbox)

    tool = RunPythonTool()
    result = _parse_result(
        await tool.execute(_make_ctx(), code="print('done')", requirements=["requests"])
    )

    assert result["error"] == "Failed to install requirements."
    assert result["exit_code"] == 1
