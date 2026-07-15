from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from sqlsaber_notebook.execution import ExecutionLimits, NotebookBackendUnavailable
from sqlsaber_notebook.execution.base import NotebookExecutionTimeout
from sqlsaber_notebook.execution.docker import (
    DockerNotebookBackend,
    DockerNotebookEnvironment,
    _bounded_timeout,
    _run_process,
    _scan_artifact_sizes,
)


def test_notebook_command_has_no_default_timeout() -> None:
    assert _bounded_timeout(None, None) is None
    assert _bounded_timeout(30, None) == 30
    assert _bounded_timeout(30, 60) == 30


def test_docker_availability_does_not_fall_back() -> None:
    backend = DockerNotebookBackend(executable="/definitely/missing/docker")
    assert backend.available() is False


async def test_docker_open_reports_missing_cli() -> None:
    backend = DockerNotebookBackend(executable="/definitely/missing/docker")
    with pytest.raises(NotebookBackendUnavailable, match="Docker CLI is not installed"):
        await backend.open([], image="image", limits=ExecutionLimits())


def test_docker_argv_is_hardened_and_uses_direct_arguments(tmp_path: Path) -> None:
    environment = DockerNotebookEnvironment(
        executable="docker",
        root=tmp_path,
        image="registry/image@sha256:digest",
        limits=ExecutionLimits(),
    )
    argv = environment._docker_argv("known-name", 99)

    assert argv[:3] == ("docker", "run", "--rm")
    assert ("--network", "none") == argv[
        argv.index("--network") : argv.index("--network") + 2
    ]
    assert ("--cap-drop", "ALL") == argv[
        argv.index("--cap-drop") : argv.index("--cap-drop") + 2
    ]
    assert "no-new-privileges" in argv
    assert "--memory" in argv and "8192m" in argv
    assert "--cpus" in argv and "4.0" in argv
    assert "--workdir" in argv and "/work/run" in argv
    assert "--allow-errors" in argv
    assert "--ExecutePreprocessor.timeout=99" in argv
    assert all("shell" not in argument for argument in argv)
    if hasattr(os, "getuid"):
        assert "--user" in argv
        assert f"{os.getuid()}:{os.getgid()}" in argv
        assert "HOME=/tmp" in argv


async def test_subprocess_timeout_runs_cleanup() -> None:
    cleaned = asyncio.Event()

    async def cleanup() -> None:
        cleaned.set()

    with pytest.raises(NotebookExecutionTimeout, match="timed out"):
        await _run_process(
            ("sleep", "5"),
            timeout=0.01,
            backend="docker",
            phase="test",
            log_limit=100,
            cleanup=cleanup,
        )
    assert cleaned.is_set()


def test_artifact_scan_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("target")
    (tmp_path / "link.txt").symlink_to(tmp_path / "target.txt")
    with pytest.raises(ValueError, match="non-regular artifact"):
        _scan_artifact_sizes(tmp_path)
