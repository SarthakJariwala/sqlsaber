"""Management commands must not print structured logs to the terminal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _saber_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = tmp_path / "home"
    home.mkdir()
    env["HOME"] = str(home)
    env["XDG_CONFIG_HOME"] = str(tmp_path / "config")
    env["XDG_DATA_HOME"] = str(tmp_path / "data")
    env["XDG_STATE_HOME"] = str(tmp_path / "state")
    env["SQLSABER_LOG_FILE"] = str(tmp_path / "sqlsaber.log")
    env.pop("SQLSABER_DEBUG", None)
    env.pop("SQLSABER_LOG_TO_STDERR", None)
    return env


def _run_saber(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sqlsaber", *args],
        check=False,
        capture_output=True,
        text=True,
        env=_saber_env(tmp_path),
    )


@pytest.mark.parametrize(
    ("args", "event", "visible"),
    [
        (("db", "list"), "db.list.start", "No database connections configured"),
        (("auth", "status"), "auth.status.start", "Authentication Status"),
        (("threads", "list"), "threads.cli.list.start", "No threads found"),
    ],
)
def test_management_commands_keep_structlog_off_the_terminal(
    tmp_path: Path, args: tuple[str, ...], event: str, visible: str
) -> None:
    result = _run_saber(tmp_path, *args)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert visible in output
    assert event not in result.stdout
    assert event not in result.stderr
    assert "[info" not in result.stdout
    assert "[info" not in result.stderr
    logged = (tmp_path / "sqlsaber.log").read_text(encoding="utf-8")
    assert f'"event": "{event}"' in logged
