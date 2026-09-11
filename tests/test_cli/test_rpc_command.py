from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from sqlsaber.cli.commands import app
from sqlsaber.cli.rpc import rpc


def test_root_help_lists_rpc(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        app(["--help"])
    assert exc_info.value.code == 0
    assert "rpc" in capsys.readouterr().out


def test_rpc_help_includes_examples_and_is_fast(capsys) -> None:
    started = time.perf_counter()
    with pytest.raises(SystemExit) as exc_info:
        app(["rpc", "--help"])
    elapsed = time.perf_counter() - started
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "Example" in output
    assert "--no-thread" in output
    assert elapsed < 0.5


def test_thread_and_no_thread_are_exclusive(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        rpc(thread="abc", no_thread=True)
    assert exc_info.value.code == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_rpc_startup_without_database_is_one_json_line(monkeypatch) -> None:
    writes: list[bytes] = []
    monkeypatch.setattr("sqlsaber.cli.rpc.claim_stdout", lambda: writes.append)
    monkeypatch.setattr(
        "sqlsaber.cli.onboarding.needs_onboarding", lambda _database=None: True
    )
    with pytest.raises(SystemExit) as exc_info:
        rpc()
    assert exc_info.value.code == 1
    assert len(writes) == 1
    record = json.loads(writes[0])
    assert record["type"] == "response"
    assert record["command"] == "startup"
    assert record["success"] is False
    assert "sqlsaber db add" in record["error"]


def test_rpc_passes_persist_thread(monkeypatch) -> None:
    from sqlsaber.cli.commands import CLIError

    captured: dict[str, object] = {}

    async def create(**kwargs):
        captured.update(kwargs)
        raise CLIError("stopped")

    writes: list[bytes] = []
    monkeypatch.setattr("sqlsaber.cli.rpc.claim_stdout", lambda: writes.append)
    monkeypatch.setattr(
        "sqlsaber.cli.onboarding.needs_onboarding", lambda _database=None: False
    )
    monkeypatch.setattr("sqlsaber.cli.commands._create_cli_saber", create)
    with pytest.raises(SystemExit) as exc_info:
        rpc(no_thread=True, database=["analytics"])
    assert exc_info.value.code == 1
    assert captured["persist_thread"] is False
    assert json.loads(writes[0])["command"] == "startup"

    captured.clear()
    writes.clear()
    with pytest.raises(SystemExit):
        rpc(database=["analytics"])
    assert captured["persist_thread"] is True


def test_subprocess_rpc_empty_config_writes_startup_json(tmp_path) -> None:
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "sqlsaber", "rpc"],
        check=False,
        capture_output=True,
        input=b"",
        env=env,
        timeout=30,
    )
    assert result.returncode == 1
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["command"] == "startup"
    assert record["success"] is False
    assert result.stdout.decode("utf-8").endswith("\n")
