"""Logging must stay off the terminal unless the user opts into console output."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_isolated(
    code: str, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    if not env or "SQLSABER_DEBUG" not in env:
        merged.pop("SQLSABER_DEBUG", None)
    if not env or "SQLSABER_LOG_TO_STDERR" not in env:
        merged.pop("SQLSABER_LOG_TO_STDERR", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )


def test_unconfigured_logger_does_not_print_to_stdout_or_stderr() -> None:
    result = _run_isolated(
        """
from sqlsaber.config.logging import get_logger, is_configured

assert is_configured() is False
get_logger("probe").info("probe.must_not_print", count=1)
print("OK")
"""
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe.must_not_print" not in result.stdout
    assert "probe.must_not_print" not in result.stderr
    assert "OK" in result.stdout


def test_setup_logging_writes_json_to_file_not_the_terminal(tmp_path: Path) -> None:
    log_file = tmp_path / "sqlsaber.log"
    result = _run_isolated(
        """
from sqlsaber.config.logging import get_logger, setup_logging

setup_logging(force=True)
get_logger("probe").info("probe.file_only", count=2)
print("OK")
""",
        env={"SQLSABER_LOG_FILE": str(log_file)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe.file_only" not in result.stdout
    assert "probe.file_only" not in result.stderr
    assert "OK" in result.stdout
    logged = log_file.read_text(encoding="utf-8")
    assert '"event": "probe.file_only"' in logged
    assert '"count": 2' in logged


def test_debug_flag_prints_console_logs_to_stderr(tmp_path: Path) -> None:
    log_file = tmp_path / "sqlsaber.log"
    result = _run_isolated(
        """
from sqlsaber.config.logging import get_logger, setup_logging

setup_logging(force=True)
get_logger("probe").info("probe.debug_console")
print("OK")
""",
        env={
            "SQLSABER_LOG_FILE": str(log_file),
            "SQLSABER_DEBUG": "1",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "probe.debug_console" not in result.stdout
    assert "probe.debug_console" in result.stderr
    assert "OK" in result.stdout
