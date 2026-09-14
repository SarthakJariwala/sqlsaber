from __future__ import annotations

import importlib.util
import os
import select
import subprocess
import sys
import time

import pytest

# Windows has no termios; skip this module instead of failing collection.
pty = pytest.importorskip("pty")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pydantic_ai._display") is None,
    reason="pydantic-ai first-run banner requires pydantic-ai 2.43+",
)

_BANNER_MARKERS = (
    "pydantic-ai v",
    "ai-setup.md",
    "Logfire",
    "OpenTelemetry",
)


def _run_on_pty(code: str) -> str:
    """Run ``code`` in a child interpreter whose stderr is a TTY.

    Args:
        code: Python source for the child.

    Returns:
        Combined PTY output as text.
    """
    env = os.environ.copy()
    env.pop("PYTEST_VERSION", None)
    env.pop("CI", None)
    env.pop("PYDANTIC_AI_NO_BANNER", None)
    env.pop("FORCE_COLOR", None)
    env["NO_COLOR"] = "1"
    env["TERM"] = "xterm-256color"

    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
    finally:
        os.close(slave_fd)

    chunks: list[bytes] = []
    deadline = time.time() + 30
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                proc.kill()
                break
            ready, _, _ = select.select([master_fd], [], [], min(0.2, remaining))
            if master_fd in ready:
                try:
                    data = os.read(master_fd, 8192)
                except OSError:
                    break
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                drain, _, _ = select.select([master_fd], [], [], 0.05)
                if master_fd in drain:
                    try:
                        data = os.read(master_fd, 8192)
                    except OSError:
                        pass
                    else:
                        if data:
                            chunks.append(data)
                break
        proc.wait(timeout=5)
    finally:
        os.close(master_fd)

    output = b"".join(chunks).decode("utf-8", "replace")
    assert proc.returncode == 0, output
    return output


def test_pydantic_ai_prints_first_run_banner_on_a_tty() -> None:
    output = _run_on_pty(
        """
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

print("LOADED", flush=True)
Agent(TestModel(), name="sqlsaber").run_sync("hello")
print("DONE", flush=True)
"""
    )
    assert "LOADED" in output
    assert "DONE" in output
    assert "pydantic-ai v" in output
    assert "ai-setup.md" in output


def test_loading_sqlsaber_agent_keeps_pydantic_banner_off_the_tty() -> None:
    output = _run_on_pty(
        """
from sqlsaber.agents.pydantic_ai_agent import SQLSaberAgent
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

_ = SQLSaberAgent
print("LOADED", flush=True)
Agent(TestModel(), name="sqlsaber").run_sync("hello")
print("DONE", flush=True)
"""
    )
    assert "LOADED" in output
    assert "DONE" in output
    for marker in _BANNER_MARKERS:
        assert marker not in output, output
