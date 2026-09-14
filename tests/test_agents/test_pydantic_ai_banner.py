from __future__ import annotations

import importlib.util
import os
import pty
import select
import sys
import time

import pytest

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

    pid, fd = pty.fork()
    if pid == 0:
        os.execve(sys.executable, [sys.executable, "-c", code], env)

    chunks: list[bytes] = []
    deadline = time.time() + 30
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 0.2)
        if fd in ready:
            try:
                data = os.read(fd, 8192)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        else:
            try:
                waited, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if waited:
                break
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    return b"".join(chunks).decode("utf-8", "replace")


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
