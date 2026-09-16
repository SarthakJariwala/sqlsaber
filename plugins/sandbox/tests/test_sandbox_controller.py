"""Execute the actual controller against a local IPython kernel, without cloud."""

import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sqlsaber_sandbox import _controller
from sqlsaber_sandbox._controller import Controller, bridge
from sqlsaber_sandbox.backends.base import CommandResult
from sqlsaber_sandbox.config import SandboxConfig
from sqlsaber_sandbox.execution import KernelExecution, SandboxError


@pytest.mark.skipif(sys.platform == "win32", reason="Guest bridge uses Unix sockets")
@pytest.mark.parametrize("stale_socket", [False, True])
async def test_bridge_startup_ping_is_quiet_until_socket_is_ready(stale_socket):
    with tempfile.TemporaryDirectory(prefix="ss-bridge-") as directory:
        root = Path(directory)
        path = root / "controller.sock"
        if stale_socket:
            with socket.socket(socket.AF_UNIX) as sock:
                sock.bind(str(path))

        async def call(operation):
            encoded = base64.b64encode(json.dumps({"operation": operation}).encode())
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                _controller.__file__,
                "call",
                str(root),
                encoded.decode(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), 10)
                return process.returncode, stdout, stderr
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

        assert await call("ping") == (1, b"", b"")
        # Only readiness probes are quiet. Losing the socket during execution
        # must still expose the underlying connection failure.
        code, stdout, stderr = await call("execute")
        assert code == 1 and not stdout
        assert (
            b"ConnectionRefusedError" if stale_socket else b"FileNotFoundError"
        ) in stderr

        controller = Controller(root, asdict(SandboxConfig()))
        async with await asyncio.start_unix_server(controller.connection, path=path):
            code, stdout, stderr = await call("ping")
            assert code == 0 and not stderr
            assert json.loads(stdout) == {
                "epoch": controller.epoch,
                "result": {"epoch": controller.epoch, "lost": False},
            }


async def test_bridge_does_not_hide_unexpected_connection_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(
        asyncio,
        "open_unix_connection",
        AsyncMock(side_effect=PermissionError("socket access denied")),
        raising=False,
    )
    encoded = base64.b64encode(b'{"operation":"ping"}').decode()
    with pytest.raises(PermissionError, match="socket access denied"):
        await bridge(tmp_path, encoded)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Runs the sandbox's POSIX shell and executable scripts on the host",
)
@pytest.mark.parametrize(
    "packages, install_exit",
    [("compatible", 0), ("missing", 0), ("incompatible", 0), ("missing", 42)],
)
async def test_kernel_bootstrap_reuses_image_packages_and_interpreter(
    tmp_path, monkeypatch, packages, install_exit
):
    python = tmp_path / "runtime bin" / "python"
    python.parent.mkdir()
    installs = tmp_path / "installs.json"
    python.write_text(
        f"""#!{sys.executable}
import json
import sys
from pathlib import Path
if sys.argv[1:3] == ['-m', 'pip']:
    Path({str(installs)!r}).write_text(json.dumps(sys.argv[1:]))
    sys.exit({install_exit})
if {packages == "missing"}:
    raise ModuleNotFoundError('ipykernel')
if {packages == "incompatible"}:
    import ipykernel, jupyter_client, matplotlib
    import importlib.metadata
    original = importlib.metadata.version
    importlib.metadata.version = lambda name: '9.0.0' if name == 'jupyter-client' else original(name)
exec(sys.argv[2])
"""
    )
    python.chmod(0o755)
    monkeypatch.setenv("PATH", str(python.parent))
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    controller_calls = []

    async def execute(command, *, timeout):
        if command.startswith("mkdir"):
            return CommandResult("", "", 0)
        if command.startswith("if [ -x /opt/conda/bin/python ]"):
            return CommandResult(str(python) + "\n", "", 0)
        if "controller.py call" in command:
            controller_calls.append(shlex.split(command))
            return CommandResult('{"epoch":"one","result":{}}', "", 0)
        # Execute the real bootstrap shell logic. The Python wrapper records pip
        # calls instead of accessing a package index or changing this environment.
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return CommandResult(stdout.decode(), stderr.decode(), process.returncode)

    backend = SimpleNamespace(
        open=AsyncMock(),
        execute=execute,
        upload=AsyncMock(),
        start_controller=AsyncMock(),
        close=AsyncMock(),
    )
    execution = KernelExecution(SandboxConfig(), backend)
    if install_exit:
        with pytest.raises(SandboxError, match="Sandbox command failed"):
            await execution.open()
        backend.start_controller.assert_not_awaited()
        backend.close.assert_awaited_once()
    else:
        await execution.open()
        assert installs.exists() is (packages != "compatible")
        assert backend.start_controller.call_args.args[0][0] == str(python)
        assert controller_calls[0][0] == str(python)
        await execution.close()
    if packages != "compatible":
        assert json.loads(installs.read_text()) == [
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-input",
            "jupyter-client>=8,<9",
            "ipykernel>=6,<8",
            "matplotlib>=3,<4",
        ]


@pytest.fixture
async def controller():
    if sys.platform == "win32":
        pytest.skip("Runs the Linux sandbox's IPC kernel on the host")
    # Jupyter IPC paths must fit the Unix socket path length, regardless of
    # pytest's checkout path or parameterized test name.
    with tempfile.TemporaryDirectory(prefix="ss-test-") as directory:
        root = Path(directory)
        (root / "run").mkdir()
        instance = Controller(
            root, asdict(SandboxConfig(cell_seconds=2, max_output_chars=80))
        )
        try:
            await instance.start()
            yield instance
        finally:
            if instance.active is not None:
                instance.active.cancel()
                await asyncio.gather(instance.active, return_exceptions=True)
            if hasattr(instance, "client"):
                instance.client.stop_channels()
            if hasattr(instance, "manager"):
                await instance.manager.shutdown_kernel(now=True)


async def run_cell(controller, code, execution_id):
    await controller.dispatch(
        {"operation": "execute", "code": code, "execution_id": execution_id}
    )
    async with asyncio.timeout(10):
        while True:
            result = await controller.dispatch(
                {"operation": "status", "execution_id": execution_id}
            )
            if result["status"] != "running":
                assert "outputs" not in result
                return json.loads(
                    (controller.root / "results" / f"{execution_id}.json").read_bytes()
                )
            await asyncio.sleep(0.01)


async def test_incremental_execution_error_and_output_drain(controller):
    assert (await run_cell(controller, "values=[3,8]", "a"))["status"] == "ok"
    failed = await run_cell(
        controller, "values.append(13)\nraise ValueError('expected')", "b"
    )
    assert failed["status"] == "error"
    assert failed["outputs"][0]["ename"] == "ValueError"
    result = await run_cell(
        controller, "import asyncio\nawait asyncio.sleep(.001)\nsum(values)", "c"
    )
    assert result["outputs"][-1]["data"]["text/plain"] == "24"
    huge = await run_cell(controller, "print('x'*100000)", "d")
    assert huge["truncated"]
    assert sum(len(output.get("text", "")) for output in huge["outputs"]) == 80
    assert (await run_cell(controller, "assert len(values)==3", "e"))["status"] == "ok"
    # Same ID never applies the mutation a second time.
    await controller.dispatch(
        {
            "operation": "execute",
            "code": "values.append(13)\nraise ValueError('expected')",
            "execution_id": "b",
        }
    )
    assert (await run_cell(controller, "assert values == [3,8,13]", "f"))[
        "status"
    ] == "ok"
    with pytest.raises(ValueError, match="different code"):
        await controller.dispatch(
            {"operation": "execute", "code": "values=[]", "execution_id": "b"}
        )


async def test_timeout_interrupt_preserves_partial_namespace(controller):
    controller.config["cell_seconds"] = 0.2
    result = await run_cell(controller, "value=41\nwhile True: pass", "loop")
    assert result["status"] == "interrupted"
    assert not controller.lost
    controller.config["cell_seconds"] = 2
    assert (await run_cell(controller, "value+1", "after"))["outputs"][-1]["data"][
        "text/plain"
    ] == "42"


async def test_kernel_death_is_detected_without_cell_timeout(controller):
    controller.config["cell_seconds"] = None
    with pytest.raises(ValueError, match="lost"):
        await run_cell(controller, "import os; os._exit(7)", "death")
    assert controller.lost


async def test_artifacts_are_snapshotted_and_symlinks_rejected(controller):
    path = controller.root / "run" / "weights.bin"
    data = b"\x00\xff\x80original"
    path.write_bytes(data)
    snapshot = controller.export()
    path.write_bytes(b"changed")
    descriptor = snapshot["files"][0]
    assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
    assert (
        controller.root / "exports" / snapshot["export_id"] / "weights.bin"
    ).read_bytes() == data
    (controller.root / "run" / "link").symlink_to(controller.root / "kernel.json")
    with pytest.raises(ValueError, match="regular"):
        controller.export()


async def test_cleanup_failure_can_be_retried():
    from unittest.mock import AsyncMock

    execution = KernelExecution(SandboxConfig())
    sandbox = SimpleNamespace(
        close=AsyncMock(side_effect=[RuntimeError("temporary"), None])
    )
    execution.backend = sandbox
    with pytest.raises(RuntimeError):
        await execution.close()
    assert execution.backend is sandbox
    await execution.close()
    assert sandbox.close.await_count == 2


async def test_unknown_dispatch_is_not_replayed():
    from unittest.mock import AsyncMock

    execution = KernelExecution(SandboxConfig())
    execution.request = AsyncMock(
        side_effect=[SandboxError("lost acknowledgement"), {"status": "unknown"}]
    )
    with pytest.raises(SandboxError, match="not replayed"):
        await execution.execute("side_effect()", "a")
    assert [call.args[0] for call in execution.request.call_args_list] == [
        "execute",
        "status",
    ]
    assert execution.lost


async def test_cancel_unknown_outcome_closes_instead_of_continuing():
    from unittest.mock import AsyncMock

    execution = KernelExecution(SandboxConfig())
    execution.request = AsyncMock(side_effect=[{}, {"status": "unknown"}])
    execution.close = AsyncMock()
    await execution._interrupt_and_settle("a")
    assert execution.lost
    execution.close.assert_awaited_once()


async def test_many_displays_are_bounded_and_next_cell_is_drained(controller):
    controller.config["cell_seconds"] = 10
    result = await run_cell(
        controller,
        "from IPython.display import display\nfor i in range(300): display('x')",
        "flood",
    )
    assert result["truncated"]
    assert len(result["outputs"]) <= 80
    assert (await run_cell(controller, "6*7", "next"))["outputs"][-1]["data"][
        "text/plain"
    ] == "42"


def test_image_retention_budget_is_cumulative(tmp_path):
    import base64

    controller = Controller(
        tmp_path, asdict(SandboxConfig(max_image_bytes=9, max_history_image_bytes=12))
    )
    records = [{"outputs": [], "chars": 0, "truncated": False} for _ in range(2)]
    message = {
        "header": {"msg_type": "display_data"},
        "content": {"data": {"image/png": base64.b64encode(b"123456789").decode()}},
    }
    for record in records:
        controller.output(record, message)
    assert len(records[0]["outputs"]) == 1
    assert not records[1]["outputs"] and records[1]["truncated"]


async def test_cancelled_open_waits_for_allocated_handle_and_deletes_it():
    from unittest.mock import AsyncMock

    allocated, respond = asyncio.Event(), asyncio.Event()
    resource = None

    async def open_backend(config):
        nonlocal resource
        allocated.set()
        await respond.wait()
        resource = "allocated-provider-id"

    async def close_backend():
        nonlocal resource
        assert resource == "allocated-provider-id"
        resource = None

    backend = SimpleNamespace(
        open=AsyncMock(side_effect=open_backend),
        close=AsyncMock(side_effect=close_backend),
    )
    execution = KernelExecution(SandboxConfig(), backend)
    opening = asyncio.create_task(execution.open())
    await allocated.wait()
    opening.cancel()
    await asyncio.sleep(0)
    assert not opening.done()
    respond.set()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert resource is None
    backend.open.assert_awaited_once()
    backend.close.assert_awaited_once()
