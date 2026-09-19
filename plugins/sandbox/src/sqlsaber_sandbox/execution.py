"""Provider-independent kernel lifetime, private bridge, and artifact verification."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from typing import Any
import uuid

from .backends import SandboxBackend, SandboxError, SessionLost, create_backend
from .config import SandboxConfig
from .result import ArtifactRef, CellResult, Workspace

_KERNEL_REQUIREMENTS = (
    "jupyter-client>=8,<9",
    "ipykernel>=6,<8",
    "matplotlib>=3,<4",
)


class KernelExecution:
    def __init__(self, config: SandboxConfig, backend: SandboxBackend | None = None):
        self.config = config
        self.root = f"/tmp/ss-{uuid.uuid4().hex[:16]}"
        self.backend = backend
        self.epoch: str | None = None
        self.lost = False
        self.settled: CellResult | None = None
        self._python = "python"
        self._opening: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None

    async def command(self, command: str, *, timeout: float | None = None) -> str:
        if self.backend is None:
            raise SessionLost("Sandbox is not open")
        timeout = timeout or self.config.transport_seconds
        try:
            async with asyncio.timeout(timeout):
                result = await self.backend.execute(command, timeout=timeout)
        except SandboxError:
            raise
        except Exception as exc:
            raise SandboxError("Sandbox command transport failed") from exc
        if result.exit_code != 0:
            raise SandboxError(
                f"Sandbox command failed: {result.stderr[: self.config.max_output_chars]}"
            )
        return result.stdout

    async def open(self) -> None:
        if self.lost:
            raise SessionLost("Sandbox is lost; close it and create a new session")
        self.backend = self.backend or create_backend(self.config.provider)
        try:
            async with asyncio.timeout(self.config.open_seconds):
                self._opening = asyncio.create_task(self.backend.open(self.config))
                # Cancellation cannot revoke an allocation already in flight.
                # Keep its response owned so cleanup can delete the result.
                await asyncio.shield(self._opening)
                await self.command(
                    f"mkdir -p {self.root}/inputs {self.root}/run; chmod 700 {self.root}"
                )
                # Jupyter images keep their scientific stack in Conda. Login
                # shells and provider execs need not preserve the image's PATH.
                self._python = (
                    await self.command(
                        "if [ -x /opt/conda/bin/python ]; then "
                        "echo /opt/conda/bin/python; else command -v python; fi"
                    )
                ).strip()
                python = shlex.quote(self._python)
                check = (
                    "import sys, ipykernel, jupyter_client, matplotlib; "
                    "from importlib.metadata import version; "
                    "from packaging.requirements import Requirement; "
                    f"requirements = map(Requirement, {_KERNEL_REQUIREMENTS!r}); "
                    "sys.exit(not all(version(req.name) in req.specifier for req in requirements))"
                )
                await self.command(
                    f"{python} -c {shlex.quote(check)} >/dev/null 2>&1 || "
                    f"{python} -m pip install --quiet --disable-pip-version-check --no-input "
                    + shlex.join(_KERNEL_REQUIREMENTS),
                    timeout=self.config.open_seconds,
                )
                await self.upload(
                    Path(__file__).with_name("_controller.py").read_bytes(),
                    f"{self.root}/controller.py",
                )
                await self.upload(
                    json.dumps(self.config.controller_config()).encode(),
                    f"{self.root}/config.json",
                )
                await self.backend.start_controller(
                    [
                        self._python,
                        f"{self.root}/controller.py",
                        "serve",
                        self.root,
                        f"{self.root}/config.json",
                    ]
                )
                while True:
                    try:
                        await self.request("ping")
                        break
                    except SandboxError:
                        await asyncio.sleep(0.3)
        except BaseException as primary:
            self.lost = True
            try:
                await self.close()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Sandbox opening and cleanup failed", [primary, cleanup]
                )
            raise

    async def upload(self, data: bytes, remote: str) -> None:
        assert self.backend is not None
        async with asyncio.timeout(self.config.open_seconds):
            await self.backend.upload(data, remote)

    async def download(self, remote: str, size: int) -> bytes:
        assert self.backend is not None
        async with asyncio.timeout(self.config.open_seconds):
            data = await self.backend.download(remote)
        if len(data) != size:
            raise SandboxError("Downloaded file size changed")
        return data

    async def request(self, operation: str, **arguments: Any) -> dict[str, Any]:
        encoded = base64.b64encode(
            json.dumps({"operation": operation, **arguments}).encode()
        ).decode()
        raw = await self.command(
            f"{shlex.quote(self._python)} {self.root}/controller.py call "
            f"{self.root} {shlex.quote(encoded)}"
        )
        try:
            response = json.loads(raw)
            epoch = response["epoch"]
            if not isinstance(epoch, str):
                raise ValueError("Invalid controller epoch")
            if self.epoch is not None and self.epoch != epoch:
                self.lost = True
                raise SessionLost("Controller changed; Python state is lost")
            self.epoch = epoch
            if "error" in response:
                if response.get("lost"):
                    self.lost = True
                    raise SessionLost(response["error"])
                raise SandboxError(response["error"])
            result = response["result"]
            if not isinstance(result, dict):
                raise ValueError("Invalid controller result")
            return result
        except (KeyError, ValueError, TypeError) as exc:
            raise SandboxError("Invalid response from sandbox controller") from exc

    async def stage(self, workspace: Workspace) -> None:
        for item in workspace.files:
            await self.upload(item.data, f"{self.root}/inputs/{item.name}")

    async def _cell_result(self, execution_id: str, status: dict) -> CellResult:
        if status.get("status") not in {"ok", "error", "interrupted"}:
            raise SessionLost(
                "Execution outcome is unknown or lost; code was not replayed"
            )
        size = status.get("size")
        # JSON escaping can expand text sixfold; images are base64 encoded.
        limit = (
            self.config.max_output_chars * 128
            + self.config.max_history_image_bytes * 2
            + 4096
        )
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= limit
        ):
            raise SandboxError("Invalid cell result size")
        data = await self.download(f"{self.root}/results/{execution_id}.json", size)
        if hashlib.sha256(data).hexdigest() != status.get("sha256"):
            raise SandboxError("Cell result digest mismatch")
        result = json.loads(data)
        if (
            result["status"] != status["status"]
            or result["execution_id"] != execution_id
        ):
            raise SandboxError("Cell result identity mismatch")
        outputs = list(result["outputs"])
        if result.get("truncated"):
            outputs.append(
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": "[Output truncated to configured preview limits]",
                }
            )
        return CellResult(
            execution_id,
            result["status"],
            tuple(outputs),
            result.get("execution_count"),
        )

    async def execute(self, code: str, execution_id: str) -> CellResult:
        if self.lost:
            raise SessionLost("Python state is lost; create a new session")
        self.settled = None
        try:
            try:
                await self.request("execute", execution_id=execution_id, code=code)
            except (SandboxError, TimeoutError):
                status = await self.request("status", execution_id=execution_id)
                if status["status"] == "unknown":
                    raise SessionLost(
                        "Execution outcome is unknown; code was not replayed"
                    )
            while True:
                status = await self.request("status", execution_id=execution_id)
                if status["status"] != "running":
                    self.settled = await self._cell_result(execution_id, status)
                    return self.settled
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            await self._interrupt_and_settle(execution_id)
            raise
        except (SandboxError, TimeoutError, KeyError, ValueError, TypeError):
            self.lost = True
            raise

    async def _interrupt_and_settle(self, execution_id: str) -> None:
        async def settle():
            await self.request("interrupt", execution_id=execution_id)
            while True:
                status = await self.request("status", execution_id=execution_id)
                if status.get("status") != "running":
                    self.settled = await self._cell_result(execution_id, status)
                    return
                await asyncio.sleep(0.1)

        task = asyncio.create_task(settle())
        try:
            async with asyncio.timeout(self.config.transport_seconds):
                await asyncio.shield(task)
        except BaseException:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.lost = True
            await self.close()

    async def artifacts(self) -> tuple[ArtifactRef, ...]:
        snapshot = await self.request("export")
        export_id = snapshot["export_id"]
        if (
            not isinstance(export_id, str)
            or re.fullmatch("[a-f0-9]{32}", export_id) is None
        ):
            raise SandboxError("Invalid artifact snapshot ID")
        descriptors = snapshot["files"]
        if (
            not isinstance(descriptors, list)
            or len(descriptors) > self.config.max_artifacts
        ):
            raise SandboxError("Too many generated artifacts")
        artifacts: list[ArtifactRef] = []
        total = 0
        names: set[str] = set()
        try:
            for item in descriptors:
                name, size = item["name"], item["size"]
                if (
                    not isinstance(name, str)
                    or not name
                    or "\\" in name
                    or any(ord(c) < 32 for c in name)
                    or PurePosixPath(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or name in names
                ):
                    raise SandboxError("Invalid artifact path")
                names.add(name)
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or not 0 <= size <= self.config.max_artifact_bytes
                ):
                    raise SandboxError("Artifact exceeds configured size budget")
                total += size
                if total > self.config.max_total_artifact_bytes:
                    raise SandboxError("Artifacts exceed configured total budget")
                data = await self.download(
                    f"{self.root}/exports/{export_id}/{name}", size
                )
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise SandboxError("Artifact digest mismatch")
                artifacts.append(ArtifactRef(name, data, item["media_type"]))
            return tuple(artifacts)
        finally:
            await self.request("release_export", export_id=export_id)

    async def close(self) -> None:
        if self._closing is None or (
            self._closing.done()
            and (self._closing.cancelled() or self._closing.exception() is not None)
        ):
            self._closing = asyncio.create_task(self._close())
        async with asyncio.timeout(self.config.open_seconds):
            await asyncio.shield(self._closing)

    async def _close(self) -> None:
        if self._opening is not None:
            # Even failed provisioning may have left a named resource owned by
            # an adapter. Preserve the opening error at its original call site.
            await asyncio.gather(self._opening, return_exceptions=True)
        if self.backend is not None:
            await self.backend.close()
