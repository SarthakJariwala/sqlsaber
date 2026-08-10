"""Remote Daytona implementation of notebook execution for legacy SDK 0.143.0."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import importlib.util
import json
import math
import os
import shlex
import stat
import tempfile
import time
import uuid
from collections.abc import Awaitable, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from ._files import build_artifact_inventory, validate_notebook_bytes
from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackend,
    NotebookBackendUnavailable,
    NotebookEnvironment,
    NotebookExecutionError,
    NotebookExecutionResult,
    NotebookExecutionTimeout,
    NotebookImageError,
    NotebookInfrastructureError,
    NotebookInput,
    NotebookLimitExceeded,
    bound_log,
    validate_artifact_path,
    validate_inputs,
)

_SDK_VERSION = "0.143.0"
_RUN_USER = "jovyan"
_RUN_GROUP = "users"
_RUNUSER = "/usr/sbin/runuser"
_PYTHON = "/opt/conda/bin/python"
_AUTO_STOP_MINUTES = 24 * 60
_DELETE_TIMEOUT_SECONDS = 60
_INVENTORY_MAX_BYTES = 128 * 1024
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_REMOTE_PARENT = "/sqlsaber-notebook"

_SECURE_PARENT_SCRIPT = r"""
import os
import pathlib
import stat
import sys

parent = pathlib.Path(sys.argv[1])

def validate_directory(path, *, allow_sticky_writable):
    info = os.lstat(path)
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode):
        raise SystemExit(f"trusted ancestor is not a directory: {path}")
    if info.st_uid != 0:
        raise SystemExit(f"trusted ancestor is not root-owned: {path}")
    if mode & 0o022 and not (allow_sticky_writable and mode & stat.S_ISVTX):
        raise SystemExit(f"trusted ancestor is writable by non-root users: {path}")

for ancestor in (*reversed(parent.parent.parents), parent.parent):
    validate_directory(ancestor, allow_sticky_writable=True)
try:
    parent.mkdir(mode=0o711)
except FileExistsError:
    pass
validate_directory(parent, allow_sticky_writable=False)
""".strip()

_ROOT_PREFLIGHT_SCRIPT = r"""
import os
import pathlib
import pwd
import shutil
import stat
import sys

run = pathlib.Path(sys.argv[1])
trusted_root = run.parent

def validate_directory(path, *, allow_sticky_writable):
    info = os.lstat(path)
    mode = stat.S_IMODE(info.st_mode)
    if not stat.S_ISDIR(info.st_mode):
        raise SystemExit(f"trusted ancestor is not a directory: {path}")
    if info.st_uid != 0:
        raise SystemExit(f"trusted ancestor is not root-owned: {path}")
    if mode & 0o022 and not (allow_sticky_writable and mode & stat.S_ISVTX):
        raise SystemExit(f"trusted ancestor is writable by non-root users: {path}")

if os.geteuid() != 0:
    raise SystemExit("Daytona control process is not root")
if pwd.getpwnam("jovyan").pw_name != "jovyan":
    raise SystemExit("jovyan user is missing")
for path in ("/usr/sbin/runuser", "/opt/conda/bin/python"):
    if not pathlib.Path(path).is_file():
        raise SystemExit(f"required executable is missing: {path}")
if shutil.which("jupyter") is None:
    raise SystemExit("jupyter is missing")
for ancestor in reversed(trusted_root.parent.parents):
    validate_directory(ancestor, allow_sticky_writable=True)
validate_directory(trusted_root.parent, allow_sticky_writable=False)
validate_directory(trusted_root, allow_sticky_writable=False)
probe = run / ".root-write-probe"
probe.write_bytes(b"ok")
probe.unlink()
""".strip()

_USER_PREFLIGHT_SCRIPT = r"""
import os
import pathlib
import pwd
import sys

source = pathlib.Path(sys.argv[1])
run = pathlib.Path(sys.argv[2])
trusted_root = run.parent
if pwd.getpwuid(os.geteuid()).pw_name != "jovyan":
    raise SystemExit("notebook process is not jovyan")
source.read_bytes()
probe = run / ".jovyan-write-probe"
probe.write_bytes(b"ok")
probe.unlink()
operations = (
    lambda: source.write_bytes(b"changed"),
    lambda: source.chmod(0o600),
    lambda: source.rename(run / "moved-input"),
    lambda: source.unlink(),
    lambda: (source.parent / "replacement").write_bytes(b"replacement"),
)
for operation in operations:
    try:
        operation()
    except OSError:
        continue
    raise SystemExit("protected notebook input was mutable")
for protected_directory in (trusted_root, trusted_root.parent):
    target = protected_directory.with_name(f"{protected_directory.name}-moved")
    try:
        protected_directory.rename(target)
    except OSError:
        continue
    raise SystemExit("trusted notebook control directory was renamable")
try:
    (trusted_root.parent / "replacement").mkdir()
except OSError:
    pass
else:
    raise SystemExit("trusted notebook parent was writable")
""".strip()

_STOP_USER_PROCESSES_SCRIPT = r"""
import os
import pwd
import signal
import time

uid = pwd.getpwnam("jovyan").pw_uid
for _ in range(4):
    found = False
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            if os.stat(entry.path).st_uid == uid:
                os.kill(int(entry.name), signal.SIGKILL)
                found = True
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    if not found:
        break
    time.sleep(0.05)
""".strip()

_INVENTORY_SCRIPT = r"""
import json
import os
import stat
import sys

root = sys.argv[1]
output = sys.argv[2]
max_artifacts = int(sys.argv[3])
max_file = int(sys.argv[4])
max_total = int(sys.argv[5])
max_notebook = int(sys.argv[6])
max_directories = 256
max_depth = 32
max_path = 1024
entries = {}
artifact_total = 0
directories = 0
try:
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        files.sort()
        relative_dir = os.path.relpath(current, root)
        depth = 0 if relative_dir == "." else len(relative_dir.split(os.sep))
        if depth > max_depth:
            raise ValueError("generated directory tree is too deep")
        directories += len(dirs)
        if directories > max_directories:
            raise ValueError("too many generated directories")
        for name in dirs:
            path = os.path.join(current, name)
            if os.path.islink(path):
                raise ValueError("symlinked generated directory")
        for name in files:
            path = os.path.join(current, name)
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("non-regular generated file")
            relative = os.path.relpath(path, root).replace(os.sep, "/")
            if len(relative) > max_path:
                raise ValueError("generated path is too long")
            if relative in entries:
                raise ValueError("duplicate generated path")
            if relative == "notebook.ipynb":
                if info.st_size > max_notebook:
                    raise ValueError("executed notebook is too large")
            else:
                if info.st_size > max_file:
                    raise ValueError("generated artifact is too large")
                artifact_total += info.st_size
                if artifact_total > max_total:
                    raise ValueError("generated artifacts are too large")
                if len(entries) >= max_artifacts + ("notebook.ipynb" in entries):
                    raise ValueError("too many generated artifacts")
            entries[relative] = {
                "size": info.st_size,
                "device": info.st_dev,
                "inode": info.st_ino,
            }
    if "notebook.ipynb" not in entries:
        raise ValueError("executed notebook is missing")
    encoded = json.dumps({"files": entries}, separators=(",", ":")).encode()
    if len(encoded) > 131072:
        raise ValueError("artifact inventory is too large")
    with open(output, "wb") as stream:
        stream.write(encoded)
except Exception as exc:
    print(str(exc)[:1024], file=sys.stderr)
    raise SystemExit(2)
""".strip()

_METADATA_SCRIPT = r"""
import json
import os
import stat
import sys

info = os.lstat(sys.argv[1])
print(json.dumps({
    "regular": stat.S_ISREG(info.st_mode),
    "size": info.st_size,
    "device": info.st_dev,
    "inode": info.st_ino,
}, separators=(",", ":")))
""".strip()

_LOG_READER_SCRIPT = r"""
import base64
import json
import pathlib
import sys

limit = int(sys.argv[3])
def read_bounded(path):
    stream = pathlib.Path(path)
    try:
        size = stream.stat().st_size
        with stream.open("rb") as handle:
            if size <= limit:
                return handle.read(limit + 1)
            head = limit // 2
            value = handle.read(head)
            handle.seek(max(0, size - (limit - head)))
            return value + handle.read(limit - head)
    except FileNotFoundError:
        return b""
print(json.dumps({
    "stdout": base64.b64encode(read_bounded(sys.argv[1])).decode(),
    "stderr": base64.b64encode(read_bounded(sys.argv[2])).decode(),
}, separators=(",", ":")))
""".strip()

_LOG_WRAPPER = 'stdout=$1; stderr=$2; shift 2; "$@" >"$stdout" 2>"$stderr"'

Identity = tuple[int, int, int]


class DaytonaNotebookBackend(NotebookBackend):
    """Open explicitly selected notebook environments in Daytona."""

    name = "daytona"

    def available(self) -> bool:
        return importlib.util.find_spec("daytona") is not None

    async def open(
        self,
        inputs: Sequence[NotebookInput],
        *,
        image: str,
        snapshot: str | None = None,
        limits: ExecutionLimits,
    ) -> DaytonaNotebookEnvironment:
        validated = validate_inputs(inputs, limits, backend=self.name)
        if snapshot is not None:
            snapshot = snapshot.strip()
            if not snapshot:
                raise NotebookImageError(
                    "Notebook snapshot cannot be empty",
                    backend=self.name,
                    phase="configuration",
                )
        sdk = _load_daytona()
        name = f"sqlsaber-notebook-{uuid.uuid4().hex}"
        source_kind = "snapshot" if snapshot is not None else "image"
        source_value = snapshot if snapshot is not None else image
        client: Any | None = None
        sandbox: Any | None = None
        try:
            client = sdk.AsyncDaytona()
            common_params = {
                "language": "python",
                "os_user": "root",
                "name": name,
                "labels": {"application": "sqlsaber", "purpose": "notebook"},
                "network_block_all": True,
                "ephemeral": True,
                "auto_stop_interval": _AUTO_STOP_MINUTES,
            }
            if snapshot is not None:
                params = sdk.CreateSandboxFromSnapshotParams(
                    snapshot=snapshot,
                    **common_params,
                )
            else:
                cpu = _daytona_cpu(limits.cpu_cores)
                memory = _daytona_memory_gib(limits.memory_mb)
                runtime_image = sdk.Image.base(image).dockerfile_commands(
                    ["USER root", "WORKDIR /home/jovyan"]
                )
                params = sdk.CreateSandboxFromImageParams(
                    image=runtime_image,
                    resources=sdk.Resources(cpu=cpu, memory=memory),
                    **common_params,
                )
            async with asyncio.timeout(limits.image_prepare_seconds):
                sandbox = await client.create(
                    params,
                    timeout=limits.image_prepare_seconds,
                )
            remote_root = f"{_REMOTE_PARENT}/{uuid.uuid4().hex}"
            environment = DaytonaNotebookEnvironment(
                sdk=sdk,
                client=client,
                sandbox=sandbox,
                sandbox_name=name,
                limits=limits,
                remote_root=remote_root,
            )
            async with asyncio.timeout(limits.open_seconds):
                await environment.stage(validated)
            return environment
        except asyncio.CancelledError:
            await _best_effort_cleanup(sdk, client, sandbox, name)
            raise
        except TimeoutError as exc:
            await _best_effort_cleanup(sdk, client, sandbox, name)
            phase = "image-prepare" if sandbox is None else "input-upload"
            seconds = (
                limits.image_prepare_seconds if sandbox is None else limits.open_seconds
            )
            raise NotebookExecutionTimeout(
                f"Daytona {phase} timed out after {seconds} seconds",
                backend=self.name,
                phase=phase,
            ) from exc
        except NotebookExecutionError:
            await _best_effort_cleanup(sdk, client, sandbox, name)
            raise
        except Exception as exc:
            await _best_effort_cleanup(sdk, client, sandbox, name)
            if _is_sdk_exception(exc, sdk, "DaytonaTimeoutError"):
                raise NotebookExecutionTimeout(
                    f"Daytona {source_kind} preparation timed out",
                    backend=self.name,
                    phase="image-prepare",
                    diagnostics=bound_log(str(exc), limits.max_log_chars),
                ) from exc
            if sandbox is None and _looks_like_image_error(exc):
                raise NotebookImageError(
                    f"Could not prepare Daytona notebook {source_kind} "
                    f"{source_value!r}",
                    backend=self.name,
                    phase="image-prepare",
                    diagnostics=bound_log(str(exc), limits.max_log_chars),
                ) from exc
            if sandbox is None:
                raise NotebookBackendUnavailable(
                    "Daytona could not create a sandbox; verify DAYTONA_API_KEY, "
                    "DAYTONA_API_URL, account limits, and service availability",
                    backend=self.name,
                    phase="environment-open",
                    diagnostics=bound_log(str(exc), limits.max_log_chars),
                ) from exc
            raise NotebookInfrastructureError(
                "Could not initialize the Daytona notebook environment",
                backend=self.name,
                phase="input-upload",
                diagnostics=bound_log(str(exc), limits.max_log_chars),
            ) from exc


class DaytonaNotebookEnvironment(NotebookEnvironment):
    """A private Daytona sandbox reused for fresh-kernel notebook runs."""

    def __init__(
        self,
        *,
        sdk: Any,
        client: Any,
        sandbox: Any,
        sandbox_name: str,
        limits: ExecutionLimits,
        remote_root: str,
    ) -> None:
        self.sdk = sdk
        self.client = client
        self.sandbox = sandbox
        self.sandbox_name = sandbox_name
        self.limits = limits
        self.remote_root = remote_root
        self.inputs_path = f"{remote_root}/inputs"
        self.run_path = f"{remote_root}/run"
        self.control_path = f"{remote_root}/control"
        self.notebook_path = f"{self.run_path}/notebook.ipynb"
        self.inventory_path = f"{self.control_path}/inventory.json"
        self.stdout_path = f"{self.control_path}/stdout.log"
        self.stderr_path = f"{self.control_path}/stderr.log"
        self._inventory: tuple[ArtifactInfo, ...] = ()
        self._identities: dict[str, Identity] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._cleanup_task: asyncio.Task[None] | None = None

    async def stage(self, inputs: Sequence[NotebookInput]) -> None:
        sentinel = f"{self.inputs_path}/.sqlsaber-preflight"
        try:
            result = await self._exec(
                "image-preflight",
                _PYTHON,
                "-c",
                _SECURE_PARENT_SCRIPT,
                _REMOTE_PARENT,
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Notebook image cannot provide a secure Daytona control parent",
                phase="image-preflight",
                image_error=True,
            )
            result = await self._exec(
                "input-upload",
                "mkdir",
                "-p",
                self.inputs_path,
                self.run_path,
                self.control_path,
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not create Daytona notebook directories",
                phase="input-upload",
            )
            await self._upload_file(sentinel, b"immutable", phase="input-upload")
            for item in inputs:
                await self._upload_file(
                    f"{self.inputs_path}/{item.name}",
                    item.data,
                    phase="input-upload",
                )
            result = await self._exec(
                "input-upload",
                "chown",
                "-R",
                "root:root",
                self.inputs_path,
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not protect Daytona notebook input ownership",
                phase="input-upload",
            )
            result = await self._exec(
                "input-upload",
                "chmod",
                "-R",
                "a-w",
                self.inputs_path,
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not make Daytona notebook inputs read-only",
                phase="input-upload",
            )
            await self._preflight_runtime(sentinel)
            result = await self._exec(
                "input-upload",
                "rm",
                "-f",
                sentinel,
                timeout=self.limits.open_seconds,
            )
            self._require_success(
                result,
                "Could not remove the Daytona input preflight sentinel",
                phase="input-upload",
            )
        except NotebookExecutionError:
            raise
        except Exception as exc:
            raise NotebookInfrastructureError(
                "Could not stage Daytona notebook inputs",
                backend="daytona",
                phase="input-upload",
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    async def execute(
        self,
        notebook: bytes,
        *,
        cell_timeout: int | None,
        command_timeout: int | None,
    ) -> NotebookExecutionResult:
        async with self._lock:
            self._ensure_open()
            validate_notebook_bytes(
                notebook,
                self.limits,
                backend="daytona",
                phase="notebook-upload",
            )
            self._inventory = ()
            self._identities = {}
            notebook_command_started = False
            run_secured = False
            try:
                await self._reset_run(notebook)
                effective_cell_timeout = _bounded_timeout(
                    cell_timeout, self.limits.cell_seconds
                )
                effective_command_timeout = _bounded_timeout(
                    command_timeout, self.limits.command_seconds
                )
                notebook_command_started = True
                response = await self._exec(
                    "notebook-execution",
                    "/bin/sh",
                    "-c",
                    _LOG_WRAPPER,
                    "sqlsaber-nbconvert",
                    self.stdout_path,
                    self.stderr_path,
                    _RUNUSER,
                    "-u",
                    _RUN_USER,
                    "--",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--inplace",
                    "--allow-errors",
                    f"--ExecutePreprocessor.timeout={effective_cell_timeout or -1}",
                    "notebook.ipynb",
                    timeout=effective_command_timeout,
                    workdir=self.run_path,
                )
                stdout, stderr = await self._read_remote_logs()
                await self._stop_run_processes()
                await self._freeze_run()
                run_secured = True
                if response[0] != 0:
                    raise NotebookInfrastructureError(
                        "Daytona notebook execution failed",
                        backend="daytona",
                        phase="notebook-execution",
                        diagnostics=stderr or stdout,
                    )
                sizes, identities = await self._inventory_sizes()
                notebook_size = sizes.pop("notebook.ipynb", None)
                if notebook_size is None:
                    raise NotebookInfrastructureError(
                        "Daytona execution did not return a notebook",
                        backend="daytona",
                        phase="notebook-download",
                    )
                if notebook_size > self.limits.max_notebook_bytes:
                    raise NotebookLimitExceeded(
                        f"Notebook exceeds {self.limits.max_notebook_bytes} bytes",
                        backend="daytona",
                        phase="notebook-download",
                    )
                executed = await self._download_frozen_file(
                    self.notebook_path,
                    expected_size=notebook_size,
                    expected_identity=identities["notebook.ipynb"],
                    byte_limit=self.limits.max_notebook_bytes,
                    phase="notebook-download",
                )
                validate_notebook_bytes(
                    executed,
                    self.limits,
                    backend="daytona",
                    phase="notebook-download",
                )
                inventory = build_artifact_inventory(
                    sizes,
                    self.limits,
                    backend="daytona",
                )
                self._identities = {
                    path: identity
                    for path, identity in identities.items()
                    if path != "notebook.ipynb"
                }
                self._inventory = inventory
                return NotebookExecutionResult(
                    notebook=executed,
                    artifacts=inventory,
                    stdout=stdout,
                    stderr=stderr,
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except asyncio.CancelledError:
                await self._terminate_after_interruption()
                raise
            except NotebookExecutionError:
                if notebook_command_started and not run_secured:
                    await self._terminate_after_interruption()
                raise
            except Exception as exc:
                if notebook_command_started and not run_secured:
                    await self._terminate_after_interruption()
                raise NotebookInfrastructureError(
                    "Could not transfer Daytona notebook results",
                    backend="daytona",
                    phase="notebook-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

    async def read_artifact(self, artifact: ArtifactInfo) -> bytes:
        async with self._lock:
            self._ensure_open()
            if artifact not in self._inventory:
                raise NotebookInfrastructureError(
                    f"Unknown artifact: {artifact.path}",
                    backend="daytona",
                    phase="artifact-download",
                )
            relative = validate_artifact_path(artifact.path, backend="daytona")
            identity = self._identities.get(artifact.path)
            if identity is None:
                raise NotebookInfrastructureError(
                    f"Artifact identity is unavailable: {artifact.path}",
                    backend="daytona",
                    phase="artifact-download",
                )
            try:
                return await self._download_frozen_file(
                    f"{self.run_path}/{relative.as_posix()}",
                    expected_size=artifact.size,
                    expected_identity=identity,
                    byte_limit=self.limits.max_artifact_bytes,
                    phase="artifact-download",
                )
            except NotebookExecutionTimeout:
                await self._terminate_after_interruption()
                raise
            except NotebookExecutionError:
                raise
            except Exception as exc:
                raise NotebookInfrastructureError(
                    f"Could not read artifact: {artifact.path}",
                    backend="daytona",
                    phase="artifact-download",
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc

    async def list_workspace(self) -> tuple[ArtifactInfo, ...]:
        async with self._lock:
            self._ensure_open()
            return self._inventory

    async def close(self) -> None:
        async with self._lock:
            if self._cleanup_task is None:
                self._closed = True
                self._inventory = ()
                self._identities = {}
                self._cleanup_task = asyncio.create_task(self._cleanup())
            task = self._cleanup_task
        await asyncio.shield(task)

    async def _reset_run(self, notebook: bytes) -> None:
        result = await self._exec(
            "run-setup",
            "rm",
            "-rf",
            "--",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not reset the Daytona run directory",
            phase="run-setup",
        )
        result = await self._exec(
            "run-setup",
            "mkdir",
            "-p",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not recreate the Daytona run directory",
            phase="run-setup",
        )
        result = await self._exec(
            "run-setup",
            "rm",
            "-f",
            self.stdout_path,
            self.stderr_path,
            self.inventory_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not reset Daytona control files",
            phase="run-setup",
        )
        await self._upload_file(
            self.notebook_path,
            notebook,
            phase="notebook-upload",
        )
        result = await self._exec(
            "run-setup",
            "chown",
            "-R",
            f"{_RUN_USER}:{_RUN_GROUP}",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image cannot prepare an unprivileged Daytona run directory",
            phase="run-setup",
            image_error=True,
        )

    async def _preflight_runtime(self, sentinel: str) -> None:
        result = await self._exec(
            "image-preflight",
            _PYTHON,
            "-c",
            _ROOT_PREFLIGHT_SCRIPT,
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image must provide a root Daytona control identity",
            phase="image-preflight",
            image_error=True,
        )
        result = await self._exec(
            "image-preflight",
            _RUNUSER,
            "-u",
            _RUN_USER,
            "--",
            "jupyter",
            "nbconvert",
            "--version",
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image must provide jovyan, runuser, jupyter, and nbconvert",
            phase="image-preflight",
            image_error=True,
        )
        result = await self._exec(
            "image-preflight",
            "chown",
            "-R",
            f"{_RUN_USER}:{_RUN_GROUP}",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image cannot prepare a jovyan-owned run directory",
            phase="image-preflight",
            image_error=True,
        )
        result = await self._exec(
            "image-preflight",
            _RUNUSER,
            "-u",
            _RUN_USER,
            "--",
            _PYTHON,
            "-c",
            _USER_PREFLIGHT_SCRIPT,
            sentinel,
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Notebook image does not preserve immutable Daytona inputs for jovyan",
            phase="image-preflight",
            image_error=True,
        )

    async def _stop_run_processes(self) -> None:
        result = await self._exec(
            "notebook-execution",
            _PYTHON,
            "-c",
            _STOP_USER_PROCESSES_SCRIPT,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not stop background Daytona notebook processes",
            phase="notebook-execution",
        )

    async def _freeze_run(self) -> None:
        result = await self._exec(
            "artifact-inventory",
            "chown",
            "-R",
            "root:root",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not freeze Daytona notebook results",
            phase="artifact-inventory",
        )
        result = await self._exec(
            "artifact-inventory",
            "chmod",
            "-R",
            "a-w",
            self.run_path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not freeze Daytona notebook results",
            phase="artifact-inventory",
        )

    async def _inventory_sizes(self) -> tuple[dict[str, int], dict[str, Identity]]:
        result = await self._exec(
            "artifact-inventory",
            _PYTHON,
            "-c",
            _INVENTORY_SCRIPT,
            self.run_path,
            self.inventory_path,
            str(self.limits.max_artifacts),
            str(self.limits.max_artifact_bytes),
            str(self.limits.max_total_artifact_bytes),
            str(self.limits.max_notebook_bytes),
            timeout=self.limits.open_seconds,
        )
        if result[0] != 0:
            raise NotebookLimitExceeded(
                "Daytona artifact inventory exceeded limits or contained unsafe files",
                backend="daytona",
                phase="artifact-inventory",
                diagnostics=result[1],
            )
        control_identity = await self._remote_identity(
            self.inventory_path,
            phase="artifact-inventory",
        )
        if control_identity[2] > _INVENTORY_MAX_BYTES:
            raise NotebookInfrastructureError(
                "Daytona returned an oversized artifact inventory",
                backend="daytona",
                phase="artifact-inventory",
            )
        payload_bytes = await self._download_frozen_file(
            self.inventory_path,
            expected_size=control_identity[2],
            expected_identity=control_identity,
            byte_limit=_INVENTORY_MAX_BYTES,
            phase="artifact-inventory",
        )
        try:
            payload = json.loads(payload_bytes)
            files = payload["files"]
            if not isinstance(files, dict):
                raise TypeError("files must be an object")
            sizes: dict[str, int] = {}
            identities: dict[str, Identity] = {}
            for path, metadata in files.items():
                if not isinstance(path, str) or not isinstance(metadata, dict):
                    raise TypeError("invalid inventory entry")
                size = metadata["size"]
                device = metadata["device"]
                inode = metadata["inode"]
                if not all(isinstance(value, int) for value in (size, device, inode)):
                    raise TypeError("invalid inventory identity")
                if size < 0 or device < 0 or inode < 0:
                    raise ValueError("negative inventory identity")
                sizes[path] = size
                identities[path] = (device, inode, size)
            return sizes, identities
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise NotebookInfrastructureError(
                "Daytona returned an invalid artifact inventory",
                backend="daytona",
                phase="artifact-inventory",
            ) from exc

    async def _read_remote_logs(self) -> tuple[str, str]:
        byte_limit = max(self.limits.max_log_chars * 4, 32_000)
        result = await self._exec(
            "notebook-execution",
            _PYTHON,
            "-c",
            _LOG_READER_SCRIPT,
            self.stdout_path,
            self.stderr_path,
            str(byte_limit),
            timeout=self.limits.open_seconds,
            result_limit=byte_limit * 3,
        )
        self._require_success(
            result,
            "Could not read bounded Daytona notebook logs",
            phase="notebook-execution",
        )
        try:
            payload = json.loads(result[1])
            stdout = base64.b64decode(payload["stdout"], validate=True)
            stderr = base64.b64decode(payload["stderr"], validate=True)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise NotebookInfrastructureError(
                "Daytona returned invalid bounded notebook logs",
                backend="daytona",
                phase="notebook-execution",
            ) from exc
        return (
            bound_log(stdout, self.limits.max_log_chars),
            bound_log(stderr, self.limits.max_log_chars),
        )

    async def _upload_file(self, path: str, data: bytes, *, phase: str) -> None:
        try:
            await self._operation(
                self.sandbox.fs.upload_file(
                    data,
                    path,
                    timeout=self.limits.open_seconds,
                ),
                timeout=self.limits.open_seconds,
                phase=phase,
            )
        except NotebookExecutionError:
            raise
        except Exception as exc:
            raise NotebookInfrastructureError(
                f"Could not upload Daytona file for {phase}",
                backend="daytona",
                phase=phase,
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    async def _download_frozen_file(
        self,
        remote_path: str,
        *,
        expected_size: int,
        expected_identity: Identity,
        byte_limit: int,
        phase: str,
    ) -> bytes:
        if expected_size < 0 or expected_size > byte_limit:
            raise NotebookLimitExceeded(
                f"Daytona file exceeds {byte_limit} bytes",
                backend="daytona",
                phase=phase,
            )
        if await self._remote_identity(remote_path, phase=phase) != expected_identity:
            raise NotebookInfrastructureError(
                "Daytona file changed after inventory",
                backend="daytona",
                phase=phase,
            )
        file_descriptor, local_name = tempfile.mkstemp(prefix="sqlsaber-daytona-")
        os.close(file_descriptor)
        local_path = Path(local_name)
        try:
            await self._operation(
                self.sandbox.fs.download_file(
                    remote_path,
                    local_name,
                    self.limits.open_seconds,
                ),
                timeout=self.limits.open_seconds,
                phase=phase,
            )
            metadata = await asyncio.to_thread(local_path.lstat)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
                raise NotebookInfrastructureError(
                    "Daytona file changed during download",
                    backend="daytona",
                    phase=phase,
                )
            data = await asyncio.to_thread(
                _read_bounded_local_file,
                local_path,
                byte_limit,
            )
            if len(data) != expected_size:
                raise NotebookInfrastructureError(
                    "Daytona file changed during download",
                    backend="daytona",
                    phase=phase,
                )
            if (
                await self._remote_identity(remote_path, phase=phase)
                != expected_identity
            ):
                raise NotebookInfrastructureError(
                    "Daytona file changed during download",
                    backend="daytona",
                    phase=phase,
                )
            return data
        finally:
            with contextlib.suppress(OSError):
                local_path.unlink()

    async def _remote_identity(self, path: str, *, phase: str) -> Identity:
        result = await self._exec(
            phase,
            _PYTHON,
            "-c",
            _METADATA_SCRIPT,
            path,
            timeout=self.limits.open_seconds,
        )
        self._require_success(
            result,
            "Could not verify Daytona file metadata",
            phase=phase,
        )
        try:
            payload = json.loads(result[1])
            if payload["regular"] is not True:
                raise ValueError("file is not regular")
            size = payload["size"]
            device = payload["device"]
            inode = payload["inode"]
            if not all(isinstance(value, int) for value in (size, device, inode)):
                raise TypeError("invalid metadata")
            if size < 0 or device < 0 or inode < 0:
                raise ValueError("negative metadata")
            return device, inode, size
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise NotebookInfrastructureError(
                "Daytona returned invalid file metadata",
                backend="daytona",
                phase=phase,
                diagnostics=result[1],
            ) from exc

    async def _exec(
        self,
        phase: str,
        *argv: str,
        timeout: int | None,
        workdir: str | None = None,
        result_limit: int | None = None,
    ) -> tuple[int, str]:
        try:
            async with asyncio.timeout(timeout):
                response = await self.sandbox.process.exec(
                    shlex.join(argv),
                    cwd=workdir,
                    timeout=timeout,
                )
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Daytona {phase} timed out after {timeout} seconds",
                backend="daytona",
                phase=phase,
            ) from exc
        except Exception as exc:
            if _is_sdk_exception(exc, self.sdk, "DaytonaTimeoutError"):
                raise NotebookExecutionTimeout(
                    f"Daytona {phase} timed out after {timeout} seconds",
                    backend="daytona",
                    phase=phase,
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc
            raise NotebookInfrastructureError(
                f"Daytona {phase} operation failed",
                backend="daytona",
                phase=phase,
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc
        return response.exit_code, bound_log(
            response.result,
            result_limit or self.limits.max_log_chars,
        )

    async def _operation(
        self,
        operation: Awaitable[Any],
        *,
        timeout: int,
        phase: str,
    ) -> Any:
        try:
            async with asyncio.timeout(timeout):
                return await operation
        except TimeoutError as exc:
            raise NotebookExecutionTimeout(
                f"Daytona {phase} timed out after {timeout} seconds",
                backend="daytona",
                phase=phase,
            ) from exc
        except Exception as exc:
            if _is_sdk_exception(exc, self.sdk, "DaytonaTimeoutError"):
                raise NotebookExecutionTimeout(
                    f"Daytona {phase} timed out after {timeout} seconds",
                    backend="daytona",
                    phase=phase,
                    diagnostics=bound_log(str(exc), self.limits.max_log_chars),
                ) from exc
            raise NotebookInfrastructureError(
                f"Daytona {phase} operation failed",
                backend="daytona",
                phase=phase,
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    def _require_success(
        self,
        result: tuple[int, str],
        message: str,
        *,
        phase: str,
        image_error: bool = False,
    ) -> None:
        if result[0] == 0:
            return
        error_type = NotebookImageError if image_error else NotebookInfrastructureError
        raise error_type(
            message,
            backend="daytona",
            phase=phase,
            diagnostics=result[1],
        )

    async def _terminate_after_interruption(self) -> None:
        self._closed = True
        self._inventory = ()
        self._identities = {}
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup())
        with contextlib.suppress(Exception):
            await asyncio.shield(self._cleanup_task)

    async def _cleanup(self) -> None:
        try:
            await _delete_resources(
                self.sdk,
                self.client,
                self.sandbox,
                self.sandbox_name,
            )
        except Exception as exc:
            raise NotebookInfrastructureError(
                "Could not remove the Daytona notebook environment",
                backend="daytona",
                phase="cleanup",
                diagnostics=bound_log(str(exc), self.limits.max_log_chars),
            ) from exc

    def _ensure_open(self) -> None:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook environment is closed",
                backend="daytona",
                phase="lifecycle",
            )


def _daytona_cpu(value: float) -> int:
    if not math.isfinite(value):
        raise NotebookLimitExceeded(
            "Daytona CPU limit must be finite",
            backend="daytona",
            phase="configuration",
        )
    result = math.floor(value)
    if result < 1:
        raise NotebookLimitExceeded(
            "Daytona requires at least one whole CPU",
            backend="daytona",
            phase="configuration",
        )
    return result


def _daytona_memory_gib(value: int) -> int:
    result = value // 1024
    if result < 1:
        raise NotebookLimitExceeded(
            "Daytona requires at least one GiB of memory",
            backend="daytona",
            phase="configuration",
        )
    return result


def _bounded_timeout(
    requested: int | None,
    configured: int | None,
) -> int | None:
    if requested is None:
        return configured
    if configured is None:
        return requested
    return min(requested, configured)


def _read_bounded_local_file(path: Path, byte_limit: int) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(byte_limit + 1)
    if len(data) > byte_limit:
        raise NotebookLimitExceeded(
            f"Daytona download exceeds {byte_limit} bytes",
            backend="daytona",
            phase="file-download",
        )
    return data


async def _delete_resources(
    sdk: Any,
    client: Any,
    sandbox: Any,
    sandbox_name: str,
) -> None:
    try:
        try:
            async with asyncio.timeout(_DELETE_TIMEOUT_SECONDS + 5):
                await sandbox.delete(timeout=_DELETE_TIMEOUT_SECONDS)
        except Exception as exc:
            if _is_sdk_exception(exc, sdk, "DaytonaNotFoundError"):
                return
            raise
        await _wait_until_deleted(
            client,
            sandbox_name,
            sdk,
            timeout=_DELETE_TIMEOUT_SECONDS,
        )
    finally:
        await asyncio.wait_for(client.close(), timeout=10)


async def _wait_until_deleted(
    client: Any,
    sandbox_name: str,
    sdk: Any,
    *,
    timeout: float,
    poll_interval: float = 0.25,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Daytona sandbox {sandbox_name!r} still exists after deletion"
            )
        try:
            await asyncio.wait_for(
                client.get(sandbox_name),
                timeout=min(10, remaining),
            )
        except Exception as exc:
            if _is_sdk_exception(exc, sdk, "DaytonaNotFoundError"):
                return
            raise
        await asyncio.sleep(min(poll_interval, max(0, deadline - time.monotonic())))


async def _best_effort_cleanup(
    sdk: Any,
    client: Any | None,
    sandbox: Any | None,
    sandbox_name: str,
) -> None:
    if client is None:
        return

    async def cleanup() -> None:
        target = sandbox
        if target is None:
            try:
                target = await _lookup_after_create_failure(
                    client,
                    sandbox_name,
                    sdk,
                )
            except BaseException:
                await asyncio.wait_for(client.close(), timeout=10)
                raise
        if target is None:
            await asyncio.wait_for(client.close(), timeout=10)
            return
        await _delete_resources(sdk, client, target, sandbox_name)

    task = asyncio.create_task(cleanup())
    with contextlib.suppress(Exception):
        await asyncio.shield(task)


async def _lookup_after_create_failure(
    client: Any,
    sandbox_name: str,
    sdk: Any,
) -> Any | None:
    for attempt in range(5):
        try:
            return await asyncio.wait_for(client.get(sandbox_name), timeout=2)
        except Exception as exc:
            if not _is_sdk_exception(exc, sdk, "DaytonaNotFoundError"):
                raise
        if attempt < 4:
            await asyncio.sleep(0.25)
    return None


def _looks_like_image_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    return status_code == 404 or any(
        marker in message
        for marker in (
            "image not found",
            "failed to pull",
            "manifest unknown",
            "registry error",
            "snapshot creation failed",
        )
    )


def _is_sdk_exception(exc: Exception, sdk: Any, *names: str) -> bool:
    types = tuple(
        candidate
        for name in names
        if isinstance((candidate := getattr(sdk, name, None)), type)
    )
    return bool(types) and isinstance(exc, types)


def _load_daytona() -> Any:
    try:
        sdk = importlib.import_module("daytona")
    except ImportError as exc:
        raise NotebookBackendUnavailable(
            "Daytona backend requires the `sqlsaber-notebook[daytona]` extra",
            backend="daytona",
            phase="availability",
        ) from exc
    try:
        installed_version = version("daytona")
    except PackageNotFoundError as exc:
        raise NotebookBackendUnavailable(
            "Could not determine the installed Daytona SDK version",
            backend="daytona",
            phase="availability",
        ) from exc
    if installed_version != _SDK_VERSION:
        raise NotebookBackendUnavailable(
            f"Daytona backend requires daytona=={_SDK_VERSION}; found "
            f"{installed_version}",
            backend="daytona",
            phase="availability",
        )
    required = (
        "AsyncDaytona",
        "CreateSandboxFromImageParams",
        "CreateSandboxFromSnapshotParams",
        "Image",
        "Resources",
        "DaytonaError",
        "DaytonaNotFoundError",
        "DaytonaRateLimitError",
        "DaytonaTimeoutError",
    )
    missing = [name for name in required if not hasattr(sdk, name)]
    if missing:
        raise NotebookBackendUnavailable(
            "Installed Daytona SDK 0.143.0 is incompatible; missing: "
            + ", ".join(missing),
            backend="daytona",
            phase="availability",
        )
    return sdk
