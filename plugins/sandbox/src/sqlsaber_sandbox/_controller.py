"""Private in-sandbox kernel controller and short-lived command bridge.

This file is copied into the sandbox and must not import SQLsaber. Only the
server imports Jupyter; the bridge uses the standard library. The socket is not
a security boundary against code running as the same sandbox user.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import uuid


class Controller:
    def __init__(self, root: Path, config: dict):
        self.root = root
        self.config = config
        self.epoch = uuid.uuid4().hex
        self.records: dict[str, dict] = {}
        self.active: asyncio.Task | None = None
        self.started = self.touched = time.monotonic()
        self.lost = False
        self.stopped = asyncio.Event()
        self.execution_started = asyncio.Event()
        self.interrupt_lock = asyncio.Lock()
        self.image_bytes = 0

    async def start(self):
        from jupyter_client import AsyncKernelManager

        self.manager = AsyncKernelManager(
            kernel_name="python3",
            transport="ipc",
            ip=str(self.root / "k"),
            connection_file=str(self.root / "kernel.json"),
        )
        await self.manager.start_kernel(cwd=str(self.root / "run"))
        self.client = self.manager.client()
        self.client.start_channels()
        await self.client.wait_for_ready(timeout=self.config["open_seconds"])
        # Keep plot display in the kernel, not in a GUI process.
        await self.client.execute_interactive(
            "get_ipython().run_line_magic('matplotlib', 'inline')",
            allow_stdin=False,
            timeout=30,
            output_hook=lambda message: None,
        )

    def output(self, record: dict, message: dict):
        kind = message["header"]["msg_type"]
        content = message["content"]
        remaining = self.config["max_output_chars"] - record["chars"]
        # Empty/repeated displays must not evade the content budget. Keep
        # draining messages after exhausting it, without retaining more objects.
        if len(record["outputs"]) >= self.config["max_output_chars"]:
            record["truncated"] = True
            return

        def text(value):
            nonlocal remaining
            value = str(value)
            clipped = value[: max(0, remaining)]
            record["chars"] += len(clipped)
            remaining -= len(clipped)
            if len(clipped) < len(value):
                record["truncated"] = True
            return clipped

        if kind == "clear_output":
            record["outputs"].clear()
        elif kind == "stream":
            value = text(content["text"])
            if value:
                record["outputs"].append(
                    {"output_type": "stream", "name": content["name"], "text": value}
                )
        elif kind == "error":
            record["outputs"].append(
                {
                    "output_type": "error",
                    "ename": text(content["ename"]),
                    "evalue": text(content["evalue"]),
                    "traceback": [
                        text(line) for line in content["traceback"] if remaining > 0
                    ],
                }
            )
        elif kind in {"display_data", "execute_result", "update_display_data"}:
            data = {}
            for mime, value in content.get("data", {}).items():
                if mime == "text/plain":
                    clipped = text(value)
                    if clipped:
                        data[mime] = clipped
                elif mime == "image/png":
                    size = len(value) * 3 // 4
                    if (
                        size <= self.config["max_image_bytes"]
                        and self.image_bytes + size
                        <= self.config["max_history_image_bytes"]
                    ):
                        data[mime] = value
                        self.image_bytes += size
                    else:
                        record["truncated"] = True
            if data:
                output = {"output_type": "display_data", "data": data, "metadata": {}}
                if kind == "execute_result":
                    output.update(
                        output_type=kind, execution_count=content["execution_count"]
                    )
                record["outputs"].append(output)

    async def collect(self, message_id: str, record: dict):
        async def shell():
            while True:
                message = await self.client.get_shell_msg()
                if message.get("parent_header", {}).get("msg_id") == message_id:
                    record["reply"] = message["content"]
                    return

        async def outputs():
            while True:
                message = await self.client.get_iopub_msg()
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                if message["header"]["msg_type"] == "execute_input":
                    self.execution_started.set()
                if (
                    message["header"]["msg_type"] == "status"
                    and message["content"]["execution_state"] == "idle"
                ):
                    return
                self.output(record, message)

        async with asyncio.TaskGroup() as group:
            group.create_task(shell())
            group.create_task(outputs())

    async def interrupt(self, record: dict):
        async with self.interrupt_lock:
            if record["status"] != "running" or record.get("interrupted"):
                return
            async with asyncio.timeout(self.config["transport_seconds"]):
                await self.execution_started.wait()
                record["interrupted"] = True
                message = self.client.session.msg("interrupt_request", {})
                self.client.control_channel.send(message)
                while True:
                    reply = await self.client.get_control_msg()
                    if (
                        reply["header"]["msg_type"] == "interrupt_reply"
                        and reply.get("parent_header", {}).get("msg_id")
                        == message["header"]["msg_id"]
                    ):
                        return

    async def execute(self, record: dict, code: str):
        collector = None
        try:
            message_id = self.client.execute(
                code, allow_stdin=False, stop_on_error=True
            )
            collector = asyncio.create_task(self.collect(message_id, record))
            try:
                async with asyncio.timeout(self.config["cell_seconds"]):
                    await asyncio.shield(collector)
            except TimeoutError:
                await self.interrupt(record)
                async with asyncio.timeout(5):
                    await asyncio.shield(collector)
            reply = record.pop("reply")
            record["status"] = (
                "interrupted" if record.get("interrupted") else reply["status"]
            )
            record["execution_count"] = reply.get("execution_count")
        except BaseException:
            self.lost = True
            record["status"] = "lost"
            if collector is not None:
                collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
            await self.manager.shutdown_kernel(now=True)
        finally:
            # Provider command stdout is not a bulk transport. Poll only small
            # status descriptors; terminal output travels through native files.
            payload = json.dumps(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"digest", "reply"}
                }
            ).encode()
            destination = self.root / "results"
            destination.mkdir(exist_ok=True)
            (destination / f"{record['execution_id']}.json").write_bytes(payload)
            record["size"] = len(payload)
            record["sha256"] = hashlib.sha256(payload).hexdigest()
            record.pop("outputs", None)
            self.touched = time.monotonic()

    def inventory(self):
        files = []
        total = 0
        for directory, dirs, names in os.walk(self.root / "run", followlinks=False):
            if any((Path(directory) / name).is_symlink() for name in dirs):
                raise ValueError("Symlink directories cannot be exported")
            dirs.sort()
            for name in sorted(names):
                path = Path(directory) / name
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Only regular generated files can be exported")
                total += info.st_size
                if (
                    info.st_size > self.config["max_artifact_bytes"]
                    or total > self.config["max_total_artifact_bytes"]
                    or len(files) >= self.config["max_artifacts"]
                ):
                    raise ValueError("Generated files exceed configured export budgets")
                files.append(
                    {
                        "name": path.relative_to(self.root / "run").as_posix(),
                        "size": info.st_size,
                        "media_type": mimetypes.guess_type(name)[0]
                        or "application/octet-stream",
                    }
                )
        return files

    def export(self):
        export_id = uuid.uuid4().hex
        destination = self.root / "exports" / export_id
        destination.mkdir(parents=True)
        files = self.inventory()
        for item in files:
            source = self.root / "run" / item["name"]
            target = destination / item["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            # Open without following a file replaced by a symlink after inventory.
            fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as reader, target.open("wb") as writer:
                if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                    raise ValueError("Artifact is not a regular file")
                remaining = item["size"]
                digest = hashlib.sha256()
                while remaining:
                    chunk = reader.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("Artifact changed during export")
                    digest.update(chunk)
                    writer.write(chunk)
                    remaining -= len(chunk)
                if reader.read(1):
                    raise ValueError("Artifact changed during export")
            item["sha256"] = digest.hexdigest()
        return {"export_id": export_id, "files": files}

    async def dispatch(self, request: dict):
        self.touched = time.monotonic()
        operation = request["operation"]
        if operation == "ping":
            return {"epoch": self.epoch, "lost": self.lost}
        if operation == "shutdown":
            self.stopped.set()
            return {}
        if operation == "status" and not await self.manager.is_alive():
            self.lost = True
            if self.active is not None:
                self.active.cancel()
                await asyncio.gather(self.active, return_exceptions=True)
        if self.lost:
            raise ValueError("Kernel state is lost; open a new session")
        if operation == "status":
            record = self.records.get(request["execution_id"])
            if record is None:
                return {"status": "unknown"}
            return {
                key: value
                for key, value in record.items()
                if key in {"status", "size", "sha256"}
            }
        if operation == "interrupt":
            record = self.records.get(request["execution_id"])
            if record is None:
                raise ValueError("Execution outcome is unknown")
            await self.interrupt(record)
            return {}
        if operation == "execute":
            execution_id = request["execution_id"]
            if (
                not isinstance(execution_id, str)
                or not execution_id
                or any(
                    char
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                    for char in execution_id
                )
            ):
                raise ValueError("Invalid execution ID")
            digest = hashlib.sha256(request["code"].encode()).hexdigest()
            previous = self.records.get(execution_id)
            if previous is not None:
                if previous["digest"] != digest:
                    raise ValueError("Execution ID already belongs to different code")
                return {"status": previous["status"]}
            if self.active is not None and not self.active.done():
                raise ValueError("Kernel is busy")
            record = {
                "execution_id": execution_id,
                "digest": digest,
                "status": "running",
                "outputs": [],
                "chars": 0,
                "truncated": False,
            }
            self.records[execution_id] = record
            self.execution_started.clear()
            self.active = asyncio.create_task(self.execute(record, request["code"]))
            return {"status": "running"}
        if self.active is not None and not self.active.done():
            raise ValueError("Kernel is busy")
        if operation == "workspace":
            return {"files": self.inventory()}
        if operation == "export":
            return self.export()
        if operation == "release_export":
            export_id = request["export_id"]
            if (
                not isinstance(export_id, str)
                or len(export_id) != 32
                or any(char not in "0123456789abcdef" for char in export_id)
            ):
                raise ValueError("Invalid export ID")
            shutil.rmtree(self.root / "exports" / export_id, ignore_errors=True)
            return {}
        raise ValueError("Unknown controller operation")

    async def connection(self, reader, writer):
        try:
            request = json.loads(await reader.readline())
            result = await self.dispatch(request)
            payload = {"epoch": self.epoch, "result": result}
        except Exception as exc:
            payload = {"epoch": self.epoch, "error": str(exc), "lost": self.lost}
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def watchdog(self):
        while not self.stopped.is_set():
            await asyncio.sleep(1)
            now = time.monotonic()
            lifetime = self.config["max_lifetime_seconds"]
            # Session inactivity includes model calls and transfers. Only the
            # host owns idle expiry; the guest can enforce absolute lifetime.
            if lifetime is not None and now - self.started >= lifetime:
                self.stopped.set()

    async def serve(self):
        try:
            await self.start()
            server = await asyncio.start_unix_server(
                self.connection,
                path=self.root / "controller.sock",
                limit=16 * 1024 * 1024,
            )
            async with server, asyncio.TaskGroup() as group:
                watcher = group.create_task(self.watchdog())
                await self.stopped.wait()
                watcher.cancel()
        finally:
            if self.active is not None:
                self.active.cancel()
                await asyncio.gather(self.active, return_exceptions=True)
            if hasattr(self, "client"):
                self.client.stop_channels()
            if hasattr(self, "manager"):
                await self.manager.shutdown_kernel(now=True)


async def bridge(root: Path, encoded: str):
    reader, writer = await asyncio.open_unix_connection(
        root / "controller.sock", limit=64 * 1024 * 1024
    )
    writer.write(base64.b64decode(encoded) + b"\n")
    await writer.drain()
    print((await reader.readline()).decode(), end="")
    writer.close()
    await writer.wait_closed()


if __name__ == "__main__":
    mode, root, argument = sys.argv[1:]
    if mode == "serve":
        asyncio.run(
            Controller(Path(root), json.loads(Path(argument).read_text())).serve()
        )
    else:
        asyncio.run(bridge(Path(root), argument))
