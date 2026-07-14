"""Transactional notebook state over the provider-neutral execution seam."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import nbformat

from ._shared import (
    DEFAULT_EXECUTION_LIMITS,
    MAX_CELL_SOURCE_CHARS,
    MAX_CELLS,
    MAX_TOTAL_SOURCE_CHARS,
)
from .execution import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookBackend,
    NotebookEnvironment,
    NotebookExecutionResult,
    NotebookInfrastructureError,
    NotebookInput,
    NotebookLimitExceeded,
)
from .result import Workspace

_MANIFEST_NAME = "manifest.json"


@dataclass(slots=True)
class NotebookSession:
    workspace: Workspace
    backend: NotebookBackend
    image: str
    execution_limits: ExecutionLimits = DEFAULT_EXECUTION_LIMITS
    include_snapshot_images: bool = False
    environment: NotebookEnvironment | None = None
    cells: list[str] = field(default_factory=list)
    outputs: list[list[dict[str, Any]]] = field(default_factory=list)
    artifacts: tuple[ArtifactInfo, ...] = ()
    sent_image_hashes: set[str] = field(default_factory=set)
    sent_image_bytes: int = 0
    run_count: int = 0
    run_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _notebook: bytes = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_sources(self.cells)
        if not self.outputs:
            self.outputs = [[] for _ in self.cells]
        elif len(self.outputs) != len(self.cells):
            raise ValueError("Notebook outputs must align one-to-one with cells")
        self._notebook = self.serialize_notebook()

    async def ensure_environment(self) -> NotebookEnvironment:
        if self._closed:
            raise NotebookInfrastructureError(
                "Notebook session is closed",
                backend=self.backend.name,
                phase="lifecycle",
            )
        if self.environment is None:
            self.environment = await self.backend.open(
                self._staged_inputs(),
                image=self.image,
                limits=self.execution_limits,
            )
        return self.environment

    async def run_notebook(
        self,
        *,
        cell_timeout: int | None = None,
        command_timeout: int | None = None,
    ) -> NotebookExecutionResult:
        """Execute all cells with a fresh kernel and install results atomically."""

        self._validate_sources(self.cells)
        environment = await self.ensure_environment()
        source_notebook = self.serialize_notebook()
        result = await environment.execute(
            source_notebook,
            cell_timeout=cell_timeout or self.execution_limits.cell_seconds,
            command_timeout=command_timeout or self.execution_limits.command_seconds,
        )
        parsed = _parse_executed_notebook(
            result.notebook,
            expected_sources=self.cells,
            backend=self.backend.name,
        )
        new_outputs = [
            [json.loads(json.dumps(output)) for output in cell.get("outputs", [])]
            for cell in parsed.cells
        ]
        self.outputs = new_outputs
        self.artifacts = result.artifacts
        self._notebook = result.notebook
        self.run_count += 1
        return result

    def serialize_notebook(self) -> bytes:
        notebook = nbformat.v4.new_notebook()
        notebook.metadata.kernelspec = {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        }
        notebook.metadata.language_info = {"name": "python"}
        notebook.cells = [
            nbformat.v4.new_code_cell(
                source=source,
                outputs=(
                    [nbformat.from_dict(output) for output in self.outputs[index]]
                    if index < len(self.outputs)
                    else []
                ),
            )
            for index, source in enumerate(self.cells)
        ]
        return nbformat.writes(notebook, version=4).encode()

    def notebook_bytes(self) -> bytes:
        return bytes(self._notebook)

    async def list_workspace(self) -> str:
        generated = await self.environment.list_workspace() if self.environment else ()
        payload = {
            "inputs": [
                {
                    "path": f"../inputs/{item.name}",
                    "size": len(item.data),
                    "sql": _manifest_for(self.workspace, item.name).sql,
                    "source": _manifest_for(self.workspace, item.name).source,
                }
                for item in self.workspace.files
            ],
            "generated": [
                {
                    "path": item.path,
                    "size": item.size,
                    "media_type": item.media_type,
                }
                for item in generated
            ],
            "working_directory": "run/",
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        environment, self.environment = self.environment, None
        if environment is not None:
            await environment.close()

    def _staged_inputs(self) -> tuple[NotebookInput, ...]:
        if any(item.name == _MANIFEST_NAME for item in self.workspace.files):
            raise NotebookLimitExceeded(
                f"Workspace filename {_MANIFEST_NAME!r} is reserved",
                backend=self.backend.name,
                phase="input-validation",
            )
        manifest = [
            {
                "file": f"../inputs/{item.name}",
                "sql": _manifest_for(self.workspace, item.name).sql,
                "source": _manifest_for(self.workspace, item.name).source,
            }
            for item in self.workspace.files
        ]
        return (
            *self.workspace.files,
            NotebookInput(
                _MANIFEST_NAME,
                json.dumps(manifest, indent=2, sort_keys=True).encode(),
            ),
        )

    def _validate_sources(self, cells: list[str]) -> None:
        if len(cells) > MAX_CELLS:
            raise NotebookLimitExceeded(
                f"Notebook has more than {MAX_CELLS} cells",
                backend=self.backend.name,
                phase="notebook-validation",
            )
        total = 0
        for index, source in enumerate(cells):
            if not isinstance(source, str):
                raise TypeError("Notebook cell source must be a string")
            if len(source) > MAX_CELL_SOURCE_CHARS:
                raise NotebookLimitExceeded(
                    f"Cell {index} exceeds {MAX_CELL_SOURCE_CHARS} characters",
                    backend=self.backend.name,
                    phase="notebook-validation",
                )
            total += len(source)
        if total > MAX_TOTAL_SOURCE_CHARS:
            raise NotebookLimitExceeded(
                f"Notebook source exceeds {MAX_TOTAL_SOURCE_CHARS} characters",
                backend=self.backend.name,
                phase="notebook-validation",
            )


def _parse_executed_notebook(
    data: bytes,
    *,
    expected_sources: list[str],
    backend: str,
) -> nbformat.NotebookNode:
    try:
        notebook = nbformat.reads(data.decode(), as_version=4)
        nbformat.validate(notebook)
    except Exception as exc:
        raise NotebookInfrastructureError(
            "Execution backend returned an invalid notebook",
            backend=backend,
            phase="notebook-download",
        ) from exc
    sources = [cell.get("source", "") for cell in notebook.cells]
    if sources != expected_sources or any(
        cell.get("cell_type") != "code" for cell in notebook.cells
    ):
        raise NotebookInfrastructureError(
            "Execution backend changed notebook source or cell structure",
            backend=backend,
            phase="notebook-download",
        )
    return notebook


def _manifest_for(workspace: Workspace, name: str):
    return next(
        (entry for entry in workspace.manifest if entry.file == name),
        _EMPTY_MANIFEST,
    )


@dataclass(frozen=True, slots=True)
class _EmptyManifest:
    sql: None = None
    source: None = None


_EMPTY_MANIFEST = _EmptyManifest()
