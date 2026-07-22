"""Shared notebook and artifact validation helpers."""

from __future__ import annotations

import json
import mimetypes
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .base import (
    ArtifactInfo,
    ExecutionLimits,
    NotebookInfrastructureError,
    NotebookLimitExceeded,
    validate_artifact_path,
)

_EXCLUDED_NAMES = {"notebook.ipynb", "__pycache__"}


def validate_notebook_bytes(
    data: bytes,
    limits: ExecutionLimits,
    *,
    backend: str,
    phase: str,
) -> None:
    if len(data) > limits.max_notebook_bytes:
        raise NotebookLimitExceeded(
            f"Notebook exceeds {limits.max_notebook_bytes} bytes",
            backend=backend,
            phase=phase,
        )
    try:
        payload: Any = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotebookInfrastructureError(
            "Backend returned a malformed notebook",
            backend=backend,
            phase=phase,
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("nbformat") != 4
        or not isinstance(payload.get("cells"), list)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise NotebookInfrastructureError(
            "Backend returned an invalid nbformat v4 notebook",
            backend=backend,
            phase=phase,
        )


def should_include_artifact(path: PurePosixPath) -> bool:
    return not any(
        part.startswith(".") or part in _EXCLUDED_NAMES for part in path.parts
    )


def build_artifact_inventory(
    sizes: Mapping[str, int],
    limits: ExecutionLimits,
    *,
    backend: str,
) -> tuple[ArtifactInfo, ...]:
    artifacts: list[ArtifactInfo] = []
    total = 0
    for path_text in sorted(sizes):
        path = validate_artifact_path(path_text, backend=backend)
        if not should_include_artifact(path):
            continue
        size = sizes[path_text]
        if not isinstance(size, int) or size < 0:
            raise NotebookInfrastructureError(
                f"Backend returned an invalid artifact size for {path_text!r}",
                backend=backend,
                phase="artifact-inventory",
            )
        if size > limits.max_artifact_bytes:
            raise NotebookLimitExceeded(
                f"Artifact exceeds {limits.max_artifact_bytes} bytes: {path_text}",
                backend=backend,
                phase="artifact-inventory",
            )
        total += size
        if total > limits.max_total_artifact_bytes:
            raise NotebookLimitExceeded(
                f"Artifacts exceed {limits.max_total_artifact_bytes} total bytes",
                backend=backend,
                phase="artifact-inventory",
            )
        media_type, _ = mimetypes.guess_type(path_text)
        artifacts.append(ArtifactInfo(path_text, size, media_type))
        if len(artifacts) > limits.max_artifacts:
            raise NotebookLimitExceeded(
                f"Run produced more than {limits.max_artifacts} artifacts",
                backend=backend,
                phase="artifact-inventory",
            )
    return tuple(artifacts)
