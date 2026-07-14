"""Standalone notebook analyst command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from ._shared import (
    MAX_WORKSPACE_FILE_BYTES,
    MAX_WORKSPACE_FILES,
    MAX_WORKSPACE_TOTAL_BYTES,
)
from .analyst import analyze, supports_notebook_images
from .execution import NotebookExecutionError, NotebookInput
from .execution.base import validate_artifact_path
from .result import ArtifactRef, ManifestEntry, Workspace


async def _main_async(
    *,
    goal: str,
    paths: list[Path],
    model: str,
    backend: str,
    output: Path,
    overwrite: bool,
) -> None:
    workspace = await _workspace_from_local_paths(paths)
    provider = _provider_from_model(model)
    if output.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output}")
    artifact_dir = output.parent / f"{output.stem}_artifacts"
    if artifact_dir.exists() and not overwrite:
        raise ValueError(f"Artifact directory already exists: {artifact_dir}")

    result = await analyze(
        goal,
        workspace,
        model=model,
        model_provider=provider,
        backend=backend,
        include_snapshot_images=supports_notebook_images(model, provider),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(output.write_bytes, result.notebook)
    await asyncio.to_thread(
        _write_artifacts,
        artifact_dir,
        result.images,
        result.files,
        overwrite,
    )
    print(result.answer)
    print(f"\nNotebook: {output}", file=sys.stderr)
    if result.images or result.files:
        print(f"Artifacts: {artifact_dir}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sqlsaber-notebook",
        description="Analyze local data files with a fresh-kernel notebook subagent.",
    )
    parser.add_argument("goal", help="Natural-language analysis goal")
    parser.add_argument("paths", nargs="+", type=Path, help="Input data files")
    parser.add_argument(
        "--model",
        default=os.getenv("SQLSABER_NOTEBOOK_MODEL"),
        help="Pydantic AI model string (or SQLSABER_NOTEBOOK_MODEL)",
    )
    parser.add_argument(
        "--backend",
        choices=("docker", "modal"),
        default=os.getenv("SQLSABER_NOTEBOOK_BACKEND", "docker"),
    )
    parser.add_argument("--output", type=Path, default=Path("analysis.ipynb"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or SQLSABER_NOTEBOOK_MODEL is required")
    try:
        asyncio.run(
            _main_async(
                goal=args.goal,
                paths=args.paths,
                model=args.model,
                backend=args.backend,
                output=args.output,
                overwrite=args.overwrite,
            )
        )
    except (NotebookExecutionError, TimeoutError, ValueError, OSError) as exc:
        parser.exit(2, f"sqlsaber-notebook: error: {exc}\n")


async def _workspace_from_local_paths(paths: list[Path]) -> Workspace:
    files: list[NotebookInput] = []
    manifest: list[ManifestEntry] = []
    names: set[str] = set()
    total = 0
    if len(paths) > MAX_WORKSPACE_FILES:
        raise ValueError(f"Too many input files; maximum is {MAX_WORKSPACE_FILES}")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Input is not a regular file: {path}")
        name = path.name
        if name in names or name == "manifest.json":
            raise ValueError(f"Duplicate or reserved input filename: {name}")
        size = path.stat().st_size
        if size > MAX_WORKSPACE_FILE_BYTES:
            raise ValueError(f"Input exceeds {MAX_WORKSPACE_FILE_BYTES} bytes: {path}")
        total += size
        if total > MAX_WORKSPACE_TOTAL_BYTES:
            raise ValueError(f"Inputs exceed {MAX_WORKSPACE_TOTAL_BYTES} total bytes")
        data = await asyncio.to_thread(path.read_bytes)
        files.append(NotebookInput(name, data))
        manifest.append(ManifestEntry(name, source="local file"))
        names.add(name)
    return Workspace(tuple(files), tuple(manifest))


def _provider_from_model(model: str) -> str:
    provider, separator, _ = model.partition(":")
    if not separator or not provider:
        raise ValueError("Model must use the 'provider:model' format")
    return provider.lower()


def _write_artifacts(
    artifact_dir: Path,
    images: list[bytes],
    files: list[ArtifactRef],
    overwrite: bool,
) -> None:
    if artifact_dir.exists() and overwrite:
        import shutil

        shutil.rmtree(artifact_dir)
    if not images and not files:
        return
    artifact_dir.mkdir(parents=True)
    entries: list[dict[str, object]] = []
    for index, image in enumerate(images, start=1):
        name = f"plot_{index}.png"
        (artifact_dir / name).write_bytes(image)
        entries.append({"file": name, "media_type": "image/png", "size": len(image)})
    for artifact in files:
        relative = validate_artifact_path(artifact.name, backend="standalone")
        path = artifact_dir.joinpath(*relative.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(artifact.data)
        entries.append(
            {
                "file": artifact.name,
                "media_type": artifact.media_type,
                "size": len(artifact.data),
            }
        )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
