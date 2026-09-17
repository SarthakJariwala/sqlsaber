"""Standalone notebook analyst command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from sqlsaber.plugin_settings import SettingValue

from .analyst import analyze, supports_notebook_images
from .config import DEFAULT_NOTEBOOK_CONFIG, NotebookConfig, WorkspaceLimits
from .execution import NotebookExecutionError, NotebookInput
from .execution.base import validate_artifact_path
from .result import ArtifactRef, ManifestEntry, Workspace
from .settings import BACKEND_CHOICES, build_notebook_config, settings

_PLUGIN_NAME = "notebook"


async def _main_async(
    *,
    goal: str,
    paths: list[Path],
    model: str,
    config: NotebookConfig,
    output: Path,
    overwrite: bool,
) -> None:
    workspace = await _workspace_from_local_paths(paths, limits=config.workspace)
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
        config=config,
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
        choices=BACKEND_CHOICES,
        default=None,
        help=(
            "Execution backend; defaults to SQLSABER_NOTEBOOK_BACKEND, then "
            "settings saved with the saber CLI, then docker"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("analysis.ipynb"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or SQLSABER_NOTEBOOK_MODEL is required")
    try:
        config = _notebook_config(args.backend)
        asyncio.run(
            _main_async(
                goal=args.goal,
                paths=args.paths,
                model=args.model,
                config=config,
                output=args.output,
                overwrite=args.overwrite,
            )
        )
    except (NotebookExecutionError, TimeoutError, ValueError, OSError) as exc:
        parser.exit(2, f"sqlsaber-notebook: error: {exc}\n")


def _notebook_config(explicit_backend: str | None) -> NotebookConfig:
    """Resolve the notebook settings saved and shared with the saber CLI.

    Standalone runs are explicit invocations, so the plugin's enabled flag,
    which only governs automatic loading into saber, is ignored. Precedence is
    the explicit --backend flag, then environment variables, then saved
    settings, then plugin defaults.
    """

    from sqlsaber.config.plugins import PluginConfigStore, resolve_settings

    overrides: dict[str, SettingValue] | None = None
    if explicit_backend is not None:
        overrides = {"backend": explicit_backend}
    saved = PluginConfigStore().get(_PLUGIN_NAME).settings
    resolved = resolve_settings(_PLUGIN_NAME, settings, saved, overrides=overrides)
    return build_notebook_config(resolved.values, resolved.secrets)


async def _workspace_from_local_paths(
    paths: list[Path],
    *,
    limits: WorkspaceLimits = DEFAULT_NOTEBOOK_CONFIG.workspace,
) -> Workspace:
    files: list[NotebookInput] = []
    manifest: list[ManifestEntry] = []
    names: set[str] = set()
    total = 0
    if len(paths) > limits.max_files:
        raise ValueError(f"Too many input files; maximum is {limits.max_files}")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Input is not a regular file: {path}")
        name = path.name
        if name in names or name == "manifest.json":
            raise ValueError(f"Duplicate or reserved input filename: {name}")
        size = path.stat().st_size
        if size > limits.max_file_bytes:
            raise ValueError(f"Input exceeds {limits.max_file_bytes} bytes: {path}")
        total += size
        if total > limits.max_total_bytes:
            raise ValueError(f"Inputs exceed {limits.max_total_bytes} total bytes")
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
