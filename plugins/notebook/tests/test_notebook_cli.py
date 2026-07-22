from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqlsaber_notebook.cli import (
    _provider_from_model,
    _workspace_from_local_paths,
    _write_artifacts,
)
from sqlsaber_notebook.result import ArtifactRef


async def test_local_workspace_uses_safe_basenames_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_bytes(b"value\n1\n")
    workspace = await _workspace_from_local_paths([source])
    assert workspace.files[0].name == "data.csv"
    assert workspace.files[0].data == b"value\n1\n"
    assert workspace.manifest[0].source == "local file"


async def test_local_workspace_rejects_duplicate_basenames(tmp_path: Path) -> None:
    first = tmp_path / "one" / "data.csv"
    second = tmp_path / "two" / "data.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("one")
    second.write_text("two")
    with pytest.raises(ValueError, match="Duplicate"):
        await _workspace_from_local_paths([first, second])


def test_artifact_writer_creates_manifest(tmp_path: Path) -> None:
    target = tmp_path / "artifacts"
    _write_artifacts(
        target,
        [b"png"],
        [ArtifactRef("nested/result.txt", b"answer", "text/plain")],
        False,
    )
    assert (target / "plot_1.png").read_bytes() == b"png"
    assert (target / "nested/result.txt").read_bytes() == b"answer"
    manifest = json.loads((target / "manifest.json").read_text())
    assert [entry["file"] for entry in manifest] == [
        "plot_1.png",
        "nested/result.txt",
    ]


def test_provider_requires_prefixed_model() -> None:
    assert _provider_from_model("anthropic:claude") == "anthropic"
    with pytest.raises(ValueError, match="provider:model"):
        _provider_from_model("claude")
