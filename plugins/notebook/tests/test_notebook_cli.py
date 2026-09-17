from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import keyring
import pytest

from sqlsaber.config.plugins import PluginConfigStore, SavedPlugin, save_secret
import sqlsaber_notebook.cli as cli
from sqlsaber_notebook.cli import (
    _provider_from_model,
    _workspace_from_local_paths,
    _write_artifacts,
)
from sqlsaber_notebook.result import AnalysisResult, ArtifactRef
from sqlsaber_notebook.settings import settings


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


@pytest.fixture
def plugin_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> PluginConfigStore:
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(tmp_path / "config")
    )
    for field in settings.fields:
        if field.env:
            monkeypatch.delenv(field.env, raising=False)
    credentials: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        keyring,
        "set_password",
        lambda service, key, value: credentials.update({(service, key): value}),
    )
    monkeypatch.setattr(
        keyring, "get_password", lambda service, key: credentials.get((service, key))
    )
    return PluginConfigStore()


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    argv: list[str],
    *,
    paths: int = 1,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_analyze(goal: str, workspace: object, **kwargs: Any) -> object:
        captured["goal"] = goal
        captured["workspace"] = workspace
        captured["kwargs"] = kwargs
        return AnalysisResult(
            answer="done", notebook=b"{}", images=[], files=[], provenance=[]
        )

    monkeypatch.setattr(cli, "analyze", fake_analyze)
    inputs = []
    for index in range(paths):
        data = tmp_path / f"data_{index}.csv"
        data.write_text("value\n1\n")
        inputs.append(str(data))
    output = tmp_path / "out.ipynb"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sqlsaber-notebook",
            "analyze the data",
            *inputs,
            "--model",
            "anthropic:claude",
            "--output",
            str(output),
            "--overwrite",
            *argv,
        ],
    )
    cli.main()
    return captured


@pytest.mark.parametrize(
    ("saved_backend", "environment", "flag", "expected"),
    [
        (None, None, None, "docker"),
        ("microsandbox", None, None, "microsandbox"),
        ("microsandbox", "modal", None, "modal"),
        ("microsandbox", "modal", "daytona", "daytona"),
    ],
)
def test_main_resolves_real_saved_settings_and_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plugin_store: PluginConfigStore,
    saved_backend: str | None,
    environment: str | None,
    flag: str | None,
    expected: str,
) -> None:
    plugin_store.save(
        "notebook",
        SavedPlugin(False, {"backend": saved_backend} if saved_backend else {}),
    )
    before = plugin_store.path.read_bytes()
    if environment:
        monkeypatch.setenv("SQLSABER_NOTEBOOK_BACKEND", environment)
    captured = _run_cli(monkeypatch, tmp_path, ["--backend", flag] if flag else [])
    assert captured["kwargs"]["config"].backend == expected
    assert plugin_store.path.read_bytes() == before
    assert (tmp_path / "out.ipynb").read_bytes() == b"{}"
    assert capsys.readouterr().out.strip() == "done"


def test_main_binds_store_credentials_without_environ_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, plugin_store: PluginConfigStore
) -> None:
    from sqlsaber_notebook.execution.daytona import DaytonaNotebookBackend

    plugin_store.save(
        "notebook",
        SavedPlugin(
            settings={"backend": "daytona", "daytona_api_url": "https://d.example/api"}
        ),
    )
    save_secret("notebook", settings.field("daytona_api_key"), "key-123")
    environ_before = dict(os.environ)
    captured = _run_cli(monkeypatch, tmp_path, [])
    backend = captured["kwargs"]["config"].backend
    assert isinstance(backend, DaytonaNotebookBackend)
    assert backend._api_key == "key-123"
    assert backend._api_url == "https://d.example/api"
    assert dict(os.environ) == environ_before


def test_main_rejects_partial_modal_pair_from_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plugin_store: PluginConfigStore,
) -> None:
    plugin_store.save("notebook", SavedPlugin(settings={"backend": "modal"}))
    save_secret("notebook", settings.field("modal_token_id"), "ak-1")
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, [])
    assert excinfo.value.code == 2
    assert "modal_token_secret" in capsys.readouterr().err


def test_main_applies_saved_workspace_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    plugin_store: PluginConfigStore,
) -> None:
    plugin_store.save(
        "notebook",
        SavedPlugin(settings={"backend": "docker", "workspace_max_files": 1}),
    )
    with pytest.raises(SystemExit) as excinfo:
        _run_cli(monkeypatch, tmp_path, [], paths=2)
    assert excinfo.value.code == 2
    assert "maximum is 1" in capsys.readouterr().err
