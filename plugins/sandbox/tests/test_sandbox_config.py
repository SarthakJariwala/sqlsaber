"""Sandbox configuration and value-object admission tests."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from sqlsaber_sandbox.config import MIB, SandboxConfig, WorkspaceLimits
from sqlsaber_sandbox.result import Workspace, WorkspaceFile


def test_sandbox_config_defaults_match_the_public_contract() -> None:
    config = SandboxConfig()

    assert config == SandboxConfig(
        provider=None,
        image=None,
        cpu_cores=None,
        memory_mb=None,
        workspace=WorkspaceLimits(),
        open_seconds=180,
        transport_seconds=30,
        cell_seconds=600,
        idle_seconds=None,
        max_lifetime_seconds=None,
        max_artifacts=50,
        max_artifact_bytes=50 * MIB,
        max_total_artifact_bytes=200 * MIB,
        max_output_chars=16_000,
        max_image_bytes=4 * MIB,
        max_history_image_bytes=24 * MIB,
    )
    assert not hasattr(config, "max_iterations")
    assert not hasattr(config, "max_model_requests")
    assert not hasattr(config, "max_source_chars")


@pytest.mark.parametrize(
    "field",
    [
        "memory_mb",
        "open_seconds",
        "transport_seconds",
        "max_lifetime_seconds",
        "max_artifacts",
        "max_artifact_bytes",
        "max_total_artifact_bytes",
        "max_output_chars",
        "max_image_bytes",
        "max_history_image_bytes",
    ],
)
def test_integer_budgets_reject_bool(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SandboxConfig(**{field: True})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("field", ["cpu_cores", "cell_seconds", "idle_seconds"])
@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan")])
def test_numeric_budgets_require_finite_positive_values(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        SandboxConfig(**{field: value})  # ty: ignore[invalid-argument-type]


def test_optional_resources_and_timers_accept_none_without_cross_field_clamps() -> None:
    config = SandboxConfig(
        provider="future-provider",
        image="custom-image",
        cpu_cores=None,
        memory_mb=None,
        cell_seconds=None,
        idle_seconds=None,
        max_lifetime_seconds=None,
        max_artifact_bytes=1,
        max_total_artifact_bytes=1,
        max_image_bytes=2,
        max_history_image_bytes=3,
    )

    assert config.provider == "future-provider"
    assert config.cell_seconds is None
    assert config.max_image_bytes > config.max_artifact_bytes


@pytest.mark.parametrize("field", ["provider", "image"])
def test_string_selectors_reject_blank_values(field: str) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        SandboxConfig(**{field: "  "})  # ty: ignore[invalid-argument-type]


def test_config_and_workspace_provenance_are_frozen_snapshots() -> None:
    provenance = {"query": "select 1"}
    workspace_file = WorkspaceFile("result_rows.json", b"rows", provenance=provenance)
    provenance["query"] = "changed"

    assert workspace_file.provenance == {"query": "select 1"}
    with pytest.raises(TypeError):
        workspace_file.provenance["query"] = "changed"  # ty: ignore[invalid-assignment]
    with pytest.raises(FrozenInstanceError):
        SandboxConfig().image = "replacement"  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "manifest.json",
        "../escape.csv",
        "nested/file.csv",
        "windows\\file.csv",
        "control\n.csv",
        "é" * 128,
    ],
)
def test_workspace_file_requires_a_safe_flat_255_byte_name(name: str) -> None:
    with pytest.raises(ValueError, match="Unsafe workspace filename"):
        WorkspaceFile(name, b"data")

    assert WorkspaceFile("a" * 255, b"data").name == "a" * 255


def test_workspace_manifest_uses_stable_relative_paths_and_provenance() -> None:
    workspace = Workspace.from_files(
        [
            WorkspaceFile(
                "result_rows.json",
                b"{}",
                media_type="application/json",
                provenance={"query": "select * from rows"},
            ),
            ("notes.txt", b"notes"),
        ]
    )

    assert workspace.manifest_bytes() == workspace.manifest_bytes()
    assert json.loads(workspace.manifest_bytes()) == [
        {
            "file": "../inputs/result_rows.json",
            "media_type": "application/json",
            "provenance": {"query": "select * from rows"},
        },
        {
            "file": "../inputs/notes.txt",
            "media_type": None,
            "provenance": {},
        },
    ]


def test_workspace_limits_enforce_count_sizes_duplicates_and_manifest() -> None:
    first = WorkspaceFile("first.bin", b"123")
    second = WorkspaceFile("second.bin", b"4567")

    with pytest.raises(ValueError, match="maximum is 1"):
        WorkspaceLimits(max_files=1).validate((first, second))
    with pytest.raises(ValueError, match="exceeds 3 bytes"):
        WorkspaceLimits(max_file_bytes=3).validate((second,))
    with pytest.raises(ValueError, match="total bytes"):
        WorkspaceLimits(max_total_bytes=6).validate((first, second))
    with pytest.raises(ValueError, match="Duplicate workspace filename"):
        WorkspaceLimits().validate((first, WorkspaceFile("first.bin", b"other")))
    with pytest.raises(ValueError, match="manifest exceeds 2 bytes"):
        WorkspaceLimits(max_manifest_bytes=2).validate((first,))


@pytest.mark.parametrize("field", WorkspaceLimits.__dataclass_fields__)
def test_workspace_limits_reject_bool(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        WorkspaceLimits(**{field: True})
