"""CLI settings declarations and binding stay lazy and provider-specific."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from sqlsaber_sandbox.config import MIB
from sqlsaber_sandbox.settings import settings


def test_settings_declare_explicit_provider_specific_surfaces() -> None:
    provider = settings.field("provider")
    assert provider.required is True
    assert provider.env == "SQLSABER_SANDBOX_PROVIDER"
    assert provider.choices == (
        "docker",
        "microsandbox",
        "e2b",
        "modal",
        "daytona",
        "sprites",
    )

    e2b = {"provider": "e2b"}
    daytona = {"provider": "daytona"}
    sprites = {"provider": "sprites"}
    assert settings.field("e2b_template").active(e2b)
    assert not settings.field("daytona_image").active(e2b)
    assert settings.field("daytona_image").active(daytona)
    assert not settings.field("cpu_cores").active(e2b)
    assert settings.field("cpu_cores").active(sprites)
    assert not settings.field("gpu").active(sprites)

    credentials = {
        field.name: (field.credential, field.env)
        for field in settings.fields
        if field.kind == "secret"
    }
    assert credentials == {
        "e2b_api_key": ("e2b.api_key", "E2B_API_KEY"),
        "daytona_api_key": ("daytona.api_key", "DAYTONA_API_KEY"),
        "modal_token_id": ("modal.token_id", "MODAL_TOKEN_ID"),
        "modal_token_secret": ("modal.token_secret", "MODAL_TOKEN_SECRET"),
        "sprites_token": ("sprites.token", "SPRITES_TOKEN"),
    }
    assert settings.field("daytona_api_url").env == "DAYTONA_API_URL"
    assert all(
        settings.field(name).advanced
        for name in (
            "cpu_cores",
            "memory_mb",
            "gpu",
            "open_seconds",
            "transport_seconds",
            "cell_seconds",
            "idle_seconds",
            "max_lifetime_seconds",
            "workspace_max_files",
            "workspace_max_file_bytes",
            "workspace_max_total_bytes",
            "workspace_max_manifest_bytes",
            "workspace_default_results",
        )
    )


def test_binding_builds_config_and_defers_backend_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "provider": "daytona",
        "daytona_image": "registry.example/analysis:3",
        "daytona_api_url": "https://daytona.example/api",
        "cpu_cores": 4.0,
        "memory_mb": 8192,
        "gpu": "2",
        "open_seconds": 90,
        "transport_seconds": 12,
        "cell_seconds": 45.5,
        "idle_seconds": 300.0,
        "max_lifetime_seconds": 1800,
        "workspace_max_files": 7,
        "workspace_max_file_bytes": 8 * MIB,
        "workspace_max_total_bytes": 21 * MIB,
        "workspace_max_manifest_bytes": 64_000,
        "workspace_default_results": 3,
    }
    secrets = {"daytona_api_key": "saved-daytona-key"}
    environment = {
        "E2B_API_KEY": "unchanged-e2b-key",
        "DAYTONA_API_KEY": "unchanged-daytona-key",
        "DAYTONA_API_URL": "https://unchanged.daytona/api",
        "MODAL_TOKEN_ID": "unchanged-modal-id",
        "MODAL_TOKEN_SECRET": "unchanged-modal-secret",
        "SPRITES_TOKEN": "unchanged-sprites-token",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls: list[tuple[str | None, dict[str, Any]]] = []
    backend = object()

    def create_backend(provider: str | None, **kwargs: Any) -> object:
        calls.append((provider, kwargs))
        return backend

    import sqlsaber_sandbox.backends as backends

    monkeypatch.setattr(backends, "create_backend", create_backend)
    factory = settings.bind(values, secrets)
    assert calls == []
    capability = factory(SimpleNamespace(workspace_input_resolver=None))
    assert calls == []

    config = capability.tool.config
    assert config.provider == "daytona"
    assert config.image == "registry.example/analysis:3"
    assert config.cpu_cores == 4
    assert config.memory_mb == 8192
    assert config.gpu == "2"
    assert config.open_seconds == 90
    assert config.transport_seconds == 12
    assert config.cell_seconds == 45.5
    assert config.idle_seconds == 300
    assert config.max_lifetime_seconds == 1800
    assert config.workspace.max_files == 7
    assert config.workspace.max_file_bytes == 8 * MIB
    assert config.workspace.max_total_bytes == 21 * MIB
    assert config.workspace.max_manifest_bytes == 64_000
    assert config.workspace.default_results == 3

    assert capability.tool._backend_factory is not None
    assert capability.tool._backend_factory() is backend
    assert calls == [
        (
            "daytona",
            {
                "e2b_api_key": None,
                "daytona_api_key": "saved-daytona-key",
                "daytona_api_url": "https://daytona.example/api",
                "modal_token_id": None,
                "modal_token_secret": None,
                "sprites_token": None,
            },
        )
    ]
    assert {name: os.environ[name] for name in environment} == environment


def test_settings_reuse_runtime_defaults_and_validation() -> None:
    factory = settings.bind({"provider": "docker"}, {})
    capability = factory(SimpleNamespace(workspace_input_resolver=None))
    config = capability.tool.config

    from sqlsaber.nested_model import INHERIT

    assert config.open_seconds == 180
    assert config.transport_seconds == 30
    assert config.cell_seconds == 600
    assert config.workspace.max_files == 50
    assert config.workspace.max_total_bytes == 250 * MIB
    assert config.model is INHERIT
    assert capability.tool._backend_factory is None

    with pytest.raises(ValueError, match="open_seconds"):
        settings.validate({"provider": "docker", "open_seconds": 0})
    with pytest.raises(ValueError, match="provider must be one of"):
        settings.validate({})


@pytest.mark.parametrize(
    ("provider", "field"),
    [
        ("docker", "docker_image"),
        ("microsandbox", "microsandbox_image"),
        ("e2b", "e2b_template"),
        ("modal", "modal_image"),
        ("daytona", "daytona_image"),
    ],
)
def test_provider_image_or_template_maps_to_runtime_image(
    provider: str, field: str
) -> None:
    factory = settings.bind({"provider": provider, field: "provider-image"}, {})
    capability = factory(SimpleNamespace(workspace_input_resolver=None))

    assert capability.tool.config.image == "provider-image"


def test_modal_tokens_are_optional_but_must_be_paired() -> None:
    factory = settings.bind({"provider": "modal"}, {})
    capability = factory(SimpleNamespace(workspace_input_resolver=None))
    assert capability.tool._backend_factory is None

    with pytest.raises(ValueError, match="configured together"):
        settings.bind({"provider": "modal"}, {"modal_token_id": "only-id"})


def test_remote_notice_discloses_uploads_charges_and_modal_native_auth() -> None:
    assert settings.notice({"provider": "docker"}) is None
    e2b = settings.notice({"provider": "e2b"})
    assert e2b is not None and "uploaded" in e2b and "charges" in e2b
    modal = settings.notice({"provider": "modal"})
    assert modal is not None and "modal token new" in modal
