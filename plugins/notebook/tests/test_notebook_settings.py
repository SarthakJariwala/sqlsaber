from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sqlsaber_notebook.config import DEFAULT_NOTEBOOK_CONFIG
from sqlsaber_notebook.execution import DEFAULT_NOTEBOOK_BACKEND
from sqlsaber_notebook.settings import (
    BACKEND_CHOICES,
    build_notebook_config,
    settings,
)

_HEAVY_MODULES = (
    "pydantic_ai",
    "modal",
    "daytona",
    "e2b",
    "microsandbox",
    "sqlsaber_notebook.capability",
    "sqlsaber_notebook.analyst",
)


def test_settings_declaration_is_import_light() -> None:
    script = (
        "import sys\n"
        "import sqlsaber_notebook.settings\n"
        f"banned = [name for name in {_HEAVY_MODULES!r} if name in sys.modules]\n"
        "print(','.join(banned))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_backend_selector_and_env_aliases_match_runtime_defaults() -> None:
    backend = settings.field("backend")
    assert backend.default == DEFAULT_NOTEBOOK_BACKEND
    assert backend.choices == BACKEND_CHOICES
    assert backend.env == "SQLSABER_NOTEBOOK_BACKEND"
    assert backend.when is None
    assert settings.field("image").env == "SQLSABER_NOTEBOOK_IMAGE"
    assert settings.field("memory_mb").default == DEFAULT_NOTEBOOK_CONFIG.memory_mb
    assert settings.field("cpu_cores").default == DEFAULT_NOTEBOOK_CONFIG.cpu_cores
    assert (
        settings.field("workspace_max_files").default
        == DEFAULT_NOTEBOOK_CONFIG.workspace.max_files
    )
    assert settings.field("command_seconds").default is None


def test_credential_fields_use_shared_keyring_identifiers() -> None:
    assert settings.field("daytona_api_key").credential == "daytona.api_key"
    assert settings.field("modal_token_id").credential == "modal.token_id"
    assert settings.field("modal_token_secret").credential == "modal.token_secret"
    for name in ("daytona_api_key", "modal_token_id", "modal_token_secret"):
        assert settings.field(name).kind == "secret"


def test_provider_fields_follow_unconditional_backend_selector() -> None:
    daytona_key = settings.field("daytona_api_key")
    modal_token = settings.field("modal_token_id")
    assert daytona_key.active({"backend": "daytona"})
    assert not daytona_key.active({"backend": "docker"})
    assert not daytona_key.active({})
    assert modal_token.active({"backend": "modal"})
    assert not modal_token.active({"backend": "daytona"})


def test_validate_reuses_notebook_config_validation() -> None:
    settings.validate({"backend": "docker"})
    settings.validate({})
    with pytest.raises(ValueError, match="memory_mb"):
        settings.validate({"backend": "docker", "memory_mb": 0})
    with pytest.raises(ValueError, match="cpu_cores"):
        settings.validate({"cpu_cores": -1})


def test_build_config_defaults_match_plugin_defaults() -> None:
    config = build_notebook_config({}, {})
    assert config.backend is None
    assert config.image is None
    assert config.memory_mb == DEFAULT_NOTEBOOK_CONFIG.memory_mb
    assert config.workspace == DEFAULT_NOTEBOOK_CONFIG.workspace
    assert config.cell_seconds == DEFAULT_NOTEBOOK_CONFIG.cell_seconds


def test_build_config_applies_values_and_null_timers() -> None:
    config = build_notebook_config(
        {
            "backend": "docker",
            "image": "registry/image@sha256:digest",
            "cell_seconds": None,
            "command_seconds": 120,
            "memory_mb": 4096,
            "cpu_cores": 2,
            "workspace_max_files": 3,
        },
        {},
    )
    assert config.backend == "docker"
    assert config.image == "registry/image@sha256:digest"
    assert config.cell_seconds is None
    assert config.command_seconds == 120
    assert config.memory_mb == 4096
    assert config.cpu_cores == 2.0
    assert config.workspace.max_files == 3


def test_build_config_rejects_explicit_null_for_nonnullable_numbers() -> None:
    # Only a missing key means "use the default"; the two execution timers
    # are the only numeric fields where null is a valid stored value.
    for name in (
        "memory_mb",
        "cpu_cores",
        "image_prepare_seconds",
        "workspace_max_files",
    ):
        with pytest.raises(ValueError, match=f"{name} cannot be null"):
            build_notebook_config({name: None}, {})


def test_build_config_without_secrets_keeps_plain_backend_names() -> None:
    for name in ("modal", "daytona"):
        config = build_notebook_config({"backend": name}, {})
        assert config.backend == name


def test_build_config_binds_daytona_credentials_without_environ_mutation() -> None:
    from sqlsaber_notebook.execution.daytona import DaytonaNotebookBackend

    environ_before = dict(os.environ)
    config = build_notebook_config(
        {"backend": "daytona", "daytona_api_url": "https://daytona.example/api"},
        {"daytona_api_key": "key-123"},
    )
    backend = config.backend
    assert isinstance(backend, DaytonaNotebookBackend)
    assert backend._api_key == "key-123"
    assert backend._api_url == "https://daytona.example/api"
    assert dict(os.environ) == environ_before


def test_build_config_binds_modal_token_pair() -> None:
    from sqlsaber_notebook.execution.modal import ModalNotebookBackend

    config = build_notebook_config(
        {"backend": "modal"},
        {"modal_token_id": "ak-token", "modal_token_secret": "as-secret"},
    )
    backend = config.backend
    assert isinstance(backend, ModalNotebookBackend)
    assert backend._token_id == "ak-token"
    assert backend._token_secret == "as-secret"


def test_build_config_rejects_partial_modal_token_pair() -> None:
    with pytest.raises(ValueError, match="modal_token_secret"):
        build_notebook_config({"backend": "modal"}, {"modal_token_id": "ak-token"})


def test_build_config_ignores_other_backend_credentials() -> None:
    config = build_notebook_config(
        {"backend": "docker"},
        {"daytona_api_key": "key-123"},
    )
    assert config.backend == "docker"


def test_bind_returns_capability_factory_with_resolved_config() -> None:
    from sqlsaber_notebook.capability import Notebook

    factory = settings.bind(
        {"backend": "docker", "memory_mb": 4096, "cell_seconds": None},
        {},
    )
    assert callable(factory)
    capability = factory(SimpleNamespace(workspace_input_resolver=None))
    assert isinstance(capability, Notebook)
    config = capability.tool._config
    assert config.backend == "docker"
    assert config.memory_mb == 4096
    assert config.cell_seconds is None


def test_bind_passes_configured_model_to_capability() -> None:
    factory = settings.bind({"backend": "docker", "model": "openai:gpt-5-mini"}, {})
    capability = factory(SimpleNamespace(workspace_input_resolver=None))

    assert capability.tool._model_name == "openai:gpt-5-mini"


def test_model_setting_inherits_session_when_unset() -> None:
    assert settings.field("model").env == "SQLSABER_NOTEBOOK_MODEL"
    factory = settings.bind({"backend": "docker"}, {})
    capability = factory(SimpleNamespace(workspace_input_resolver=None))

    assert capability.tool._model_name is None


def test_notice_describes_remote_upload_and_native_login() -> None:
    assert settings.notice({"backend": "docker"}) is None
    assert settings.notice({}) is None
    assert settings.notice({"backend": "microsandbox"}) is None
    modal_notice = settings.notice({"backend": "modal"})
    assert modal_notice is not None
    assert "uploaded" in modal_notice
    assert "charges" in modal_notice
    assert "modal setup" in modal_notice
    daytona_notice = settings.notice({"backend": "daytona"})
    assert daytona_notice is not None
    assert "uploaded" in daytona_notice
    assert "charges" in daytona_notice
