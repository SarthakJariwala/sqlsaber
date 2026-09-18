"""CLI settings declaration shared by the saber editor and standalone CLI.

Importing this module must stay side-effect free and light: no capability
construction, no provider SDK imports, no prompts. Only ``bind`` (and the
credentialed branch of :func:`build_notebook_config`) loads runtime code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

from sqlsaber.plugin_settings import (
    PluginSettings,
    Setting,
    SettingsValues,
    configured_model,
    model_setting,
)

from .config import DEFAULT_NOTEBOOK_CONFIG, NotebookConfig, WorkspaceLimits
from .execution import DEFAULT_NOTEBOOK_BACKEND, NotebookBackend

BACKEND_CHOICES = ("docker", "microsandbox", "modal", "daytona", "e2b")

_DEFAULTS = DEFAULT_NOTEBOOK_CONFIG
_UNSET = object()


def _backend_is(expected: str) -> Callable[[SettingsValues], bool]:
    def active(values: SettingsValues) -> bool:
        return (values.get("backend") or DEFAULT_NOTEBOOK_BACKEND) == expected

    return active


_FIELDS = (
    Setting(
        name="backend",
        label="Execution backend",
        default=DEFAULT_NOTEBOOK_BACKEND,
        choices=BACKEND_CHOICES,
        env="SQLSABER_NOTEBOOK_BACKEND",
        help=(
            "Where analysis notebooks run. docker and microsandbox are local; "
            "modal, daytona, and e2b upload workspace files to that provider."
        ),
    ),
    Setting(
        name="image",
        label="Notebook image",
        env="SQLSABER_NOTEBOOK_IMAGE",
        help=(
            "Container image for notebook execution, preferably pinned by "
            "digest. Unset uses the default Jupyter scipy-notebook digest."
        ),
    ),
    model_setting(label="Analyst model"),
    Setting(
        name="e2b_api_key",
        label="E2B API key",
        kind="secret",
        env="E2B_API_KEY",
        when=_backend_is("e2b"),
        credential="e2b.api_key",
        help="Stored in the OS keyring. Unset uses E2B_API_KEY.",
    ),
    Setting(
        name="daytona_api_key",
        label="Daytona API key",
        kind="secret",
        env="DAYTONA_API_KEY",
        when=_backend_is("daytona"),
        credential="daytona.api_key",
        help=(
            "Stored in the OS keyring. Unset lets the Daytona SDK use its "
            "native environment configuration."
        ),
    ),
    Setting(
        name="daytona_api_url",
        label="Daytona API endpoint",
        env="DAYTONA_API_URL",
        when=_backend_is("daytona"),
        help="Unset uses DAYTONA_API_URL or the Daytona cloud default.",
    ),
    Setting(
        name="modal_token_id",
        label="Modal token ID",
        kind="secret",
        env="MODAL_TOKEN_ID",
        when=_backend_is("modal"),
        credential="modal.token_id",
        help=(
            "Stored in the OS keyring with the token secret. Unset both to "
            "use the native `modal setup` login."
        ),
    ),
    Setting(
        name="modal_token_secret",
        label="Modal token secret",
        kind="secret",
        env="MODAL_TOKEN_SECRET",
        when=_backend_is("modal"),
        credential="modal.token_secret",
        help="Stored in the OS keyring; required together with the token ID.",
    ),
    Setting(
        name="cpu_cores",
        label="CPU cores",
        kind="number",
        default=_DEFAULTS.cpu_cores,
        advanced=True,
        help="Daytona uses whole cores and requires at least 1.",
    ),
    Setting(
        name="memory_mb",
        label="Memory (MiB)",
        kind="integer",
        default=_DEFAULTS.memory_mb,
        advanced=True,
        help="Daytona allocates whole GiB and requires at least 1024.",
    ),
    Setting(
        name="image_prepare_seconds",
        label="Image preparation timeout (seconds)",
        kind="integer",
        default=_DEFAULTS.image_prepare_seconds,
        advanced=True,
        help="Bounds image pulls and remote sandbox provisioning.",
    ),
    Setting(
        name="open_seconds",
        label="Environment open timeout (seconds)",
        kind="integer",
        default=_DEFAULTS.open_seconds,
        advanced=True,
        help="Bounds environment startup and file transfer operations.",
    ),
    Setting(
        name="cell_seconds",
        label="Cell timeout (seconds)",
        kind="integer",
        default=_DEFAULTS.cell_seconds,
        advanced=True,
        help="Per-cell execution timeout; null disables the cell timer.",
    ),
    Setting(
        name="command_seconds",
        label="Notebook run timeout (seconds)",
        kind="integer",
        default=_DEFAULTS.command_seconds,
        advanced=True,
        help=(
            "Whole-notebook execution timeout; null (the default) disables "
            "it. Provider lifetime limits still apply."
        ),
    ),
    Setting(
        name="workspace_max_files",
        label="Workspace file count limit",
        kind="integer",
        default=_DEFAULTS.workspace.max_files,
        advanced=True,
        help="Maximum staged workspace files, excluding manifest.json.",
    ),
    Setting(
        name="workspace_max_file_bytes",
        label="Workspace per-file byte limit",
        kind="integer",
        default=_DEFAULTS.workspace.max_file_bytes,
        advanced=True,
        help="Maximum size of one staged workspace file.",
    ),
    Setting(
        name="workspace_max_total_bytes",
        label="Workspace total byte limit",
        kind="integer",
        default=_DEFAULTS.workspace.max_total_bytes,
        advanced=True,
        help="Maximum combined size of staged workspace files.",
    ),
    Setting(
        name="workspace_default_results",
        label="Default SQL results",
        kind="integer",
        default=_DEFAULTS.workspace.default_results,
        advanced=True,
        help="Recent execute_sql results staged when files is omitted.",
    ),
)


def _text(values: SettingsValues, name: str) -> str | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be text")
    return value


def _integer(values: SettingsValues, name: str, default: int) -> int:
    value = values.get(name, _UNSET)
    if value is _UNSET:
        return default
    if value is None:
        raise ValueError(f"{name} cannot be null; omit it to use the default")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _optional_seconds(
    values: SettingsValues, name: str, default: int | None
) -> int | None:
    value = values.get(name, _UNSET)
    if value is _UNSET:
        return default
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return value


def _number(values: SettingsValues, name: str, default: float) -> float:
    value = values.get(name, _UNSET)
    if value is _UNSET:
        return default
    if value is None:
        raise ValueError(f"{name} cannot be null; omit it to use the default")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _runtime_backend(
    values: SettingsValues,
    secrets: Mapping[str, str] | None,
) -> NotebookBackend | str | None:
    """Bind CLI-resolved credentials to a backend instance when present.

    Without applicable credentials the selector stays a plain name, so
    provider SDKs keep their native environment and login behavior.
    """

    name = _text(values, "backend")
    if secrets is None:
        return name
    selected = name or DEFAULT_NOTEBOOK_BACKEND
    if selected == "e2b" and (api_key := secrets.get("e2b_api_key")) is not None:
        from .execution.e2b import E2BNotebookBackend

        return E2BNotebookBackend(api_key=api_key)
    if selected == "daytona":
        api_key = secrets.get("daytona_api_key")
        api_url = _text(values, "daytona_api_url")
        if api_key is not None or api_url is not None:
            from .execution.daytona import DaytonaNotebookBackend

            return DaytonaNotebookBackend(api_key=api_key, api_url=api_url)
    if selected == "modal":
        token_id = secrets.get("modal_token_id")
        token_secret = secrets.get("modal_token_secret")
        if (token_id is None) != (token_secret is None):
            raise ValueError(
                "Modal credentials require both modal_token_id and "
                "modal_token_secret; set both, or neither to use the native "
                "`modal setup` login"
            )
        if token_id is not None and token_secret is not None:
            from .execution.modal import ModalNotebookBackend

            return ModalNotebookBackend(
                token_id=token_id,
                token_secret=token_secret,
            )
    return name


def build_notebook_config(
    values: SettingsValues,
    secrets: Mapping[str, str] | None = None,
) -> NotebookConfig:
    """Translate resolved CLI settings into the runtime NotebookConfig.

    Missing values keep the plugin-owned defaults. NotebookConfig validation
    applies unchanged and raises ValueError for invalid budgets.
    """

    workspace = WorkspaceLimits(
        max_files=_integer(
            values, "workspace_max_files", _DEFAULTS.workspace.max_files
        ),
        max_file_bytes=_integer(
            values, "workspace_max_file_bytes", _DEFAULTS.workspace.max_file_bytes
        ),
        max_total_bytes=_integer(
            values, "workspace_max_total_bytes", _DEFAULTS.workspace.max_total_bytes
        ),
        default_results=_integer(
            values, "workspace_default_results", _DEFAULTS.workspace.default_results
        ),
    )
    return NotebookConfig(
        workspace=workspace,
        backend=_runtime_backend(values, secrets),
        image=_text(values, "image"),
        image_prepare_seconds=_integer(
            values, "image_prepare_seconds", _DEFAULTS.image_prepare_seconds
        ),
        open_seconds=_integer(values, "open_seconds", _DEFAULTS.open_seconds),
        cell_seconds=_optional_seconds(values, "cell_seconds", _DEFAULTS.cell_seconds),
        command_seconds=_optional_seconds(
            values, "command_seconds", _DEFAULTS.command_seconds
        ),
        memory_mb=_integer(values, "memory_mb", _DEFAULTS.memory_mb),
        cpu_cores=_number(values, "cpu_cores", _DEFAULTS.cpu_cores),
        model=configured_model(values),
    )


def _validate(values: SettingsValues) -> None:
    build_notebook_config(values)


def _bind(values: SettingsValues, secrets: Mapping[str, str]) -> object:
    config = build_notebook_config(values, secrets)
    from .capability import capability

    return partial(capability, config=config)


_REMOTE_NOTICES = {
    "e2b": (
        "E2B notebooks run remotely: workspace files (SQL results and selected "
        "inputs) are uploaded to E2B. Template builds and sandboxes may incur "
        "usage charges. Sandboxes have a one-hour lifetime."
    ),
    "modal": (
        "Modal notebooks run remotely: workspace files (SQL results and "
        "selected inputs) are uploaded to Modal and sandbox runtime may incur "
        "charges. Without saved tokens, the native `modal setup` login is used."
    ),
    "daytona": (
        "Daytona notebooks run remotely: workspace files (SQL results and "
        "selected inputs) are uploaded to the configured Daytona service and "
        "sandboxes may incur usage charges."
    ),
}


def _notice(values: SettingsValues) -> str | None:
    return _REMOTE_NOTICES.get(str(values.get("backend") or DEFAULT_NOTEBOOK_BACKEND))


settings = PluginSettings(
    fields=_FIELDS,
    validate=_validate,
    bind=_bind,
    notice=_notice,
)
