"""Side-effect-free CLI settings declaration for the sandbox capability."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

from sqlsaber.plugin_settings import (
    PluginSettings,
    Setting,
    SettingsValues,
    SettingValue,
    configured_model,
    model_setting,
)

from .config import DEFAULT_SANDBOX_CONFIG, SandboxConfig, WorkspaceLimits

if TYPE_CHECKING:
    from sqlsaber.capabilities.plugins import PluginContext

    from .capability import Sandbox

_PROVIDERS = ("docker", "microsandbox", "e2b", "modal", "daytona", "sprites")
_REMOTE_PROVIDERS = frozenset({"e2b", "modal", "daytona", "sprites"})
_IMAGE_FIELDS = {
    "docker": "docker_image",
    "microsandbox": "microsandbox_image",
    "e2b": "e2b_template",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _when_provider(*providers: str) -> Callable[[SettingsValues], bool]:
    selected = frozenset(providers)
    return lambda values: values.get("provider") in selected


def _resolved(values: SettingsValues, name: str, default: SettingValue) -> SettingValue:
    return values[name] if name in values else default


def _sandbox_config(values: SettingsValues) -> SandboxConfig:
    defaults = DEFAULT_SANDBOX_CONFIG
    provider = cast(str | None, values.get("provider"))
    if provider not in _PROVIDERS:
        raise ValueError(f"provider must be one of: {', '.join(_PROVIDERS)}")
    image_field = _IMAGE_FIELDS.get(provider)
    image = cast(str | None, values.get(image_field)) if image_field else None
    workspace_defaults = defaults.workspace
    workspace = WorkspaceLimits(
        max_files=cast(
            int,
            _resolved(values, "workspace_max_files", workspace_defaults.max_files),
        ),
        max_file_bytes=cast(
            int,
            _resolved(
                values,
                "workspace_max_file_bytes",
                workspace_defaults.max_file_bytes,
            ),
        ),
        max_total_bytes=cast(
            int,
            _resolved(
                values,
                "workspace_max_total_bytes",
                workspace_defaults.max_total_bytes,
            ),
        ),
        max_manifest_bytes=cast(
            int,
            _resolved(
                values,
                "workspace_max_manifest_bytes",
                workspace_defaults.max_manifest_bytes,
            ),
        ),
        default_results=cast(
            int,
            _resolved(
                values,
                "workspace_default_results",
                workspace_defaults.default_results,
            ),
        ),
    )
    return SandboxConfig(
        provider=provider,
        image=image,
        cpu_cores=cast(float | None, values.get("cpu_cores")),
        memory_mb=cast(int | None, values.get("memory_mb")),
        gpu=cast(str | None, values.get("gpu")),
        workspace=workspace,
        open_seconds=cast(
            int, _resolved(values, "open_seconds", defaults.open_seconds)
        ),
        transport_seconds=cast(
            int, _resolved(values, "transport_seconds", defaults.transport_seconds)
        ),
        cell_seconds=cast(
            float | None,
            _resolved(values, "cell_seconds", defaults.cell_seconds),
        ),
        idle_seconds=cast(
            float | None,
            _resolved(values, "idle_seconds", defaults.idle_seconds),
        ),
        max_lifetime_seconds=cast(
            int | None,
            _resolved(values, "max_lifetime_seconds", defaults.max_lifetime_seconds),
        ),
        model=configured_model(values),
    )


def _validate(values: SettingsValues) -> None:
    _sandbox_config(values)


def _bind(
    values: SettingsValues, secrets: Mapping[str, str]
) -> Callable[[PluginContext], Sandbox]:
    config = _sandbox_config(values)
    provider = cast(str, config.provider)
    e2b_api_key = secrets.get("e2b_api_key")
    daytona_api_key = secrets.get("daytona_api_key")
    daytona_api_url = cast(str | None, values.get("daytona_api_url"))
    modal_token_id = secrets.get("modal_token_id")
    modal_token_secret = secrets.get("modal_token_secret")
    sprites_token = secrets.get("sprites_token")
    if (modal_token_id is None) != (modal_token_secret is None):
        raise ValueError("Modal token ID and token secret must be configured together")
    has_backend_overrides = any(
        value is not None
        for value in (
            e2b_api_key,
            daytona_api_key,
            daytona_api_url,
            modal_token_id,
            modal_token_secret,
            sprites_token,
        )
    )

    def create_capability(context: PluginContext) -> Sandbox:
        from .capability import capability

        if not has_backend_overrides:
            return capability(context, config=config)

        from .backends import create_backend

        def create_configured_backend():
            return create_backend(
                provider,
                e2b_api_key=e2b_api_key,
                daytona_api_key=daytona_api_key,
                daytona_api_url=daytona_api_url,
                modal_token_id=modal_token_id,
                modal_token_secret=modal_token_secret,
                sprites_token=sprites_token,
            )

        return capability(
            context,
            config=config,
            backend_factory=create_configured_backend,
        )

    return create_capability


def _notice(values: SettingsValues) -> str | None:
    provider = values.get("provider")
    if provider not in _REMOTE_PROVIDERS:
        return None
    notice = (
        f"Selected query results and files are uploaded to {str(provider).title()}; "
        "provider charges may apply."
    )
    if provider == "modal":
        notice += (
            " Leave both token fields blank to use Modal's native authentication "
            "(for example, `modal token new`)."
        )
    return notice


settings = PluginSettings(
    fields=(
        Setting(
            name="provider",
            label="Sandbox provider",
            choices=_PROVIDERS,
            env="SQLSABER_SANDBOX_PROVIDER",
            required=True,
            help="Choose a local or remote provider explicitly.",
        ),
        model_setting(label="Analyst model"),
        Setting(
            name="docker_image",
            label="Docker image",
            when=_when_provider("docker"),
            help="Container image; blank uses SQLsaber's pinned scientific image.",
        ),
        Setting(
            name="microsandbox_image",
            label="Microsandbox image",
            when=_when_provider("microsandbox"),
            help="MicroVM image; blank uses SQLsaber's pinned scientific image.",
        ),
        Setting(
            name="e2b_template",
            label="E2B template",
            when=_when_provider("e2b"),
            help="E2B template name or ID; configure resources in the template.",
        ),
        Setting(
            name="modal_image",
            label="Modal image",
            when=_when_provider("modal"),
            help="Registry image; blank uses SQLsaber's pinned scientific image.",
        ),
        Setting(
            name="daytona_image",
            label="Daytona image",
            when=_when_provider("daytona"),
            help="Registry image; blank uses SQLsaber's pinned scientific image.",
        ),
        Setting(
            name="cpu_cores",
            label="CPU cores",
            kind="number",
            advanced=True,
            when=_when_provider(
                "docker", "microsandbox", "modal", "daytona", "sprites"
            ),
            help="Requested CPU cores; blank uses the provider default.",
        ),
        Setting(
            name="memory_mb",
            label="Memory (MiB)",
            kind="integer",
            advanced=True,
            when=_when_provider(
                "docker", "microsandbox", "modal", "daytona", "sprites"
            ),
            help="Requested memory in MiB; blank uses the provider default.",
        ),
        Setting(
            name="gpu",
            label="GPU",
            advanced=True,
            when=_when_provider("docker", "modal", "daytona"),
            help="Provider-specific GPU request; blank disables GPU allocation.",
        ),
        Setting(
            name="e2b_api_key",
            label="E2B API key",
            kind="secret",
            env="E2B_API_KEY",
            credential="e2b.api_key",
            when=_when_provider("e2b"),
            help="Optional when the E2B SDK can authenticate natively.",
        ),
        Setting(
            name="daytona_api_key",
            label="Daytona API key",
            kind="secret",
            env="DAYTONA_API_KEY",
            credential="daytona.api_key",
            when=_when_provider("daytona"),
            help="Optional when the Daytona SDK can authenticate from its environment.",
        ),
        Setting(
            name="daytona_api_url",
            label="Daytona API endpoint",
            env="DAYTONA_API_URL",
            when=_when_provider("daytona"),
            help="Custom Daytona API URL; blank uses the SDK default.",
        ),
        Setting(
            name="modal_token_id",
            label="Modal token ID",
            kind="secret",
            env="MODAL_TOKEN_ID",
            credential="modal.token_id",
            when=_when_provider("modal"),
            help="Set with the token secret, or leave both blank for native Modal auth.",
        ),
        Setting(
            name="modal_token_secret",
            label="Modal token secret",
            kind="secret",
            env="MODAL_TOKEN_SECRET",
            credential="modal.token_secret",
            when=_when_provider("modal"),
            help="Set with the token ID, or leave both blank for native Modal auth.",
        ),
        Setting(
            name="sprites_token",
            label="Sprites token",
            kind="secret",
            env="SPRITES_TOKEN",
            credential="sprites.token",
            when=_when_provider("sprites"),
            help="Sprites API access token.",
        ),
        Setting(
            name="open_seconds",
            label="Open timeout (seconds)",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.open_seconds,
            advanced=True,
        ),
        Setting(
            name="transport_seconds",
            label="Transport timeout (seconds)",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.transport_seconds,
            advanced=True,
        ),
        Setting(
            name="cell_seconds",
            label="Cell timeout (seconds)",
            kind="number",
            default=DEFAULT_SANDBOX_CONFIG.cell_seconds,
            advanced=True,
            help="Use null to disable the per-cell timeout.",
        ),
        Setting(
            name="idle_seconds",
            label="Idle timeout (seconds)",
            kind="number",
            default=DEFAULT_SANDBOX_CONFIG.idle_seconds,
            advanced=True,
        ),
        Setting(
            name="max_lifetime_seconds",
            label="Maximum lifetime (seconds)",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.max_lifetime_seconds,
            advanced=True,
        ),
        Setting(
            name="workspace_max_files",
            label="Maximum workspace files",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.workspace.max_files,
            advanced=True,
        ),
        Setting(
            name="workspace_max_file_bytes",
            label="Maximum bytes per workspace file",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.workspace.max_file_bytes,
            advanced=True,
        ),
        Setting(
            name="workspace_max_total_bytes",
            label="Maximum total workspace bytes",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.workspace.max_total_bytes,
            advanced=True,
        ),
        Setting(
            name="workspace_max_manifest_bytes",
            label="Maximum workspace manifest bytes",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.workspace.max_manifest_bytes,
            advanced=True,
        ),
        Setting(
            name="workspace_default_results",
            label="Default recent query results",
            kind="integer",
            default=DEFAULT_SANDBOX_CONFIG.workspace.default_results,
            advanced=True,
        ),
    ),
    validate=_validate,
    bind=_bind,
    notice=_notice,
)
