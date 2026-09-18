"""Authentication CLI commands."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Annotated

import cyclopts
import keyring

from sqlsaber.cli.prompts import AsyncPrompter
from sqlsaber.cli.output import fail, fail_usage, out
from sqlsaber.cli.safety import confirm_action
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.config.logging import get_logger
from sqlsaber.render import blocks as b

config_manager = AuthConfigManager()
logger = get_logger(__name__)

auth_app = cyclopts.App(
    name="auth",
    help="Manage authentication configuration",
    help_epilogue=(
        "Examples:\n\n"
        "saber auth status\n\n"
        "saber auth setup openai-codex\n\n"
        "saber auth reset openai --yes"
    ),
)


@auth_app.command(
    help_epilogue=("Examples:\n\nsaber auth setup\n\nsaber auth setup openai-codex")
)
def setup(
    provider: Annotated[
        str | None,
        cyclopts.Parameter(help="Provider to configure (omit to select interactively)"),
    ] = None,
) -> None:
    """Configure authentication for SQLsaber.

    Examples:
        saber auth setup
        saber auth setup openai-codex
    """
    from sqlsaber.cli.workflows.auth_setup import setup_auth

    if provider is not None:
        canonical_provider = providers.canonical(provider.strip().lower())
        if canonical_provider is None:
            choices = ", ".join(providers.all_keys())
            fail_usage(
                f"unsupported provider '{provider}'.\n"
                f"  Choose from: {choices}\n"
                "  Example: saber auth setup openai-codex"
            )
        provider = canonical_provider

    out(b.md("**SQLsaber Authentication Setup**"))

    async def run_setup():
        prompter = AsyncPrompter()
        api_key_manager = APIKeyManager()
        return await setup_auth(
            prompter=prompter,
            auth_manager=config_manager,
            api_key_manager=api_key_manager,
            provider=provider,
        )

    logger.info("auth.setup.start")
    configured, provider = asyncio.run(run_setup())
    logger.info("auth.setup.complete", success=bool(configured), provider=str(provider))

    if not configured:
        fail("no authentication was configured.")

    out(b.md("You can change this anytime by running `saber auth setup` again."))


@auth_app.command(help_epilogue="Example:\n\nsaber auth status")
def status():
    """Show current authentication configuration and provider key status.

    Example:
        saber auth status
    """
    logger.info("auth.status.start")
    auth_method = config_manager.get_auth_method()

    out(b.md("**Authentication Status**"))

    from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

    codex_store = OpenAICodexCredentialStore()
    codex_configured = codex_store.is_configured()
    if auth_method is None and not codex_configured:
        out(
            b.warn("No authentication method configured"),
            b.md(
                "Run `saber auth setup` to configure an API key or ChatGPT "
                "subscription access."
            ),
        )
        logger.info("auth.status.none_configured")
        return
    if auth_method is AuthMethod.API_KEY:
        out(b.success("API Key authentication configured"))
    elif codex_configured:
        out(b.success("OpenAI Codex authentication configured"))

    api_key_manager = APIKeyManager()
    rows: list[dict[str, str]] = []
    configured = codex_configured
    for provider in providers.api_key_keys():
        env_var = api_key_manager.get_env_var_name(provider)
        service = api_key_manager._get_service_name(provider)
        from_env = bool(os.getenv(env_var))
        from_keyring = bool(keyring.get_password(service, provider))
        if from_env:
            state = f"configured via {env_var}"
            configured = True
        elif from_keyring:
            state = "configured"
            configured = True
        else:
            state = "not configured"
        rows.append({"provider": provider, "status": state})

    rows.append(
        {
            "provider": "openai-codex",
            "status": "connected" if codex_configured else "not connected",
        }
    )
    out(
        b.table(
            rows,
            columns=(b.Column("provider", "Provider"), b.Column("status", "Status")),
        )
    )
    if codex_configured:
        out(
            b.md(f"OpenAI Codex credentials: `{codex_store.path}`", role="muted"),
            b.md(
                "Token refresh is serialized within one process. Concurrent SQLsaber "
                "processes can race while rotating credentials.",
                role="muted",
            ),
        )
    if not configured:
        out(
            b.warn("No authentication credentials configured"),
            b.md(
                "Run `saber auth setup` to configure an API key or ChatGPT "
                "subscription access."
            ),
        )
        logger.info("auth.status.none_configured")
    logger.info("auth.status.complete", method=str(auth_method))


def _remove_openai_codex_credentials(
    *,
    yes: bool,
) -> bool:
    from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

    store = OpenAICodexCredentialStore()
    if not store.is_configured():
        out(b.warn("No stored OpenAI Codex credentials found. Nothing to reset."))
        return False
    confirmed = confirm_action(
        yes=yes,
        prompt="Remove SQLsaber's stored OpenAI Codex credentials?",
        non_interactive_command="saber auth reset openai-codex --yes",
    )
    if not confirmed:
        out(b.warn("Reset cancelled."))
        return False
    try:
        store.delete()
    except OSError as exc:
        fail(f"could not remove OpenAI Codex credentials: {exc}")
    if config_manager.get_auth_method() is AuthMethod.OPENAI_CODEX:
        config_manager.clear_auth_method()
    return True


@auth_app.command(
    help_epilogue=("Examples:\n\nsaber auth reset\n\nsaber auth reset openai --yes")
)
def reset(
    provider: Annotated[
        str | None,
        cyclopts.Parameter(help="Provider to reset (omit to select interactively)"),
    ] = None,
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
):
    """Reset stored credentials for a selected provider.

    Examples:
        saber auth reset
        saber auth reset openai --yes
    """
    out(b.md("**SQLsaber Authentication Reset**"))

    if provider is None:
        if not sys.stdin.isatty():
            fail_usage(
                "PROVIDER is required when stdin is not a terminal.\n"
                "  Example: saber auth reset openai --yes"
            )
        provider = asyncio.run(
            AsyncPrompter().select(
                "Select provider to reset:",
                choices=list(providers.all_keys()),
            )
        )

    if provider is None:
        out(b.warn("Reset cancelled."))
        logger.info("auth.reset.cancelled_no_provider")
        return

    canonical_provider = providers.canonical(provider.strip().lower())
    if canonical_provider is None:
        choices = ", ".join(providers.all_keys())
        fail_usage(
            f"unsupported provider '{provider}'.\n"
            f"  Choose from: {choices}\n"
            "  Example: saber auth reset openai --yes"
        )
    provider = canonical_provider

    if providers.auth_kind(provider) is providers.AuthKind.OPENAI_CODEX:
        if _remove_openai_codex_credentials(yes=yes):
            out(b.success("Reset complete."))
        return

    api_key_manager = APIKeyManager()
    service = api_key_manager._get_service_name(provider)

    api_key_present = bool(keyring.get_password(service, provider))

    if not api_key_present:
        out(b.warn(f"No stored credentials found for {provider}. Nothing to reset."))
        logger.info("auth.reset.nothing_to_reset", provider=provider)
        return

    confirmed = confirm_action(
        yes=yes,
        prompt=f"Remove the stored {provider.title()} API key from your keyring?",
        non_interactive_command=f"saber auth reset {provider} --yes",
    )

    if not confirmed:
        out(b.warn("Reset cancelled."))
        logger.info("auth.reset.cancelled_confirm", provider=provider)
        return

    try:
        keyring.delete_password(service, provider)
        out(b.success(f"Removed {provider} API key from keyring"))
        logger.info("auth.reset.api_key_removed", provider=provider)
    except Exception as e:
        logger.warning(
            "auth.reset.api_key_remove_failed", provider=provider, error=str(e)
        )
        fail(f"could not remove API key: {e}")

    out(
        b.success("Reset complete."),
        b.md("Environment variables are not modified by this command.", role="muted"),
    )
    logger.info("auth.reset.complete", provider=provider)


def create_auth_app() -> cyclopts.App:
    """Return the authentication management CLI app."""

    return auth_app
