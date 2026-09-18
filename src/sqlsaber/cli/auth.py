"""Authentication CLI commands."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Protocol

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

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials


class _CodexCredentialWriter(Protocol):
    async def save(self, credentials: OpenAICodexCredentials) -> None: ...


class _CodexLoginFlow(Protocol):
    def authorization_url(self) -> str: ...
    async def exchange_code_from_callback(self) -> OpenAICodexCredentials: ...


auth_app = cyclopts.App(
    name="auth",
    help="Manage authentication configuration",
    help_epilogue=(
        "Examples:\n\n"
        "saber auth status\n\n"
        "saber auth login openai-codex\n\n"
        "saber auth reset openai --yes"
    ),
)


async def _login_openai_codex(
    *,
    store: _CodexCredentialWriter,
    flow: _CodexLoginFlow,
    open_browser: Callable[[str], bool],
    timeout_seconds: float = 600,
) -> None:
    """Run the browser callback flow and persist its returned credentials."""

    callback = asyncio.create_task(flow.exchange_code_from_callback())
    try:
        await asyncio.sleep(0)
        if callback.done():
            credentials = await callback
        else:
            open_browser(flow.authorization_url())
            async with asyncio.timeout(timeout_seconds):
                credentials = await callback
        await store.save(credentials)
    finally:
        if not callback.done():
            callback.cancel()
            with suppress(asyncio.CancelledError):
                await callback


def _canonical_codex_provider(provider: str) -> str:
    canonical = providers.canonical(provider.strip().lower())
    if (
        canonical is None
        or providers.auth_kind(canonical) is not providers.AuthKind.OPENAI_CODEX
    ):
        raise ValueError(
            "only the openai-codex provider supports browser login and logout"
        )
    return canonical


@auth_app.command(help_epilogue="Example:\n\nsaber auth login openai-codex")
def login(
    provider: Annotated[
        str,
        cyclopts.Parameter(help="Subscription provider to authenticate"),
    ] = "openai-codex",
) -> None:
    """Sign in to OpenAI Codex with browser OAuth."""

    try:
        provider = _canonical_codex_provider(provider)
    except ValueError as exc:
        fail_usage(f"{exc}.\n  Example: saber auth login openai-codex")

    import webbrowser

    from pydantic_ai.providers.openai_codex import OpenAICodexOAuthFlow

    from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

    store = OpenAICodexCredentialStore()
    flow = OpenAICodexOAuthFlow()
    out(b.md("**OpenAI Codex Login**"))

    def open_browser(url: str) -> bool:
        out(
            b.md("Complete sign-in in your browser."),
            b.md(f"If the browser does not open, visit: {url}"),
        )
        return webbrowser.open(url)

    try:
        asyncio.run(
            _login_openai_codex(
                store=store,
                flow=flow,
                open_browser=open_browser,
            )
        )
    except TimeoutError:
        fail(
            "OpenAI Codex login timed out after 10 minutes. Run "
            "`saber auth login openai-codex` to try again."
        )
    except Exception as exc:
        fail(
            f"OpenAI Codex login failed: {exc}\n"
            "  Run `saber auth login openai-codex` to try again."
        )

    config_manager.set_auth_method(AuthMethod.OPENAI_CODEX)
    out(
        b.success("OpenAI Codex login saved for SQLsaber."),
        b.md(f"Credentials: `{store.path}`", role="muted"),
        b.md(
            "Token refresh is serialized within one SQLsaber process. Concurrent "
            "SQLsaber processes can race while rotating credentials.",
            role="muted",
        ),
    )


@auth_app.command(help_epilogue="Example:\n\nsaber auth setup")
def setup():
    """Configure authentication for SQLsaber (API keys).

    Example:
        saber auth setup
    """
    from sqlsaber.cli.workflows.auth_setup import setup_auth

    out(b.md("**SQLsaber Authentication Setup**"))

    async def run_setup():
        prompter = AsyncPrompter()
        api_key_manager = APIKeyManager()
        return await setup_auth(
            prompter=prompter,
            auth_manager=config_manager,
            api_key_manager=api_key_manager,
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
                "Run `saber auth setup` for an API key or "
                "`saber auth login openai-codex` for subscription access."
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
                "Run `saber auth setup` for an API key or "
                "`saber auth login openai-codex` for subscription access."
            ),
        )
        logger.info("auth.status.none_configured")
    logger.info("auth.status.complete", method=str(auth_method))


def _remove_openai_codex_credentials(
    *,
    yes: bool,
    non_interactive_command: str,
) -> bool:
    from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

    store = OpenAICodexCredentialStore()
    if not store.is_configured():
        out(b.warn("No SQLsaber OpenAI Codex login found. Nothing to remove."))
        return False
    confirmed = confirm_action(
        yes=yes,
        prompt="Remove SQLsaber's stored OpenAI Codex login?",
        non_interactive_command=non_interactive_command,
    )
    if not confirmed:
        out(b.warn("Logout cancelled."))
        return False
    try:
        store.delete()
    except OSError as exc:
        fail(f"could not remove OpenAI Codex credentials: {exc}")
    if config_manager.get_auth_method() is AuthMethod.OPENAI_CODEX:
        config_manager.clear_auth_method()
    return True


@auth_app.command(
    help_epilogue=(
        "Examples:\n\n"
        "saber auth logout openai-codex\n\n"
        "saber auth logout openai-codex --yes"
    )
)
def logout(
    provider: Annotated[
        str,
        cyclopts.Parameter(help="Subscription provider to disconnect"),
    ] = "openai-codex",
    yes: Annotated[
        bool,
        cyclopts.Parameter(["--yes"], help="Skip confirmation prompt"),
    ] = False,
) -> None:
    """Remove SQLsaber's stored OpenAI Codex login."""

    try:
        provider = _canonical_codex_provider(provider)
    except ValueError as exc:
        fail_usage(f"{exc}.\n  Example: saber auth logout openai-codex --yes")

    out(b.md("**OpenAI Codex Logout**"))
    if _remove_openai_codex_credentials(
        yes=yes,
        non_interactive_command=f"saber auth logout {provider} --yes",
    ):
        out(b.success("Logged out of OpenAI Codex for SQLsaber."))


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
    """Reset stored API key credentials for a selected provider.

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
        if _remove_openai_codex_credentials(
            yes=yes,
            non_interactive_command=f"saber auth reset {provider} --yes",
        ):
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
