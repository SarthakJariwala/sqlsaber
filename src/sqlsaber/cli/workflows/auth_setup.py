"""Shared auth setup logic for onboarding and CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
import os
from typing import TYPE_CHECKING, Final, Protocol

from sqlsaber.cli.prompts import Prompter
from sqlsaber.cli.output import err, out
from sqlsaber.config import providers
from sqlsaber.config.api_keys import APIKeyManager
from sqlsaber.config.auth import AuthConfigManager, AuthMethod
from sqlsaber.render import blocks as b

DEFAULT_PROVIDER: Final[str] = "openai"

if TYPE_CHECKING:
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials


class CodexCredentialWriter(Protocol):
    async def save(self, credentials: OpenAICodexCredentials) -> None: ...


class CodexOAuthFlow(Protocol):
    def authorization_url(self) -> str: ...
    async def exchange_code_from_callback(self) -> OpenAICodexCredentials: ...


type OpenAICodexSetup = Callable[[], Awaitable[bool]]


async def authenticate_openai_codex(
    *,
    store: CodexCredentialWriter,
    flow: CodexOAuthFlow,
    open_browser: Callable[[str], bool],
    timeout_seconds: float = 600,
) -> None:
    """Run browser OAuth and persist the returned credentials."""

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


async def configure_openai_codex() -> bool:
    """Connect SQLsaber to ChatGPT through native browser OAuth."""

    import webbrowser

    from pydantic_ai.providers.openai_codex import OpenAICodexOAuthFlow

    from sqlsaber.config.openai_codex import OpenAICodexCredentialStore

    store = OpenAICodexCredentialStore()
    flow = OpenAICodexOAuthFlow()
    out(b.md("**ChatGPT Subscription Authentication**"))

    def open_browser(url: str) -> bool:
        out(
            b.md("Complete authentication in your browser."),
            b.md(f"If the browser does not open, visit: {url}"),
        )
        return webbrowser.open(url)

    try:
        await authenticate_openai_codex(
            store=store,
            flow=flow,
            open_browser=open_browser,
        )
    except TimeoutError:
        err(
            b.error(
                "OpenAI Codex authentication timed out after 10 minutes. Run "
                "`saber auth setup openai-codex` to try again."
            )
        )
        return False
    except Exception as exc:
        err(
            b.error(
                f"OpenAI Codex authentication failed: {exc}\n"
                "Run `saber auth setup openai-codex` to try again."
            )
        )
        return False

    out(
        b.success("OpenAI Codex authentication configured successfully!"),
        b.md(f"Credentials: `{store.path}`", role="muted"),
        b.md(
            "Token refresh is serialized within one SQLsaber process. Concurrent "
            "SQLsaber processes can race while rotating credentials.",
            role="muted",
        ),
    )
    return True


async def select_provider(
    prompter: Prompter, default: str = DEFAULT_PROVIDER
) -> str | None:
    """Interactive provider selection.

    Args:
        prompter: Prompter instance for interaction
        default: Default provider to select

    Returns:
        Selected provider name or None if cancelled

    """

    provider = await prompter.select(
        "Select AI provider:", choices=providers.all_keys(), default=default
    )
    return provider


async def configure_api_key(
    provider: str,
    prompter: Prompter,
    api_key_manager: APIKeyManager,
    auth_manager: AuthConfigManager,
) -> bool:
    """Configure API key for a provider.

    Args:
        provider: Provider name
        api_key_manager: APIKeyManager instance
        auth_manager: AuthConfigManager instance

    Returns:
        True if API key configured successfully, False otherwise

    """

    api_key = api_key_manager.get_configured_api_key(provider)
    if not api_key:
        api_key = await prompter.secret(
            f"Enter your {provider.title()} API key (leave blank to skip):"
        )
        api_key = api_key.strip() if api_key else None
        if api_key and not api_key_manager.store_api_key(provider, api_key):
            return False

    if not api_key:
        return False
    auth_manager.set_auth_method(AuthMethod.API_KEY)
    return True


async def setup_auth(
    prompter: Prompter,
    auth_manager: AuthConfigManager,
    api_key_manager: APIKeyManager,
    default_provider: str = DEFAULT_PROVIDER,
    provider: str | None = None,
    openai_codex_setup: OpenAICodexSetup = configure_openai_codex,
) -> tuple[bool, str | None]:
    """Interactive authentication setup.

    Args:
        prompter: Prompter instance for interaction
        auth_manager: AuthConfigManager instance
        api_key_manager: APIKeyManager instance
        default_provider: Default provider to select
        provider: Provider selected by a non-interactive caller
        openai_codex_setup: Browser OAuth setup operation

    Returns:
        Tuple of (success: bool, provider: str | None)

    """

    if provider is None:
        provider = await select_provider(prompter, default=default_provider)

    if provider is None:
        return False, None

    if providers.auth_kind(provider) is providers.AuthKind.OPENAI_CODEX:
        configured = await openai_codex_setup()
        if not configured:
            return False, None
        auth_manager.set_auth_method(AuthMethod.OPENAI_CODEX)
        return True, provider

    env_var = api_key_manager.get_env_var_name(provider)
    api_key_in_env = bool(os.getenv(env_var))
    api_key_in_keyring = api_key_manager.has_stored_api_key(provider)

    if api_key_in_env or api_key_in_keyring:
        parts: list[str] = []
        if api_key_in_keyring:
            parts.append("stored API key")
        if api_key_in_env:
            parts.append(f"{env_var} environment variable")
        summary = ", ".join(parts)
        out(b.md(f"Existing authentication found for {provider}: {summary}"))

    if api_key_in_keyring:
        reset_api_key = await prompter.confirm(
            f"{provider.title()} API key is stored in your keyring. Reset before continuing?",
            default=False,
        )
        if not reset_api_key:
            out(b.warn("No changes made to stored API key credentials."))
            return True, None
        if not api_key_manager.delete_api_key(provider):
            err(b.error("Failed to remove existing API key credentials."))
            return False, None
        out(b.md(f"{provider.title()} API key removed from keyring."))
        api_key_in_keyring = False

    if api_key_in_env:
        out(
            b.md(
                f"{env_var} is set in your environment. Update it there if you need a new value."
            )
        )

    out(
        b.md(
            f"To use {provider.title()}, you need an API key.\n"
            f"You can set the {env_var} environment variable,\n"
            "or enter it now to store securely in your OS keychain."
        )
    )

    api_key_configured = await configure_api_key(
        provider, prompter, api_key_manager, auth_manager
    )

    if api_key_configured:
        out(b.success(f"{provider.title()} API key configured successfully!"))
        return True, provider

    out(b.warn("No API key provided."))
    return False, None
