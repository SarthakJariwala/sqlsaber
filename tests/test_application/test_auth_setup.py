"""Tests for the interactive auth setup flow."""

from __future__ import annotations

from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sqlsaber.application.auth_setup import setup_auth
from sqlsaber.config.auth import AuthMethod


class DummyPrompter:
    """Simple prompter stub that yields predefined responses."""

    def __init__(
        self,
        selects: list[Any] | None = None,
        confirms: list[bool | None] | None = None,
    ):
        self._selects = deque(selects or [])
        self._confirms = deque(confirms or [])

    async def select(
        self,
        message: str,
        choices: list[Any] | None = None,
        default: Any = None,
        use_search_filter: bool = False,
        use_jk_keys: bool = True,
    ) -> Any:
        try:
            return self._selects.popleft()
        except IndexError:
            return None

    async def confirm(self, message: str, default: bool = False) -> bool | None:
        try:
            return self._confirms.popleft()
        except IndexError:
            return None

    async def text(self, message: str, default: str = "", validate=None) -> str | None:
        return None

    async def path(self, message: str, only_directories: bool = False) -> str | None:
        return None


@pytest.mark.asyncio
async def test_setup_auth_resets_existing_api_key(monkeypatch: pytest.MonkeyPatch):
    """Stored API key is cleared when the user opts into reset."""
    prompter = DummyPrompter(selects=["openai"], confirms=[True])
    auth_manager = MagicMock()
    api_key_manager = MagicMock()
    api_key_manager.get_env_var_name.return_value = "OPENAI_API_KEY"
    api_key_manager.has_stored_api_key.return_value = True
    api_key_manager.delete_api_key.return_value = True

    configure_api_key = AsyncMock(return_value=True)

    with (
        patch("sqlsaber.application.auth_setup.configure_api_key", configure_api_key),
        patch("sqlsaber.application.auth_setup.OAuthTokenManager"),
    ):
        success, provider = await setup_auth(
            prompter=prompter,
            auth_manager=auth_manager,
            api_key_manager=api_key_manager,
            allow_oauth=True,
            default_provider="openai",
        )

    assert success is True
    assert provider == "openai"
    api_key_manager.delete_api_key.assert_called_once_with("openai")
    configure_api_key.assert_awaited_once()


@pytest.mark.asyncio
async def test_setup_auth_resets_anthropic_oauth(monkeypatch: pytest.MonkeyPatch):
    """Existing Anthropic OAuth tokens are cleared before reconfiguration."""
    prompter = DummyPrompter(
        selects=["anthropic", AuthMethod.CLAUDE_PRO], confirms=[True]
    )
    auth_manager = MagicMock()
    auth_manager.get_auth_method.return_value = AuthMethod.CLAUDE_PRO
    api_key_manager = MagicMock()
    api_key_manager.get_env_var_name.return_value = "ANTHROPIC_API_KEY"
    api_key_manager.has_stored_api_key.return_value = False

    oauth_manager = MagicMock()
    oauth_manager.has_oauth_token.return_value = True
    oauth_manager.remove_oauth_token.return_value = True

    configure_oauth = AsyncMock(return_value=True)

    with (
        patch(
            "sqlsaber.application.auth_setup.OAuthTokenManager",
            return_value=oauth_manager,
        ),
        patch(
            "sqlsaber.application.auth_setup.configure_oauth_anthropic",
            configure_oauth,
        ),
    ):
        success, provider = await setup_auth(
            prompter=prompter,
            auth_manager=auth_manager,
            api_key_manager=api_key_manager,
            allow_oauth=True,
            default_provider="anthropic",
        )

    assert success is True
    assert provider == "anthropic"
    oauth_manager.remove_oauth_token.assert_called_once_with("anthropic")
    auth_manager.clear_auth_method.assert_called_once()
    configure_oauth.assert_awaited_once()
