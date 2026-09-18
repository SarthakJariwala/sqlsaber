from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from sqlsaber.cli import auth as auth_cli
from sqlsaber.config.auth import AuthConfigManager
from sqlsaber.config.openai_codex import OpenAICodexCredentialStore


def credentials() -> OpenAICodexCredentials:
    return OpenAICodexCredentials(
        access_token="fake-access",
        refresh_token="fake-refresh",
        account_id="fake-account",
    )


class FakeFlow:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def authorization_url(self) -> str:
        return "https://auth.example.test/authorize"

    async def exchange_code_from_callback(self) -> OpenAICodexCredentials:
        self.events.append("callback-listening")
        await asyncio.sleep(0)
        return credentials()


class FakeStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.saved: OpenAICodexCredentials | None = None

    async def save(self, value: OpenAICodexCredentials) -> None:
        self.events.append("credentials-saved")
        self.saved = value


@pytest.mark.asyncio
async def test_login_starts_callback_before_opening_browser_and_saves_result() -> None:
    events: list[str] = []
    store = FakeStore(events)

    def open_browser(url: str) -> bool:
        assert url == "https://auth.example.test/authorize"
        events.append("browser-opened")
        return True

    await auth_cli._login_openai_codex(
        store=store,
        flow=FakeFlow(events),
        open_browser=open_browser,
        timeout_seconds=1,
    )

    assert events == [
        "callback-listening",
        "browser-opened",
        "credentials-saved",
    ]
    assert store.saved == credentials()


def configure_paths(monkeypatch, tmp_path) -> Callable[[], OpenAICodexCredentialStore]:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    monkeypatch.setattr(
        "platformdirs.user_config_dir", lambda *args, **kwargs: str(config_dir)
    )
    monkeypatch.setattr(
        "platformdirs.user_data_dir", lambda *args, **kwargs: str(data_dir)
    )
    monkeypatch.setattr(auth_cli, "config_manager", AuthConfigManager())
    monkeypatch.setattr(auth_cli.providers, "api_key_keys", lambda: [])
    return OpenAICodexCredentialStore


def test_status_reports_sqlsaber_codex_login_and_refresh_limit(
    monkeypatch, tmp_path, capsys
) -> None:
    store_factory = configure_paths(monkeypatch, tmp_path)
    asyncio.run(store_factory().save(credentials()))

    auth_cli.status()

    output = capsys.readouterr().out
    assert "openai-codex" in output
    assert "connected" in output
    assert str(store_factory().path) in output
    assert "concurrent sqlsaber processes" in output.casefold()


def test_fresh_status_points_to_api_key_setup_and_codex_login(
    monkeypatch, tmp_path, capsys
) -> None:
    configure_paths(monkeypatch, tmp_path)

    auth_cli.status()

    output = capsys.readouterr().out
    assert "No authentication method configured" in output
    assert "saber auth setup" in output
    assert "saber auth login openai-codex" in output


def test_logout_removes_only_sqlsaber_codex_credentials(
    monkeypatch, tmp_path, capsys
) -> None:
    store_factory = configure_paths(monkeypatch, tmp_path)
    store = store_factory()
    asyncio.run(store.save(credentials()))

    auth_cli.logout("openai-codex", yes=True)

    assert store.is_configured() is False
    assert "Logged out of OpenAI Codex" in capsys.readouterr().out


def test_login_rejects_api_key_provider_before_starting_oauth(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        auth_cli.login("openai")

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "only the openai-codex provider" in error
    assert "saber auth login openai-codex" in error
