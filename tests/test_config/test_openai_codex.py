from __future__ import annotations

import json
import os
import stat

import pytest
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from sqlsaber.config.openai_codex import (
    OpenAICodexAuthError,
    OpenAICodexCredentialStore,
)


def credentials(access: str = "access-1") -> OpenAICodexCredentials:
    return OpenAICodexCredentials(
        access_token=access,
        refresh_token=f"refresh-for-{access}",
        account_id="account-1",
    )


@pytest.mark.asyncio
async def test_store_round_trips_and_rotates_credentials_atomically(tmp_path) -> None:
    path = tmp_path / "private" / "openai_codex_credentials.json"
    store = OpenAICodexCredentialStore(path)

    await store.save(credentials())
    await store.save(credentials("access-2"))

    loaded = await store.load()
    assert loaded == credentials("access-2")
    assert json.loads(path.read_text()) == {
        "version": 1,
        "access_token": "access-2",
        "refresh_token": "refresh-for-access-2",
        "account_id": "account-1",
    }
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


@pytest.mark.asyncio
async def test_missing_credentials_name_the_login_command(tmp_path) -> None:
    store = OpenAICodexCredentialStore(tmp_path / "missing.json")

    with pytest.raises(OpenAICodexAuthError, match="saber auth login openai-codex"):
        await store.load()


def test_preflight_rejects_malformed_credentials_before_model_request(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.write_text('{"version": 1, "access_token": "missing other fields"}')
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(OpenAICodexAuthError, match="Malformed.*login openai-codex"):
        OpenAICodexCredentialStore(path).preflight()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics")
async def test_store_rejects_credentials_readable_by_other_users(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "access_token": "access",
                "refresh_token": "refresh",
                "account_id": "account",
            }
        )
    )
    path.chmod(0o644)

    with pytest.raises(OpenAICodexAuthError, match="chmod 600"):
        await OpenAICodexCredentialStore(path).load()


@pytest.mark.asyncio
async def test_delete_is_idempotent(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    store = OpenAICodexCredentialStore(path)
    await store.save(credentials())

    assert store.delete() is True
    assert store.delete() is False
    assert store.is_configured() is False
