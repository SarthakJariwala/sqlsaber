from __future__ import annotations

import json
import os
from pathlib import Path
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
async def test_missing_credentials_name_the_setup_command(tmp_path) -> None:
    path = tmp_path / "missing.json"
    store = OpenAICodexCredentialStore(path)

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        await store.load()

    assert str(exc_info.value) == (
        "No SQLsaber OpenAI Codex credentials were found. Run "
        "`saber auth setup openai-codex`."
    )
    assert str(path) not in str(exc_info.value)


def test_preflight_rejects_malformed_credentials_before_model_request(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.write_text('{"version": 1, "access_token": "missing other fields"}')
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        OpenAICodexCredentialStore(path).preflight()

    assert str(exc_info.value) == (
        "Malformed SQLsaber OpenAI Codex credentials. Run "
        "`saber auth setup openai-codex` again."
    )
    assert str(path) not in str(exc_info.value)


def test_unreadable_credentials_hide_path_and_retain_cause(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.write_text("{}")
    if os.name != "nt":
        path.chmod(0o600)

    def fail_read_text(self: Path, *args, **kwargs) -> str:
        del self, args, kwargs
        raise OSError("raw failure at /private/credential/path")

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        OpenAICodexCredentialStore(path).preflight()

    assert str(exc_info.value) == (
        "Could not read SQLsaber OpenAI Codex credentials. Run "
        "`saber auth setup openai-codex` again."
    )
    assert str(path) not in str(exc_info.value)
    assert "/private/credential/path" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, OSError)


def test_credential_metadata_errors_hide_path_and_retain_cause(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "openai_codex_credentials.json"

    def fail_lstat(self: Path) -> os.stat_result:
        del self
        raise PermissionError("raw failure at /private/credential/path")

    monkeypatch.setattr(Path, "lstat", fail_lstat)

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        OpenAICodexCredentialStore(path).preflight()

    assert str(exc_info.value) == (
        "Could not read SQLsaber OpenAI Codex credentials. Run "
        "`saber auth setup openai-codex` again."
    )
    assert str(path) not in str(exc_info.value)
    assert "/private/credential/path" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, PermissionError)


def test_invalid_utf8_credentials_are_reported_as_malformed(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.write_bytes(b"\xff")
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        OpenAICodexCredentialStore(path).preflight()

    assert str(exc_info.value) == (
        "Malformed SQLsaber OpenAI Codex credentials. Run "
        "`saber auth setup openai-codex` again."
    )
    assert str(path) not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


def test_non_regular_credential_path_names_the_required_repair(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    path.mkdir()

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        OpenAICodexCredentialStore(path).preflight()

    assert str(exc_info.value) == (
        f"OpenAI Codex credential path `{path}` is not a regular file. "
        "Move or remove it, then run `saber auth setup openai-codex` again."
    )


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

    with pytest.raises(OpenAICodexAuthError) as exc_info:
        await OpenAICodexCredentialStore(path).load()

    assert str(exc_info.value) == (
        f"OpenAI Codex credentials at `{path}` are readable by other users. "
        f"Run `chmod 600 {path}` and retry."
    )


@pytest.mark.asyncio
async def test_delete_is_idempotent(tmp_path) -> None:
    path = tmp_path / "openai_codex_credentials.json"
    store = OpenAICodexCredentialStore(path)
    await store.save(credentials())

    assert store.delete() is True
    assert store.delete() is False
    assert store.is_configured() is False
