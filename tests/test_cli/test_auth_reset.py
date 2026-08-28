"""Tests for provider-aware auth reset CLI."""

from unittest.mock import patch

import pytest

from sqlsaber.cli.auth import reset as auth_reset


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
def test_reset_openai_api_key(mock_get_password, mock_delete):
    """Removes OpenAI API key when present in keyring."""

    def fake_get_password(service: str, provider: str):
        if service == "sqlsaber-openai-api-key" and provider == "openai":
            return "sk-openai"
        return None

    mock_get_password.side_effect = fake_get_password

    auth_reset("openai", yes=True)

    mock_delete.assert_called_once_with("sqlsaber-openai-api-key", "openai")


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
def test_reset_anthropic_api_key_only(mock_get_password, mock_delete):
    """Removes Anthropic API key when present in keyring."""

    def fake_get_password(service: str, provider: str):
        if service == "sqlsaber-anthropic-api-key" and provider == "anthropic":
            return "sk-anthropic"
        return None

    mock_get_password.side_effect = fake_get_password

    auth_reset("anthropic", yes=True)

    mock_delete.assert_called_once_with("sqlsaber-anthropic-api-key", "anthropic")


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
def test_reset_no_credentials_noop(mock_get_password, mock_delete):
    """If no credentials are stored, reset is a no-op for that provider."""
    mock_get_password.return_value = None

    auth_reset("groq", yes=True)

    mock_delete.assert_not_called()


@patch("sqlsaber.cli.auth.keyring.delete_password", side_effect=RuntimeError("locked"))
@patch("sqlsaber.cli.auth.keyring.get_password", return_value="stored-key")
def test_reset_delete_failure_exits_nonzero(mock_get_password, mock_delete, capsys):
    """Credential deletion failures are reported on stderr and fail the command."""
    with pytest.raises(SystemExit) as exc_info:
        auth_reset("openai", yes=True)

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "could not remove API key" in captured.err
    assert "Reset complete" not in captured.out
