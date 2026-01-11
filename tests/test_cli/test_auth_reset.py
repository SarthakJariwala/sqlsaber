"""Tests for provider-aware auth reset CLI."""

from unittest.mock import patch

from sqlsaber.cli.auth import reset as auth_reset


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
@patch("sqlsaber.cli.auth.questionary.confirm")
@patch("sqlsaber.cli.auth.questionary.select")
def test_reset_openai_api_key(
    mock_select, mock_confirm, mock_get_password, mock_delete
):
    """Removes OpenAI API key when present in keyring."""
    mock_select.return_value.ask.return_value = "openai"
    mock_confirm.return_value.ask.return_value = True

    def fake_get_password(service: str, provider: str):
        if service == "sqlsaber-openai-api-key" and provider == "openai":
            return "sk-openai"
        return None

    mock_get_password.side_effect = fake_get_password

    auth_reset()

    mock_delete.assert_called_once_with("sqlsaber-openai-api-key", "openai")


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
@patch("sqlsaber.cli.auth.questionary.confirm")
@patch("sqlsaber.cli.auth.questionary.select")
def test_reset_anthropic_api_key_only(
    mock_select, mock_confirm, mock_get_password, mock_delete
):
    """Removes Anthropic API key when present in keyring."""
    mock_select.return_value.ask.return_value = "anthropic"
    mock_confirm.return_value.ask.return_value = True

    def fake_get_password(service: str, provider: str):
        if service == "sqlsaber-anthropic-api-key" and provider == "anthropic":
            return "sk-anthropic"
        return None

    mock_get_password.side_effect = fake_get_password

    auth_reset()

    mock_delete.assert_called_once_with("sqlsaber-anthropic-api-key", "anthropic")


@patch("sqlsaber.cli.auth.keyring.delete_password")
@patch("sqlsaber.cli.auth.keyring.get_password")
@patch("sqlsaber.cli.auth.questionary.select")
def test_reset_no_credentials_noop(mock_select, mock_get_password, mock_delete):
    """If no credentials are stored, reset is a no-op for that provider."""
    mock_select.return_value.ask.return_value = "groq"
    mock_get_password.return_value = None

    auth_reset()

    mock_delete.assert_not_called()
