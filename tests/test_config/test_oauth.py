"""Tests for OAuth functionality."""

import json
from unittest.mock import Mock, patch

from sqlsaber.clients.anthropic import AnthropicClient
from sqlsaber.config.oauth_flow import AnthropicOAuthFlow
from sqlsaber.config.oauth_tokens import OAuthToken, OAuthTokenManager


class TestOAuthToken:
    """Test OAuth token functionality."""

    def test_from_dict(self):
        """Test creating token from dictionary."""
        data = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_at": "2024-12-31T23:59:59Z",
            "token_type": "Bearer",
        }
        token = OAuthToken.from_dict(data)
        assert token.access_token == "access-123"
        assert token.refresh_token == "refresh-456"
        assert token.expires_at == "2024-12-31T23:59:59Z"
        assert token.token_type == "Bearer"

    def test_to_dict(self):
        """Test converting token to dictionary."""
        token = OAuthToken(
            access_token="access-123",
            refresh_token="refresh-456",
            expires_at="2024-12-31T23:59:59Z",
        )
        data = token.to_dict()
        expected = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_at": "2024-12-31T23:59:59Z",
            "token_type": "Bearer",
        }
        assert data == expected

    def test_is_expired_no_expiry(self):
        """Test token without expiry is not expired."""
        token = OAuthToken("access-123", "refresh-456")
        assert not token.is_expired()

    def test_expires_soon_no_expiry(self):
        """Test token without expiry doesn't expire soon."""
        token = OAuthToken("access-123", "refresh-456")
        assert not token.expires_soon()


class TestOAuthTokenManager:
    """Test OAuth token manager functionality."""

    def test_init(self):
        """Test manager initialization."""
        manager = OAuthTokenManager()
        assert manager.service_prefix == "sqlsaber"

    def test_get_service_name(self):
        """Test service name generation."""
        manager = OAuthTokenManager()
        assert manager._get_service_name("anthropic") == "sqlsaber-anthropic-oauth"

    @patch("keyring.get_password")
    def test_get_oauth_token_not_found(self, mock_get):
        """Test getting token when none exists."""
        mock_get.return_value = None
        manager = OAuthTokenManager()
        token = manager.get_oauth_token("anthropic")
        assert token is None

    @patch("keyring.get_password")
    def test_get_oauth_token_success(self, mock_get):
        """Test getting valid token."""
        token_data = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_at": "2099-12-31T23:59:59Z",
            "token_type": "Bearer",
        }
        mock_get.return_value = json.dumps(token_data)

        manager = OAuthTokenManager()
        token = manager.get_oauth_token("anthropic")

        assert token is not None
        assert token.access_token == "access-123"
        assert token.refresh_token == "refresh-456"

    @patch("keyring.set_password")
    def test_store_oauth_token(self, mock_set):
        """Test storing OAuth token."""
        manager = OAuthTokenManager()
        token = OAuthToken("access-123", "refresh-456")

        result = manager.store_oauth_token("anthropic", token)
        assert result is True
        mock_set.assert_called_once()


class TestAnthropicOAuthFlow:
    """Test OAuth flow functionality."""

    def test_init(self):
        """Test OAuth flow initialization."""
        flow = AnthropicOAuthFlow()
        assert flow.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert isinstance(flow.token_manager, OAuthTokenManager)

    def test_generate_pkce(self):
        """Test PKCE generation."""
        flow = AnthropicOAuthFlow()
        verifier, challenge = flow._generate_pkce()

        assert len(verifier) > 0
        assert len(challenge) > 0
        assert verifier != challenge

    def test_create_authorization_url(self):
        """Test authorization URL creation."""
        flow = AnthropicOAuthFlow()
        url, verifier = flow._create_authorization_url()

        assert "claude.ai/oauth/authorize" in url
        assert "client_id=" in url
        assert "code_challenge=" in url
        assert "scope=" in url
        assert len(verifier) > 0

    @patch("sqlsaber.config.oauth_flow.OAuthTokenManager")
    def test_has_valid_authentication_true(self, mock_manager_class):
        """Test valid authentication check."""
        mock_token = Mock()
        mock_token.is_expired.return_value = False

        mock_manager = Mock()
        mock_manager.get_oauth_token.return_value = mock_token
        mock_manager_class.return_value = mock_manager

        flow = AnthropicOAuthFlow()
        assert flow.has_valid_authentication() is True

    @patch("sqlsaber.config.oauth_flow.OAuthTokenManager")
    def test_has_valid_authentication_false(self, mock_manager_class):
        """Test invalid authentication check."""
        mock_manager = Mock()
        mock_manager.get_oauth_token.return_value = None
        mock_manager_class.return_value = mock_manager

        flow = AnthropicOAuthFlow()
        assert flow.has_valid_authentication() is False


class TestAnthropicClientOAuth:
    """Test OAuth integration with AnthropicClient."""

    def test_oauth_headers(self):
        """Test OAuth headers match Claude Code CLI specification."""
        client = AnthropicClient(oauth_token="test-oauth-token")
        headers = client._get_headers()

        # Required OAuth headers
        assert headers["Authorization"] == "Bearer test-oauth-token"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["anthropic-beta"] == "oauth-2025-04-20"
        assert headers["Content-Type"] == "application/json"

        # Claude Code CLI headers
        assert headers["User-Agent"] == "ClaudeCode/1.0 (Anthropic Claude Code CLI)"
        assert headers["Accept"] == "application/json"
        assert headers["X-Client-Name"] == "claude-code"
        assert headers["X-Client-Version"] == "1.0.0"

        # Should not have API key header
        assert "x-api-key" not in headers
