"""OAuth token management for SQLSaber."""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import keyring
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class OAuthToken:
    """Represents an OAuth token with metadata."""

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: str | None = None,
        token_type: str = "Bearer",
    ):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
        self.token_type = token_type

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthToken":
        """Create token from dictionary."""
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data.get("expires_at"),
            token_type=data.get("token_type", "Bearer"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert token to dictionary."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "token_type": self.token_type,
        }

    def is_expired(self) -> bool:
        """Check if the token is expired."""
        if not self.expires_at:
            return False

        try:
            expires_dt = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) >= expires_dt
        except (ValueError, AttributeError):
            # If we can't parse the expiration, assume expired for safety
            return True

    def expires_soon(self, buffer_seconds: int = 300) -> bool:
        """Check if token expires within buffer_seconds."""
        if not self.expires_at:
            return False

        try:
            expires_dt = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))

            return (
                datetime.now(timezone.utc) + timedelta(seconds=buffer_seconds)
            ) >= expires_dt
        except (ValueError, AttributeError):
            return True


class OAuthTokenManager:
    """Manages OAuth tokens with secure storage and refresh logic."""

    def __init__(self):
        self.service_prefix = "sqlsaber"

    def get_oauth_token(self, provider: str) -> OAuthToken | None:
        """Get OAuth token for the specified provider."""
        service_name = self._get_service_name(provider)

        try:
            token_data = keyring.get_password(service_name, provider)
            if not token_data:
                return None

            # Parse the stored JSON
            data = json.loads(token_data)
            token = OAuthToken.from_dict(data)

            # Check if token is expired
            if token.is_expired():
                console.print(
                    f"OAuth token for {provider} has expired and needs refresh",
                    style="dim yellow",
                )
                return token  # Return anyway for refresh attempt

            if token.expires_soon():
                console.print(
                    f"OAuth token for {provider} expires soon, consider refreshing",
                    style="dim yellow",
                )

            return token

        except Exception as e:
            logger.warning(f"Failed to retrieve OAuth token for {provider}: {e}")
            return None

    def store_oauth_token(self, provider: str, token: OAuthToken) -> bool:
        """Store OAuth token securely."""
        service_name = self._get_service_name(provider)

        try:
            token_data = json.dumps(token.to_dict())
            keyring.set_password(service_name, provider, token_data)
            console.print(f"OAuth token for {provider} stored securely", style="green")
            return True
        except Exception as e:
            logger.error(f"Failed to store OAuth token for {provider}: {e}")
            console.print(
                f"Warning: Could not store OAuth token in keyring: {e}",
                style="yellow",
            )
            return False

    def update_oauth_token(
        self, provider: str, access_token: str, expires_at: str | None = None
    ) -> bool:
        """Update only the access token (keep refresh token)."""
        existing_token = self.get_oauth_token(provider)
        if not existing_token:
            console.print(
                f"No existing OAuth token found for {provider}", style="yellow"
            )
            return False

        # Update the access token while preserving refresh token
        updated_token = OAuthToken(
            access_token=access_token,
            refresh_token=existing_token.refresh_token,
            expires_at=expires_at,
            token_type=existing_token.token_type,
        )

        return self.store_oauth_token(provider, updated_token)

    def remove_oauth_token(self, provider: str) -> bool:
        """Remove OAuth token from storage."""
        service_name = self._get_service_name(provider)

        try:
            keyring.delete_password(service_name, provider)
            console.print(f"OAuth token for {provider} removed", style="green")
            return True
        except keyring.errors.PasswordDeleteError:
            # Token doesn't exist
            return True
        except Exception as e:
            logger.error(f"Failed to remove OAuth token for {provider}: {e}")
            console.print(f"Warning: Could not remove OAuth token: {e}", style="yellow")
            return False

    def has_oauth_token(self, provider: str) -> bool:
        """Check if OAuth token exists for provider."""
        return self.get_oauth_token(provider) is not None

    def _get_service_name(self, provider: str) -> str:
        """Get the keyring service name for OAuth tokens."""
        return f"{self.service_prefix}-{provider}-oauth"
