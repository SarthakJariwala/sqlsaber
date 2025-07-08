"""Synchronous OAuth flow management for Anthropic Claude Pro authentication."""

import base64
import hashlib
import logging
import secrets
import urllib.parse
import webbrowser
from datetime import datetime, timezone

import httpx
import questionary
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .oauth_tokens import OAuthToken, OAuthTokenManager

console = Console()
logger = logging.getLogger(__name__)


CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"


class AnthropicOAuthFlow:
    """Handles the complete OAuth flow for Anthropic Claude Pro authentication."""

    def __init__(self):
        self.client_id = CLIENT_ID
        self.token_manager = OAuthTokenManager()

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
            .decode("utf-8")
            .rstrip("=")
        )
        return verifier, challenge

    def _create_authorization_url(self) -> tuple[str, str]:
        """Create OAuth authorization URL with PKCE."""
        verifier, challenge = self._generate_pkce()

        params = {
            "code": "true",
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": "https://console.anthropic.com/oauth/code/callback",
            "scope": "org:create_api_key user:profile user:inference",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
        }

        url = "https://claude.ai/oauth/authorize?" + urllib.parse.urlencode(params)
        return url, verifier

    def _exchange_code_for_tokens(self, code: str, verifier: str) -> dict[str, str]:
        """Exchange authorization code for access and refresh tokens."""
        # Handle the code format (may have # separator for state)
        code_parts = code.split("#")
        auth_code = code_parts[0]
        state = code_parts[1] if len(code_parts) > 1 else verifier

        data = {
            "code": auth_code,
            "state": state,
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "redirect_uri": "https://console.anthropic.com/oauth/code/callback",
            "code_verifier": verifier,
        }

        with httpx.Client() as client:
            response = client.post(
                "https://console.anthropic.com/v1/oauth/token",
                headers={"Content-Type": "application/json"},
                json=data,
            )

            if not response.is_success:
                error_msg = (
                    f"Token exchange failed: {response.status_code} {response.text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

            return response.json()

    def refresh_access_token(self, refresh_token: str) -> dict[str, str]:
        """Refresh access token using refresh token."""
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        with httpx.Client() as client:
            response = client.post(
                "https://console.anthropic.com/v1/oauth/token",
                headers={"Content-Type": "application/json"},
                json=data,
            )

            if not response.is_success:
                error_msg = (
                    f"Token refresh failed: {response.status_code} {response.text}"
                )
                logger.error(error_msg)
                raise Exception(error_msg)

            return response.json()

    def authenticate(self) -> bool:
        """Complete OAuth authentication flow."""
        console.print(
            "\n[bold blue]Claude Pro/Max Subscription Authentication[/bold blue]"
        )
        console.print(
            "This will open your web browser to authenticate with your Claude subscription.\n"
        )

        # Check if user wants to proceed
        if not questionary.confirm(
            "Continue with browser-based authentication?", default=True
        ).ask():
            console.print("[yellow]Authentication cancelled.[/yellow]")
            return False

        try:
            # Step 1: Create authorization URL
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Preparing authentication...", total=None)

                auth_url, verifier = self._create_authorization_url()
                progress.update(task, description="Opening browser...")

                # Open browser for user authorization
                webbrowser.open(auth_url)

            console.print("\n[green]✓[/green] Browser opened for authentication")
            console.print(
                "[dim]If your browser didn't open automatically, visit this URL:[/dim]"
            )
            console.print(f"[dim]{auth_url}[/dim]\n")

            # Get authorization code from user
            console.print("After authorizing, you'll be redirected to a callback URL.")
            console.print(
                "Copy the 'code' that shows up on your screen and paste it here."
            )

            auth_code = questionary.text(
                "Enter the authorization code:",
                validate=lambda x: len(x.strip()) > 0
                or "Authorization code is required",
            ).ask()

            if not auth_code:
                console.print("[yellow]Authentication cancelled.[/yellow]")
                return False

            # Step 2: Exchange code for tokens
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Exchanging code for tokens...", total=None)

                tokens = self._exchange_code_for_tokens(auth_code.strip(), verifier)

                # Calculate expiration time if provided
                expires_at = None
                if "expires_in" in tokens:
                    expires_in = int(tokens["expires_in"])
                    expires_dt = datetime.now(timezone.utc).timestamp() + expires_in
                    expires_at = datetime.fromtimestamp(
                        expires_dt, timezone.utc
                    ).isoformat()

                # Store tokens
                oauth_token = OAuthToken(
                    access_token=tokens["access_token"],
                    refresh_token=tokens["refresh_token"],
                    expires_at=expires_at,
                )

                if self.token_manager.store_oauth_token("anthropic", oauth_token):
                    console.print(
                        "\n[bold green]✓ Authentication successful![/bold green]"
                    )
                    console.print(
                        "Your Claude Pro/Max subscription is now configured for SQLSaber."
                    )
                    return True
                else:
                    console.print("[red]✗ Failed to store authentication tokens.[/red]")
                    return False

        except KeyboardInterrupt:
            console.print("\n[yellow]Authentication cancelled by user.[/yellow]")
            return False
        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            console.print(f"[red]✗ Authentication failed: {str(e)}[/red]")
            return False

    def refresh_token_if_needed(self) -> OAuthToken | None:
        """Refresh OAuth token if it's expired or expiring soon."""
        current_token = self.token_manager.get_oauth_token("anthropic")
        if not current_token:
            return None

        # If token is not expired and not expiring soon, return it as-is
        if not current_token.is_expired() and not current_token.expires_soon():
            return current_token

        # Attempt to refresh
        try:
            console.print("Refreshing OAuth token...", style="dim")
            new_tokens = self.refresh_access_token(current_token.refresh_token)

            # Calculate new expiration time
            expires_at = None
            if "expires_in" in new_tokens:
                expires_in = int(new_tokens["expires_in"])
                expires_dt = datetime.now(timezone.utc).timestamp() + expires_in
                expires_at = datetime.fromtimestamp(
                    expires_dt, timezone.utc
                ).isoformat()

            # Create new token object
            refreshed_token = OAuthToken(
                access_token=new_tokens["access_token"],
                refresh_token=new_tokens.get(
                    "refresh_token", current_token.refresh_token
                ),
                expires_at=expires_at,
            )

            # Store the refreshed token
            if self.token_manager.store_oauth_token("anthropic", refreshed_token):
                console.print("OAuth token refreshed successfully", style="green")
                return refreshed_token
            else:
                console.print("Failed to store refreshed token", style="yellow")
                return current_token

        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            console.print(
                "Token refresh failed. You may need to re-authenticate.", style="yellow"
            )
            return current_token

    def remove_authentication(self) -> bool:
        """Remove stored OAuth authentication."""
        return self.token_manager.remove_oauth_token("anthropic")

    def has_valid_authentication(self) -> bool:
        """Check if valid OAuth authentication exists."""
        token = self.token_manager.get_oauth_token("anthropic")
        return token is not None and not token.is_expired()
