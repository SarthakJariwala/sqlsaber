"""SQLsaber-owned durable credentials for OpenAI Codex subscription auth."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import platformdirs
from pydantic import Field, TypeAdapter, ValidationError
from pydantic_ai.providers.openai_codex import (
    OpenAICodexCredentialSource,
    OpenAICodexCredentials,
)

_CREDENTIALS_VERSION = 1


@dataclass(frozen=True, slots=True)
class _StoredCredentials:
    version: Literal[1]
    access_token: Annotated[str, Field(min_length=1)]
    refresh_token: Annotated[str, Field(min_length=1)]
    account_id: Annotated[str, Field(min_length=1)]


_STORED_CREDENTIALS = TypeAdapter(_StoredCredentials)


class OpenAICodexAuthError(ValueError):
    """A local Codex credential set cannot authenticate a model."""


class PreflightOpenAICodexCredentialSource(
    OpenAICodexCredentialSource,
    Protocol,
):
    """Codex credentials that can be checked before entering the HTTP stack."""

    def preflight(self) -> None:
        """Raise an actionable error when local credentials cannot be loaded."""
        ...


class OpenAICodexCredentialStore(PreflightOpenAICodexCredentialSource):
    """Persist one Codex credential set in SQLsaber's private app data.

    Pydantic AI serializes refreshes inside one process. Its credential-source
    protocol has no cross-process lock, so concurrent SQLsaber processes can race
    while rotating a single-use refresh token.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (
            Path(platformdirs.user_data_dir("sqlsaber", "sqlsaber"))
            / "openai_codex_credentials.json"
        )

    async def load(self) -> OpenAICodexCredentials:
        """Load and validate SQLsaber's stored credential set."""

        return self._load()

    def preflight(self) -> None:
        """Validate local credentials before the model enters the HTTP stack."""

        self._load()

    def _load(self) -> OpenAICodexCredentials:
        self._check_private_file()
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            raise OpenAICodexAuthError(
                "No SQLsaber OpenAI Codex login was found. Run "
                "`saber auth login openai-codex`."
            ) from None
        except (OSError, json.JSONDecodeError) as exc:
            raise OpenAICodexAuthError(
                f"Could not read SQLsaber OpenAI Codex credentials at "
                f"`{self.path}`: {exc}. Run `saber auth login openai-codex` again."
            ) from exc
        return self._parse(data)

    async def save(self, credentials: OpenAICodexCredentials) -> None:
        """Atomically persist a complete, freshly rotated credential set."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            self.path.parent.chmod(0o700)

        payload = {
            "version": _CREDENTIALS_VERSION,
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "account_id": credentials.account_id,
        }
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w") as temporary_file:
                json.dump(payload, temporary_file)
                temporary_file.write("\n")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            if os.name != "nt":
                temporary_path.chmod(0o600)
            os.replace(temporary_path, self.path)
            if os.name != "nt":
                self.path.chmod(0o600)
                directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

    def is_configured(self) -> bool:
        """Return whether a regular, non-symlink credential file exists."""

        return self.path.is_file() and not self.path.is_symlink()

    def delete(self) -> bool:
        """Delete SQLsaber's stored credentials if present."""

        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True

    def _check_private_file(self) -> None:
        try:
            file_stat = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise OpenAICodexAuthError(
                f"Refusing to read OpenAI Codex credentials from non-regular file "
                f"`{self.path}`."
            )
        if os.name != "nt":
            if stat.S_IMODE(file_stat.st_mode) & 0o077:
                raise OpenAICodexAuthError(
                    f"OpenAI Codex credentials at `{self.path}` are readable by "
                    f"other users. Run `chmod 600 {self.path}` and retry."
                )
            if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
                raise OpenAICodexAuthError(
                    f"OpenAI Codex credentials at `{self.path}` are not owned by "
                    "the current user."
                )

    def _parse(self, data: object) -> OpenAICodexCredentials:
        try:
            record = _STORED_CREDENTIALS.validate_python(data)
        except ValidationError:
            raise self._malformed() from None
        return OpenAICodexCredentials(
            access_token=record.access_token,
            refresh_token=record.refresh_token,
            account_id=record.account_id,
        )

    def _malformed(self) -> OpenAICodexAuthError:
        return OpenAICodexAuthError(
            f"Malformed SQLsaber OpenAI Codex credentials at `{self.path}`. "
            "Run `saber auth login openai-codex` again."
        )
