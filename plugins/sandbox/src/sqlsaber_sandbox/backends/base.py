"""Native sandbox transport contract; interpreter semantics live above it."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..config import SandboxConfig


class SandboxError(RuntimeError):
    """Execution or transport failure, never an instruction to replay code."""


class SessionLost(SandboxError):
    """The live interpreter cannot be safely continued."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int


class SandboxBackend(Protocol):
    """One owned environment. Imports and provisioning must remain lazy.

    Commands return complete output or raise; transfers preserve binary bytes.
    start_controller must retain its process and drain handles until close.
    close must work after partial open and keep failed deletions retryable.
    The caller owns timeouts for provisioning and transfers; execute enforces
    its supplied timeout. A session owns and closes any injected backend.
    """

    async def open(self, config: SandboxConfig) -> None: ...

    async def execute(self, command: str, *, timeout: float) -> CommandResult: ...

    async def upload(self, data: bytes, path: str) -> None: ...

    async def download(self, path: str) -> bytes: ...

    async def start_controller(self, argv: Sequence[str]) -> None: ...

    async def close(self) -> None: ...
