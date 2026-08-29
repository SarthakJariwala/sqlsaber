"""Public errors raised by the SQLSaber SDK lifecycle."""

from __future__ import annotations


class SQLSaberError(Exception):
    """Base class for SQLSaber SDK errors."""


class SQLSaberClosedError(SQLSaberError):
    """Raised when an operation uses a closed SQLSaber instance."""


class RunInProgressError(SQLSaberError):
    """Raised when an operation conflicts with an active query."""


class ThreadResumeError(SQLSaberError):
    """Base class for failures to resume a stored thread."""


class ThreadNotFoundError(ThreadResumeError):
    """Raised when a requested thread does not exist."""

    def __init__(self, thread_id: str) -> None:
        self.thread_id = thread_id
        super().__init__(f"Thread not found: {thread_id}")


class ThreadResumeHistoryError(ThreadResumeError):
    """Raised when stored message history cannot be loaded safely."""

    def __init__(self, thread_id: str, reason: str) -> None:
        self.thread_id = thread_id
        self.reason = reason
        super().__init__(f"Thread {thread_id} has invalid resume history: {reason}")


class ThreadResumeMetadataError(ThreadResumeError):
    """Raised when stored resume metadata is missing or unsafe."""

    def __init__(self, thread_id: str, reason: str) -> None:
        self.thread_id = thread_id
        self.reason = reason
        super().__init__(f"Thread {thread_id} has invalid resume metadata: {reason}")


class ThreadDatabaseRequiredError(ThreadResumeMetadataError):
    """Raised when automatic resume cannot safely select a database."""


class ThreadDatabaseUnavailableError(ThreadResumeError):
    """Raised when stored configured database names are no longer available."""

    def __init__(self, thread_id: str, database_names: tuple[str, ...]) -> None:
        self.thread_id = thread_id
        self.database_names = database_names
        names = ", ".join(database_names)
        super().__init__(
            f"Thread {thread_id} requires unavailable configured databases: {names}"
        )
