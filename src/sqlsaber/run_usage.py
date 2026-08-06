"""Run-local usage-limit intent shared with nested SQLSaber agents."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from pydantic_ai.usage import UsageLimits

_EXPLICIT_USAGE_LIMITS: ContextVar[UsageLimits | None] = ContextVar(
    "sqlsaber_explicit_usage_limits",
    default=None,
)


def current_usage_limits() -> UsageLimits | None:
    """Return limits explicitly supplied to the current SQLSaber run, if any."""
    return _EXPLICIT_USAGE_LIMITS.get()


@contextmanager
def bind_usage_limits(usage_limits: UsageLimits | None) -> Iterator[None]:
    """Make a run's explicit limit selection available to nested agents."""
    token = _EXPLICIT_USAGE_LIMITS.set(usage_limits)
    try:
        yield
    finally:
        _EXPLICIT_USAGE_LIMITS.reset(token)
