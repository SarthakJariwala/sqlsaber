"""Thread metadata helpers for resumable session state."""

from __future__ import annotations

import json
from typing import Any

type DatabaseSelector = str | list[str]
THREAD_METADATA_VERSION = 1
DATABASE_SELECTOR_KEY = "database_selector"
RESUME_DISABLED_KEY = "resume_disabled"
RESUME_DISABLED_REASON_KEY = "resume_disabled_reason"


def _validate_database_selector(value: Any) -> DatabaseSelector:
    """Validate and normalize a stored database selector."""
    if isinstance(value, str):
        if not value:
            raise ValueError("Thread database selector must not be empty.")
        return value

    if isinstance(value, list):
        if not value:
            raise ValueError("Thread database selector list must not be empty.")
        selector: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError(
                    "Thread database selector list must contain only non-empty strings."
                )
            selector.append(item)
        return selector

    raise ValueError("Thread database selector must be a string or a list of strings.")


def encode_thread_extra_metadata(*, database_selector: DatabaseSelector) -> str:
    """Encode resumable thread metadata as JSON."""
    selector = _validate_database_selector(database_selector)
    return json.dumps(
        {
            "version": THREAD_METADATA_VERSION,
            DATABASE_SELECTOR_KEY: selector,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def encode_thread_resume_disabled_metadata(*, reason: str) -> str:
    """Encode metadata indicating resume requires an explicit database override."""
    if not reason:
        raise ValueError("Thread resume disabled reason must not be empty.")
    return json.dumps(
        {
            "version": THREAD_METADATA_VERSION,
            RESUME_DISABLED_KEY: True,
            RESUME_DISABLED_REASON_KEY: reason,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def resolve_thread_database_selector(
    *, database_name: str | None, extra_metadata: str | None
) -> DatabaseSelector | None:
    """Return the database selector to use when resuming a thread.

    The structured selector in ``extra_metadata`` is authoritative when present.
    Legacy threads without metadata can still resume from a single stored
    database name, but comma-joined legacy multi-database names are rejected
    because resolving them as one name can fail or target the wrong database.
    """
    if extra_metadata is not None:
        try:
            metadata = json.loads(extra_metadata)
        except json.JSONDecodeError as exc:
            raise ValueError("Thread extra_metadata contains invalid JSON.") from exc
        if not isinstance(metadata, dict):
            raise ValueError("Thread extra_metadata must be a JSON object.")

        if metadata.get(RESUME_DISABLED_KEY) is True:
            reason = metadata.get(RESUME_DISABLED_REASON_KEY)
            detail = f" {reason}" if isinstance(reason, str) and reason else ""
            raise ValueError(
                "This thread cannot be resumed automatically because its database "
                f"selector was not saved.{detail} Resume it with explicit "
                "--database/-d options."
            )

        selector = metadata.get(DATABASE_SELECTOR_KEY)
        if selector is not None:
            return _validate_database_selector(selector)

    if database_name and "," in database_name:
        raise ValueError(
            "This thread was saved with multiple databases but does not include "
            "resumable database metadata. Resume it with explicit repeated "
            "--database/-d options."
        )

    return database_name
