"""Helpers for resolving text inputs that may refer to files."""

import errno
import os
from pathlib import Path


def sanitize_terminal_text(value: object) -> str:
    """Make untrusted text safe to pass to an ANSI-aware terminal renderer."""
    safe: list[str] = []
    for char in str(value):
        codepoint = ord(char)
        if char in {"\n", "\t"} or 0x20 <= codepoint < 0x7F or codepoint >= 0xA0:
            safe.append(char)
        elif codepoint < 0x20 or codepoint == 0x7F:
            safe.append(f"\\x{codepoint:02X}")
        elif codepoint < 0xA0:
            safe.append(f"\\u{codepoint:04X}")
        else:
            safe.append(char)
    return "".join(safe)


def resolve_text_input(value: str | Path | None) -> str | None:
    """Resolve a text input that may be a file path.

    - If `value` is a path (or a string path) to an existing file, read it.
    - Otherwise treat it as literal text.

    Passing an empty string returns an empty string.
    """

    if value is None:
        return None

    if isinstance(value, str) and not value.strip():
        # Avoid interpreting empty/whitespace strings as filesystem paths.
        return ""

    # If it's obviously literal text (contains newlines or is too long to be
    # a valid filename), skip filesystem probing to avoid ENAMETOOLONG errors.
    if isinstance(value, str):
        if "\n" in value or "\r" in value:
            return value

        seps = {os.sep}
        if os.altsep:
            seps.add(os.altsep)

        has_sep = any(sep in value for sep in seps)
        # 255 is the max filename component length on most POSIX filesystems
        if not has_sep and len(value) > 255:
            return value

    candidate = (
        value.expanduser() if isinstance(value, Path) else Path(value).expanduser()
    )

    try:
        if candidate.exists():
            if candidate.is_dir():
                raise ValueError(
                    f"Input path '{candidate}' is a directory, expected a file"
                )
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        # Treat as literal text when path probing fails due to invalid path
        if e.errno in (errno.ENAMETOOLONG, errno.EINVAL):
            return str(value)
        raise

    return str(value)
