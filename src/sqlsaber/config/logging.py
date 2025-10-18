"""Central logging configuration for SQLSaber using structlog.

This module provides a single entry point `setup_logging()` to configure
structured logging across the project, plus a helper `get_logger()` to
retrieve namespaced loggers.

Defaults:
- JSON logs to a rotating file under the user log directory.
- Optional pretty console logs when `SQLSABER_DEBUG=1` or
  `SQLSABER_LOG_TO_STDERR=1`.

Environment variables:
- `SQLSABER_LOG_LEVEL` (default: INFO)
- `SQLSABER_LOG_FILE` (default: <user_log_dir>/sqlsaber.log)
- `SQLSABER_LOG_TO_STDERR` (0/1, default: 0)
- `SQLSABER_LOG_ROTATION` ("time" or "size", default: "time")
- `SQLSABER_LOG_WHEN` (Timed rotation unit, default: "midnight")
- `SQLSABER_LOG_INTERVAL` (Timed rotation interval, default: 1)
- `SQLSABER_LOG_BACKUP_COUNT` (number of rotated files to keep, default: 14)
- `SQLSABER_LOG_MAX_BYTES` (for size rotation, default: 10485760)
"""

from __future__ import annotations

import logging
import os
from logging import Handler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

import platformdirs
import structlog

_CONFIGURED = False


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def default_log_dir() -> Path:
    return Path(platformdirs.user_log_dir("sqlsaber", "sqlsaber"))


def default_log_file() -> Path:
    return default_log_dir() / "sqlsaber.log"


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Return a structlog logger bound to `name`.

    Prefer using this over the stdlib `logging.getLogger` in new code.
    """
    return structlog.get_logger(name)


def _build_file_handler(log_path: Path, level: int) -> Handler:
    rotation = os.getenv("SQLSABER_LOG_ROTATION", "time").strip().lower()

    # Formatter that renders as JSON for files
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        foreign_pre_chain=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
    )

    if rotation == "size":
        max_bytes = int(os.getenv("SQLSABER_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
        backup_count = int(os.getenv("SQLSABER_LOG_BACKUP_COUNT", "5"))
        handler: Handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
    else:
        when = os.getenv("SQLSABER_LOG_WHEN", "midnight")
        interval = int(os.getenv("SQLSABER_LOG_INTERVAL", "1"))
        backup_count = int(os.getenv("SQLSABER_LOG_BACKUP_COUNT", "14"))
        handler = TimedRotatingFileHandler(
            log_path,
            when=when,
            interval=interval,
            backupCount=backup_count,
            encoding="utf-8",
            utc=True,
        )

    handler.setLevel(level)
    handler.setFormatter(json_formatter)
    return handler


def _build_console_handler(level: int) -> Handler:
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ],
    )

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(console_formatter)
    return ch


def setup_logging(*, force: bool = False) -> None:
    """Configure structlog + stdlib logging with sensible defaults.

    Call this early in the CLI startup. It's safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    # Resolve level
    level_name = os.getenv("SQLSABER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Log file path
    log_file_env = os.getenv("SQLSABER_LOG_FILE")
    log_path = Path(log_file_env) if log_file_env else default_log_file()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Whether to also log to console (stderr)
    to_console = _to_bool(os.getenv("SQLSABER_LOG_TO_STDERR")) or _to_bool(
        os.getenv("SQLSABER_DEBUG")
    )

    # Configure structlog to hand off to stdlib formatting
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Build handlers
    handlers: list[Handler] = []
    handlers.append(_build_file_handler(log_path, level))
    if to_console:
        handlers.append(_build_console_handler(level))

    # Install handlers on root logger
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        root.addHandler(h)

    # Capture warnings too
    logging.captureWarnings(True)

    # Pre-bind useful context
    try:
        from importlib.metadata import PackageNotFoundError, version  # type: ignore

        try:
            ver = version("sqlsaber")
        except PackageNotFoundError:  # during dev
            ver = "dev"
    except Exception:
        ver = "unknown"

    structlog.contextvars.bind_contextvars(app="sqlsaber", version=ver)

    _CONFIGURED = True


__all__ = [
    "setup_logging",
    "get_logger",
    "default_log_dir",
    "default_log_file",
]
