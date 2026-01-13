"""Check for available updates on PyPI.

This is best-effort and must never fail the CLI.
"""

import asyncio
import os
from dataclasses import dataclass
from importlib import metadata

import httpx
from rich.console import Console

from sqlsaber.config.logging import get_logger
from sqlsaber.theme.manager import create_console

PACKAGE_NAME = "sqlsaber"
ENV_SKIP_VERSION_CHECK = "SQLSABER_SKIP_VERSION_CHECK"
PYPI_URL = f"https://pypi.org/pypi/{PACKAGE_NAME}/json"

_LOG = get_logger(__name__)
_SCHEDULED = False


@dataclass(frozen=True)
class VersionInfo:
    current: str
    latest: str


def _env_disables_check() -> bool:
    value = os.getenv(ENV_SKIP_VERSION_CHECK)
    if value is None:
        return False

    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    return True


def _get_version_from_pyproject() -> str | None:
    try:
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            return None

        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = data.get("project", {}).get("version")
        return version if isinstance(version, str) else None
    except Exception:
        return None


def _get_current_version() -> str | None:
    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return _get_version_from_pyproject()
    except Exception:
        return None


async def _fetch_latest_version() -> str | None:
    timeout = httpx.Timeout(connect=1.0, read=1.5, write=1.0, pool=1.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(PYPI_URL)
            resp.raise_for_status()
            data = resp.json()
            version = data.get("info", {}).get("version")
            return version if isinstance(version, str) else None
    except Exception:
        return None


def _parse_version(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in v.split("."):
        num = ""
        for ch in part:
            if ch.isdigit():
                num += ch
            else:
                break
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _is_newer(latest: str, current: str) -> bool:
    try:
        from packaging.version import Version

        return Version(latest) > Version(current)
    except Exception:
        return _parse_version(latest) > _parse_version(current)


def _print_update_notice(console: Console) -> None:
    console.print("A new version is now available!\n")
    console.print(f"Run: uv tool update {PACKAGE_NAME}\n")


async def _check_and_notify(console: Console) -> None:
    current = _get_current_version()
    if not current:
        return

    latest = await _fetch_latest_version()
    if not latest:
        return

    if _is_newer(latest, current):
        _LOG.info("update_check.available", current=current, latest=latest)
        _print_update_notice(console)


async def _run_safely(console: Console) -> None:
    try:
        await _check_and_notify(console)
    except Exception:
        return


def schedule_update_check(console: Console | None = None) -> None:
    """Schedule a non-blocking update check.

    Safe to call multiple times; only schedules once per process.
    """

    global _SCHEDULED
    if _SCHEDULED:
        return

    if _env_disables_check():
        return

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    _SCHEDULED = True
    asyncio.create_task(_run_safely(console or create_console()))
