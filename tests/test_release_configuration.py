import json
import tomllib
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).parents[1]
_CONFIG_PATH = _REPOSITORY_ROOT / ".github" / "release-please-config.json"
_EXPECTED_LOCKFILES = {
    ".": {"uv.lock", "plugins/notebook/uv.lock"},
    "plugins/notebook": {"uv.lock", "plugins/notebook/uv.lock"},
    "plugins/sandbox": {"uv.lock", "plugins/sandbox/uv.lock"},
    "plugins/viz": {"uv.lock", "plugins/viz/uv.lock"},
}


def _repository_path(package_path: str, extra_file_path: str) -> Path:
    if extra_file_path.startswith("/"):
        return Path(extra_file_path.removeprefix("/"))
    if package_path == ".":
        return Path(extra_file_path)
    return Path(package_path) / extra_file_path


def test_release_please_updates_lockfiles_atomically() -> None:
    config: dict[str, Any] = json.loads(_CONFIG_PATH.read_text())

    for package_path, expected_lockfiles in _EXPECTED_LOCKFILES.items():
        package = config["packages"][package_path]
        package_name = package["package-name"]
        lockfile_entries = {
            _repository_path(package_path, extra_file["path"]): extra_file
            for extra_file in package["extra-files"]
        }

        assert {path.as_posix() for path in lockfile_entries} == expected_lockfiles

        for lockfile_path, extra_file in lockfile_entries.items():
            assert extra_file["type"] == "toml"
            # release-please's formatting-preserving TOML parser wraps scalar
            # values, so filters address the wrapped name through `.value`.
            assert extra_file["jsonpath"] == (
                f"$.package[?(@.name.value=='{package_name}')].version"
            )

            lockfile = tomllib.loads((_REPOSITORY_ROOT / lockfile_path).read_text())
            matching_packages = [
                locked_package
                for locked_package in lockfile["package"]
                if locked_package["name"] == package_name
            ]
            assert len(matching_packages) == 1
