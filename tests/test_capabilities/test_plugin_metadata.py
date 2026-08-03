"""Compatibility checks for in-repository capability plugins."""

import tomllib
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).parents[2]


def _project_metadata(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)["project"]


def test_plugins_require_capabilities_compatible_sqlsaber() -> None:
    core = _project_metadata(_REPOSITORY_ROOT / "pyproject.toml")
    core_version = tuple(int(part) for part in core["version"].split("."))
    assert core_version >= (0, 69, 0)

    minimum_versions = {
        "sandbox": "0.69.0",
        "viz": "0.69.0",
        "notebook": "0.70.0",
    }
    for plugin, minimum_version in minimum_versions.items():
        metadata = _project_metadata(
            _REPOSITORY_ROOT / "plugins" / plugin / "pyproject.toml"
        )
        sqlsaber_requirements = [
            requirement
            for requirement in metadata["dependencies"]
            if requirement.startswith("sqlsaber")
        ]
        assert sqlsaber_requirements == [f"sqlsaber>={minimum_version}"]
