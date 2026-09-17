"""Run CLI plugin configuration checks with disposable settings and no cloud calls.

Run from the repository root: uv run python scripts/verify_plugin_settings.py
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sqlsaber-plugin-proof-") as temporary:
        home = Path(temporary)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(
                ("SQLSABER_", "E2B_", "DAYTONA_", "MODAL_", "SPRITES_")
            )
            and key != "FORCE_COLOR"
        }
        env.update(
            HOME=str(home),
            XDG_CONFIG_HOME=str(home / "config"),
            XDG_DATA_HOME=str(home / "data"),
            XDG_CACHE_HOME=str(home / "cache"),
            PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        )

        def run(
            *args: str, code: int = 0, extra_env: dict[str, str] | None = None
        ) -> str:
            result = subprocess.run(
                [sys.executable, "-m", "sqlsaber", "plugins", *args],
                env=env | (extra_env or {}),
                input="",
                text=True,
                capture_output=True,
                timeout=30,
            )
            assert result.returncode == code, result.stdout + result.stderr
            output = result.stdout + result.stderr
            assert "\x1b" not in output
            print(f"PASS saber plugins {' '.join(args)} (exit {code})")
            return output.replace("\\_", "_")

        path_result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from sqlsaber.config.plugins import PluginConfigStore; print(PluginConfigStore().path)",
            ],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        path = Path(path_result.stdout.strip())
        if not path.resolve().is_relative_to(home.resolve()):
            raise SystemExit("Cannot isolate the config directory on this platform")
        listing = run("list")
        assert "notebook" in listing and "sandbox" in listing
        run("setup", "notebook", "--set", "backend=docker", "--set", "memory_mb=3072")
        output = run("show", "notebook")
        assert "3072" in output and "saved" in output
        saved = json.loads(path.read_text())
        assert saved["plugins"]["notebook"]["settings"] == {
            "backend": "docker",
            "memory_mb": 3072,
        }
        before = path.read_bytes()
        output = run("set", "notebook", "memory_mb", "-1", code=2)
        assert "positive" in output and path.read_bytes() == before
        run("set", "notebook", "not_a_setting", "123", code=2)
        assert path.read_bytes() == before
        output = run(
            "show", "notebook", extra_env={"SQLSABER_NOTEBOOK_BACKEND": "modal"}
        )
        assert "environment (SQLSABER_NOTEBOOK_BACKEND)" in output
        assert "modal" in output and "docker" in output
        assert path.read_bytes() == before
        run("disable", "notebook")
        assert json.loads(path.read_text())["plugins"]["notebook"]["enabled"] is False
        run("enable", "notebook")
        run("unset", "notebook", "memory_mb")
        assert (
            "memory_mb"
            not in json.loads(path.read_text())["plugins"]["notebook"]["settings"]
        )
        run("setup", "sandbox", "--set", "provider=e2b", code=2)
        run("setup", "sandbox", "--set", "provider=e2b", "--yes")
        output = run(
            "show", "sandbox", extra_env={"E2B_API_KEY": "proof-secret-not-a-real-key"}
        )
        assert "configured" in output
        assert "proof-secret-not-a-real-key" not in output + path.read_text()
        run("setup", "sandbox", "--set", "provider=docker")
        run("setup", "notebook", code=2)
        print(
            "PASS persisted read-back, precedence, validation, redaction, enable/disable, headless setup"
        )
        print(
            "No provider resources were created. Disposable settings removed on exit."
        )


if __name__ == "__main__":
    main()
