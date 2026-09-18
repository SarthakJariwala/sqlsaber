from __future__ import annotations

import subprocess
import sys

from sqlsaber_viz.settings import settings


def test_settings_declaration_is_import_light() -> None:
    script = (
        "import sys\n"
        "import sqlsaber_viz.settings\n"
        "banned = [name for name in "
        "('pydantic_ai', 'sqlsaber_viz.capability', 'sqlsaber_viz.tools') "
        "if name in sys.modules]\n"
        "print(','.join(banned))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_settings_declare_one_model_field() -> None:
    model = settings.field("model")
    assert model.kind == "model"
    assert model.parse("openai:gpt-5-mini") == "openai:gpt-5-mini"
    assert model.parse("  ") is None
