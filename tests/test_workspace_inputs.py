from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any, cast

import pytest

from sqlsaber import WorkspaceResolutionContext

_REPOSITORY_ROOT = Path(__file__).parents[1]


def test_workspace_resolution_context_freezes_authorization_metadata() -> None:
    metadata = {"tenant_id": "acme"}
    context = WorkspaceResolutionContext(
        run_id="run-1",
        conversation_id="conversation-1",
        tool_call_id="tool-1",
        metadata=metadata,
    )

    metadata["tenant_id"] = "changed"
    assert context.metadata == {"tenant_id": "acme"}
    with pytest.raises(TypeError):
        cast(Any, context.metadata)["tenant_id"] = "changed"


def test_workspace_input_boundary_has_no_storage_provider_dependencies() -> None:
    notebook_project = tomllib.loads(
        (_REPOSITORY_ROOT / "plugins/notebook/pyproject.toml").read_text()
    )
    dependencies = "\n".join(notebook_project["project"]["dependencies"]).lower()
    for forbidden in ("nova", "google-cloud", "boto", "s3", "gcs"):
        assert forbidden not in dependencies

    forbidden_imports = {"nova", "google.cloud", "boto3", "botocore"}
    sources = [
        _REPOSITORY_ROOT / "src/sqlsaber/workspace_inputs.py",
        *_REPOSITORY_ROOT.glob("plugins/notebook/src/sqlsaber_notebook/**/*.py"),
    ]
    for source in sources:
        tree = ast.parse(source.read_text(), filename=str(source))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        for module in imported:
            assert not any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_imports
            ), f"Storage-provider coupling in {source}: {module}"
