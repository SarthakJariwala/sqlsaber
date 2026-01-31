"""Sandboxed Python execution tools."""

import base64
import json
import os
import re
import shlex
import tempfile
from typing import Iterable

from pydantic_ai import RunContext

from sqlsaber.tools.base import Tool
from sqlsaber.tools.registry import ToolRegistry
from sqlsaber.utils.json_utils import json_dumps

PROVIDER_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "daytona": ("DAYTONA_API_KEY",),
    "e2b": ("E2B_API_KEY",),
    "sprites": ("SPRITES_TOKEN",),
    "hopx": ("HOPX_API_KEY",),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
    "cloudflare": ("CLOUDFLARE_SANDBOX_BASE_URL", "CLOUDFLARE_API_TOKEN"),
}

DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 600
MAX_CODE_CHARS = 20000
MAX_REQUIREMENTS = 10
MAX_REQUIREMENT_CHARS = 200
TOOL_OUTPUT_FILE_PATTERN = re.compile(r"^result_[A-Za-z0-9._-]+\.json$")


def _has_env_values(names: Iterable[str]) -> bool:
    return all(os.getenv(name) for name in names)


def _modal_config_available() -> bool:
    config_path = os.getenv("MODAL_CONFIG_PATH")
    modal_config = (
        os.path.expanduser(config_path)
        if config_path
        else os.path.expanduser("~/.modal.toml")
    )
    return os.path.isfile(modal_config)


def sandbox_providers_available() -> bool:
    """Return True when at least one sandbox provider is configured."""

    for env_names in PROVIDER_ENV_REQUIREMENTS.values():
        if _has_env_values(env_names):
            return True

    if _modal_config_available():
        return True

    return False


def register_tools(registry: ToolRegistry | None = None):
    """Register sandbox tools when providers are available.

    Returns list of tool classes when registration should occur.
    """

    if not sandbox_providers_available():
        return None

    tool_classes = [RunPythonTool]
    if registry is not None:
        for tool_class in tool_classes:
            if tool_class().name in registry.list_tools():
                return None
            registry.register(tool_class)
        return tool_classes

    return tool_classes


def _build_python_command(code: str) -> str:
    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return (
        'python -c "import base64; '
        f"code=base64.b64decode('{encoded}').decode('utf-8'); "
        "exec(compile(code, '<sandbox>', 'exec'))\""
    )


class RunPythonTool(Tool):
    """Run Python code in a sandboxed environment."""

    requires_ctx = True

    @property
    def name(self) -> str:
        return "run_python"

    async def execute(
        self,
        ctx: RunContext,
        code: str,
        requirements: list[str] | None = None,
        file: str | None = None,
        timeout_seconds: int | None = None,
    ) -> str:
        """Execute Python code inside a remote sandbox.

        Notes:
            - To use a SQL result file, you MUST pass `file` parameter.
              The file is uploaded to `/tmp/<file>` inside the sandbox.
            - Only stdout/stderr is returned. Use `print(...)` (or write to stdout)
              to see output.

        Args:
            code: Python code to execute.
            requirements: Optional pip requirements to install before execution.
            file: Optional file key from a previous tool output to upload.
                When provided, the file is uploaded to `/tmp/<file>`.
            timeout_seconds: Optional timeout for sandbox execution (seconds).
        """

        if not sandbox_providers_available():
            return json_dumps(
                {
                    "error": (
                        "No sandbox provider configured. Set at least one provider API "
                        "key (e.g., E2B_API_KEY, DAYTONA_API_KEY, SPRITES_TOKEN, "
                        "HOPX_API_KEY, MODAL_TOKEN_ID/MODAL_TOKEN_SECRET, or "
                        "CLOUDFLARE_SANDBOX_BASE_URL/CLOUDFLARE_API_TOKEN)."
                    )
                }
            )

        if not code or not code.strip():
            return json_dumps({"error": "No Python code provided."})

        if len(code) > MAX_CODE_CHARS:
            return json_dumps(
                {"error": f"Python code too large (max {MAX_CODE_CHARS} characters)."}
            )

        cleaned_requirements = [
            req.strip() for req in (requirements or []) if req.strip()
        ]
        if len(cleaned_requirements) > MAX_REQUIREMENTS:
            return json_dumps(
                {"error": (f"Too many requirements (max {MAX_REQUIREMENTS}).")}
            )

        if any(len(req) > MAX_REQUIREMENT_CHARS for req in cleaned_requirements):
            return json_dumps({"error": ("Requirement entry too long.")})

        timeout_value = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        if timeout_value < 1:
            timeout_value = 1
        if timeout_value > MAX_TIMEOUT_SECONDS:
            timeout_value = MAX_TIMEOUT_SECONDS

        try:
            from sandboxes import Sandbox

            async with Sandbox.create(timeout=timeout_value) as sandbox:
                remote_data_path = None
                if file:
                    if not TOOL_OUTPUT_FILE_PATTERN.match(file):
                        return json_dumps(
                            {
                                "error": "Invalid data file key format.",
                            }
                        )
                    tool_call_id = file.removeprefix("result_").removesuffix(".json")
                    payload = _find_tool_output_payload(ctx, tool_call_id)
                    if payload is None:
                        return json_dumps(
                            {
                                "error": "Tool output not found in message history.",
                            }
                        )
                    remote_data_path = f"/tmp/{file}"
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix=".json",
                            delete=False,
                            encoding="utf-8",
                        ) as temp_file:
                            temp_file.write(json_dumps(payload))
                            temp_path = temp_file.name
                        await sandbox.upload(temp_path, remote_data_path)
                    finally:
                        if temp_path:
                            try:
                                os.unlink(temp_path)
                            except OSError:
                                pass

                if cleaned_requirements:
                    install_command = (
                        "python -m pip install --quiet --disable-pip-version-check "
                        "--no-input "
                        + " ".join(shlex.quote(req) for req in cleaned_requirements)
                    )
                    install_result = await sandbox.execute(install_command)
                    if install_result.exit_code != 0:
                        return json_dumps(
                            {
                                "error": "Failed to install requirements.",
                                "exit_code": install_result.exit_code,
                                "stdout": install_result.stdout,
                                "stderr": install_result.stderr,
                            }
                        )

                command = _build_python_command(code)
                result = await sandbox.execute(command)

                return json_dumps(
                    {
                        "success": result.success,
                        "exit_code": result.exit_code,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "data_path": remote_data_path,
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive catch-all
            return json_dumps({"error": f"Python sandbox execution failed: {exc}"})


def _find_tool_output_payload(ctx: RunContext, tool_call_id: str) -> dict | None:
    for message in reversed(ctx.messages):
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", "") not in (
                "tool-return",
                "builtin-tool-return",
            ):
                continue
            if getattr(part, "tool_call_id", None) != tool_call_id:
                continue
            content = getattr(part, "content", None)
            if isinstance(content, dict):
                return content
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    return {"result": content}
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed}
    return None
