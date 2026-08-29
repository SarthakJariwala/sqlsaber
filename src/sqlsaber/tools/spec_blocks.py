"""Turn a ToolDisplaySpec into blocks. Replaces SpecRenderer's dual path."""

from __future__ import annotations

import json
from typing import Any

from sqlsaber.render.blocks import (
    Block,
    Column,
    code,
    error,
    json_block,
    key_values,
    md,
    panel,
    table,
)
from sqlsaber.tools.display import (
    FieldMappings,
    ResultFormat,
    ShowArgs,
    TableConfig,
    ToolDisplaySpec,
)


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def blocks_from_spec_executing(
    tool_name: str, args: dict[str, Any], spec: ToolDisplaySpec
) -> tuple[Block, ...]:
    """Blocks for a tool-executing event.

    Args:
        tool_name: Tool name used in templates.
        args: Tool arguments.
        spec: Display specification.

    Returns:
        Heading plus optional argument key-values.
    """
    config = spec.executing
    message = _format_template(config.message, {"tool_name": tool_name, **args})
    icon = f"{config.icon} " if config.icon else ""
    line = f"{icon}{message}".strip()
    blocks: list[Block] = [md(f"**{line}**")]
    shown = _resolve_args_to_show(config.show_args, args)
    if shown:
        blocks.append(
            key_values({str(key): _cell(value) for key, value in shown.items()})
        )
    return tuple(blocks)


def blocks_from_spec_result(
    tool_name: str, result: object, spec: ToolDisplaySpec
) -> tuple[Block, ...]:
    """Blocks for a tool result.

    Args:
        tool_name: Unused; kept for call-site symmetry.
        result: Tool return value (JSON string, dict, or other).
        spec: Display specification.

    Returns:
        One or more blocks describing the result.
    """
    del tool_name
    data, raw = _parse_result(result)
    config = spec.result
    error_text = _extract_error(data, config.fields)
    if error_text:
        return (error(error_text),)

    fmt = _resolve_format(config.format, data, config.fields)
    title = _format_title(config.title, data)
    output = _extract_output(data, config.fields)
    blocks = _format_output(
        fmt, output, data, raw, title, config.table, config.code_language
    )
    success_value = _extract_success(data, config.fields)
    if success_value is False and not error_text:
        return (*blocks, error("Operation reported failure."))
    return blocks


def _format_output(
    fmt: ResultFormat,
    output: object,
    data: object,
    raw: str | None,
    title: str | None,
    table_config: TableConfig | None,
    code_language: str | None,
) -> tuple[Block, ...]:
    if fmt == "json":
        if raw is not None:
            return (code(raw, "json"),)
        return (json_block(data),)
    if fmt == "code":
        text = "" if output is None else str(output)
        return (code(text, code_language or ""),)
    if fmt == "panel":
        body = "" if output is None else str(output)
        return (panel([md(body)], title=title),)
    if fmt == "table":
        rows = _normalize_rows(output)
        if not rows:
            return (md("*No results*"),)
        columns = _resolve_columns(rows, table_config)
        max_rows = table_config.max_rows if table_config else 20
        return (
            table(
                rows,
                columns=columns,
                caption=title,
                max_rows=max_rows,
            ),
        )
    if fmt == "key_value":
        if isinstance(output, dict):
            return (key_values({str(k): _cell(v) for k, v in output.items()}),)
        return (md(str(output)),)
    return (json_block(data),)


def _format_template(template: str, values: dict[str, Any]) -> str:
    try:
        return template.format_map(_SafeFormatDict(values))
    except Exception:
        return template


def _format_title(title: str | None, data: object) -> str | None:
    if not title:
        return None
    values = _coerce_mapping(data) or {}
    return _format_template(title, {"result": data, **values})


def _resolve_args_to_show(
    show_args: list[str] | ShowArgs, tool_args: dict[str, Any]
) -> dict[str, Any]:
    if show_args == "none":
        return {}
    if show_args == "all":
        return tool_args
    return {key: tool_args[key] for key in show_args if key in tool_args}


def _parse_result(result: object) -> tuple[object, str | None]:
    if isinstance(result, dict | list):
        return result, None
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result, result
        return parsed, result
    return {"output": str(result)}, None


def _extract_output(data: object, fields: FieldMappings) -> object:
    mapping = _coerce_mapping(data)
    if mapping is not None:
        if fields.items and fields.items in mapping:
            return mapping[fields.items]
        if fields.output in mapping:
            return mapping[fields.output]
        return mapping
    return data


def _extract_error(data: object, fields: FieldMappings) -> str | None:
    mapping = _coerce_mapping(data)
    if mapping is None:
        return None
    if fields.error in mapping and mapping[fields.error]:
        return str(mapping[fields.error])
    if "error" in mapping and mapping["error"]:
        return str(mapping["error"])
    if "stderr" in mapping and mapping["stderr"]:
        return str(mapping["stderr"])
    return None


def _extract_success(data: object, fields: FieldMappings) -> bool | None:
    mapping = _coerce_mapping(data)
    if mapping is None:
        return None
    if fields.success and fields.success in mapping:
        value = mapping[fields.success]
        if isinstance(value, bool):
            return value
    return None


def _resolve_format(
    fmt: ResultFormat, data: object, fields: FieldMappings
) -> ResultFormat:
    if fmt != "auto":
        return fmt
    output = _extract_output(data, fields)
    if isinstance(output, list):
        return "table"
    if isinstance(output, dict):
        return "key_value"
    if isinstance(output, str):
        return "panel"
    if isinstance(data, list):
        return "table"
    if isinstance(data, dict):
        return "key_value"
    return "json"


def _normalize_rows(output: object) -> list[dict[str, Any]]:
    if output is None:
        return []
    if isinstance(output, list):
        rows: list[dict[str, Any]] = []
        for item in output:
            mapping = _coerce_mapping(item)
            if mapping is not None:
                rows.append(mapping)
            else:
                rows.append({"value": item})
        return rows
    mapping = _coerce_mapping(output)
    if mapping is not None:
        return [mapping]
    return [{"value": output}]


def _coerce_mapping(data: object) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    return {str(key): value for key, value in data.items()}


def _resolve_columns(
    rows: list[dict[str, Any]], config: TableConfig | None
) -> list[Column]:
    if config and config.columns:
        return [Column(field=col.field, header=col.header) for col in config.columns]
    if not rows:
        return [Column(field="value", header="Value")]
    return [Column(field=key, header=key) for key in rows[0]]


def _cell(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
