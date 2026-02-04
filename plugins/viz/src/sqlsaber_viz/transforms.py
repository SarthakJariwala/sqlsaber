"""Transform pipeline for visualization data."""

from __future__ import annotations

import re
from datetime import datetime

from .spec import (
    FilterConfig,
    FilterTransform,
    LimitTransform,
    SortItem,
    SortTransform,
    Transform,
)


def apply_transforms(rows: list[dict], transforms: list[Transform]) -> list[dict]:
    """Apply transform pipeline to rows."""
    result = list(rows)
    for transform in transforms:
        if isinstance(transform, SortTransform):
            result = apply_sort(result, transform.sort)
        elif isinstance(transform, LimitTransform):
            result = result[: transform.limit]
        elif isinstance(transform, FilterTransform):
            result = apply_filter(result, transform.filter)
    return result


def apply_sort(rows: list[dict], sorts: list[SortItem]) -> list[dict]:
    """Sort rows by multiple fields."""
    result = list(rows)
    for sort in reversed(sorts):
        field = sort.field
        descending = sort.dir == "desc"
        result = sorted(result, key=lambda row: _sort_key(row.get(field)))
        if descending:
            result.reverse()
        # Always push None/missing values to the end.
        non_none = [row for row in result if row.get(field) is not None]
        none_rows = [row for row in result if row.get(field) is None]
        result = non_none + none_rows
    return result


def apply_filter(rows: list[dict], filter_config: FilterConfig) -> list[dict]:
    """Filter rows by condition."""
    field = filter_config.field
    op = filter_config.op
    target = filter_config.value

    filtered: list[dict] = []
    for row in rows:
        value = row.get(field)
        if _compare(value, op, target):
            filtered.append(row)
    return filtered


def _sort_key(value: object) -> tuple[int, object]:
    if value is None:
        return (3, "")

    numeric = _coerce_number(value)
    if numeric is not None:
        return (0, numeric)

    time_val = _coerce_time(value)
    if time_val is not None:
        return (1, time_val)

    return (2, str(value).lower())


def _compare(value: object, op: str, target: object) -> bool:
    if op in ("==", "!="):
        result = _equals(value, target)
        return result if op == "==" else not result

    left_num = _coerce_number(value)
    right_num = _coerce_number(target)
    if left_num is not None and right_num is not None:
        return _compare_ordered(left_num, right_num, op)

    left_time = _coerce_time(value)
    right_time = _coerce_time(target)
    if left_time is not None and right_time is not None:
        return _compare_ordered(left_time, right_time, op)

    return False


def _equals(value: object, target: object) -> bool:
    if value is None or target is None:
        return value is target

    left_num = _coerce_number(value)
    right_num = _coerce_number(target)
    if left_num is not None and right_num is not None:
        return left_num == right_num

    left_time = _coerce_time(value)
    right_time = _coerce_time(target)
    if left_time is not None and right_time is not None:
        return left_time == right_time

    return value == target


def _compare_ordered(left: object, right: object, op: str) -> bool:
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    return False


def _coerce_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # Handle Z suffix (e.g., "2024-01-01T00:00:00Z")
        normalized = value
        if value.endswith("Z"):
            normalized = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            pass
        # Try YYYY-MM format (e.g., "2023-06")
        if re.match(r"^\d{4}-\d{2}$", value):
            try:
                return datetime.fromisoformat(f"{value}-01")
            except ValueError:
                pass
        return None
    return None
