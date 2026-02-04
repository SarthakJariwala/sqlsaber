"""Transform pipeline tests."""

from sqlsaber_viz.spec import FilterConfig, FilterTransform, LimitTransform, SortItem, SortTransform
from sqlsaber_viz.transforms import apply_filter, apply_sort, apply_transforms


def test_apply_sort_desc_with_none() -> None:
    rows = [{"a": 2}, {"a": None}, {"a": 1}]
    sorted_rows = apply_sort(rows, [SortItem(field="a", dir="desc")])
    assert [row["a"] for row in sorted_rows] == [2, 1, None]


def test_apply_filter_numeric_string() -> None:
    rows = [{"a": "2"}, {"a": "foo"}, {"a": 3}]
    filtered = apply_filter(rows, FilterConfig(field="a", op=">", value=2))
    assert [row["a"] for row in filtered] == [3]


def test_apply_transforms_pipeline() -> None:
    rows = [{"a": 3}, {"a": 1}, {"a": 2}]
    transforms = [
        SortTransform(sort=[SortItem(field="a", dir="asc")]),
        LimitTransform(limit=2),
    ]
    result = apply_transforms(rows, transforms)
    assert [row["a"] for row in result] == [1, 2]
