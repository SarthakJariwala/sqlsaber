"""Data loader tests."""

from sqlsaber_viz.data_loader import extract_data_summary, infer_column_type


def test_extract_data_summary_types() -> None:
    payload = {
        "row_count": 2,
        "results": [
            {"name": "A", "value": 1, "when": "2024-01-02T03:04:05"},
            {"name": "B", "value": 2, "when": "2024-01-03T04:05:06"},
        ],
    }

    summary = extract_data_summary(payload)
    columns = {col["name"]: col for col in summary["columns"]}

    assert summary["row_count"] == 2
    assert columns["name"]["type"] == "string"
    assert columns["value"]["type"] == "number"
    assert columns["when"]["type"] == "time"


def test_infer_column_type_nulls() -> None:
    assert infer_column_type([None, None]) == "null"
