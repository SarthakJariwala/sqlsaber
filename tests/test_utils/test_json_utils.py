"""Tests for JSON utilities with enhanced serialization."""

import datetime as dt
import json
import uuid
from decimal import Decimal

import pytest

from sqlsaber.utils.json_utils import EnhancedJSONEncoder, json_dumps


class TestEnhancedJSONEncoder:
    """Tests for EnhancedJSONEncoder."""

    def test_decimal_serialization(self) -> None:
        """Decimal values should be converted to float."""
        data = {"price": Decimal("19.99"), "quantity": Decimal("100")}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["price"] == 19.99
        assert parsed["quantity"] == 100.0

    def test_decimal_precision(self) -> None:
        """Decimal with many decimal places should convert to float."""
        data = {"value": Decimal("3.141592653589793")}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert abs(parsed["value"] - 3.141592653589793) < 1e-10

    def test_datetime_serialization(self) -> None:
        """datetime values should be converted to ISO 8601 format."""
        data = {"created_at": dt.datetime(2024, 1, 15, 10, 30, 45)}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["created_at"] == "2024-01-15T10:30:45"

    def test_datetime_with_microseconds(self) -> None:
        """datetime with microseconds should include them in ISO format."""
        data = {"timestamp": dt.datetime(2024, 1, 15, 10, 30, 45, 123456)}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["timestamp"] == "2024-01-15T10:30:45.123456"

    def test_date_serialization(self) -> None:
        """date values should be converted to ISO 8601 format."""
        data = {"birth_date": dt.date(1990, 5, 20)}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["birth_date"] == "1990-05-20"

    def test_time_serialization(self) -> None:
        """time values should be converted to ISO 8601 format."""
        data = {"start_time": dt.time(14, 30, 0)}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["start_time"] == "14:30:00"

    def test_uuid_serialization(self) -> None:
        """UUID values should be converted to string."""
        test_uuid = uuid.UUID("12345678-1234-5678-1234-567812345678")
        data = {"id": test_uuid}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["id"] == "12345678-1234-5678-1234-567812345678"

    def test_bytes_serialization(self) -> None:
        """bytes values should be converted to base64 string."""
        data = {"binary_data": b"hello world"}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["binary_data"] == "aGVsbG8gd29ybGQ="

    def test_bytearray_serialization(self) -> None:
        """bytearray values should be converted to base64 string."""
        data = {"binary_data": bytearray(b"test data")}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["binary_data"] == "dGVzdCBkYXRh"

    def test_memoryview_serialization(self) -> None:
        """memoryview values should be converted to base64 string."""
        data = {"binary_data": memoryview(b"memory")}
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["binary_data"] == "bWVtb3J5"

    def test_nested_special_types(self) -> None:
        """Nested structures with special types should serialize correctly."""
        data = {
            "order": {
                "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
                "total": Decimal("99.99"),
                "created_at": dt.datetime(2024, 1, 15, 10, 30, 45),
                "items": [
                    {"price": Decimal("49.99"), "quantity": 2},
                    {"price": Decimal("0.01"), "quantity": 1},
                ],
            }
        }
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed["order"]["id"] == "12345678-1234-5678-1234-567812345678"
        assert parsed["order"]["total"] == 99.99
        assert parsed["order"]["created_at"] == "2024-01-15T10:30:45"
        assert parsed["order"]["items"][0]["price"] == 49.99

    def test_standard_types_still_work(self) -> None:
        """Standard JSON types should still serialize correctly."""
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        result = json.dumps(data, cls=EnhancedJSONEncoder)
        parsed = json.loads(result)

        assert parsed == data

    def test_unsupported_type_raises_error(self) -> None:
        """Unsupported types should raise TypeError."""

        class CustomObject:
            pass

        data = {"custom": CustomObject()}
        with pytest.raises(TypeError, match="not JSON serializable"):
            json.dumps(data, cls=EnhancedJSONEncoder)


class TestJsonDumps:
    """Tests for json_dumps helper function."""

    def test_uses_enhanced_encoder_by_default(self) -> None:
        """json_dumps should use EnhancedJSONEncoder by default."""
        data = {"price": Decimal("19.99")}
        result = json_dumps(data)
        parsed = json.loads(result)

        assert parsed["price"] == 19.99

    def test_all_special_types(self) -> None:
        """json_dumps should handle all special types."""
        data = {
            "decimal": Decimal("123.45"),
            "datetime": dt.datetime(2024, 1, 15, 10, 30, 45),
            "date": dt.date(2024, 1, 15),
            "time": dt.time(10, 30, 45),
            "uuid": uuid.UUID("12345678-1234-5678-1234-567812345678"),
            "bytes": b"binary",
        }
        result = json_dumps(data)
        parsed = json.loads(result)

        assert parsed["decimal"] == 123.45
        assert parsed["datetime"] == "2024-01-15T10:30:45"
        assert parsed["date"] == "2024-01-15"
        assert parsed["time"] == "10:30:45"
        assert parsed["uuid"] == "12345678-1234-5678-1234-567812345678"
        assert parsed["bytes"] == "YmluYXJ5"

    def test_kwargs_passthrough(self) -> None:
        """json_dumps should pass kwargs to json.dumps."""
        data = {"key": "value"}
        result = json_dumps(data, indent=2)

        assert "  " in result

    def test_custom_encoder_override(self) -> None:
        """Custom encoder can still be passed to override default."""

        class CustomEncoder(json.JSONEncoder):
            def default(self, obj: object) -> str:
                if isinstance(obj, Decimal):
                    return f"DECIMAL:{obj}"
                return super().default(obj)

        data = {"price": Decimal("19.99")}
        result = json_dumps(data, cls=CustomEncoder)
        parsed = json.loads(result)

        assert parsed["price"] == "DECIMAL:19.99"

    def test_simulates_database_result(self) -> None:
        """Simulate a typical database query result with mixed types."""
        db_result = [
            {
                "id": uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890"),
                "name": "Product A",
                "price": Decimal("29.99"),
                "created_at": dt.datetime(2024, 1, 15, 10, 30, 45),
                "updated_at": dt.date(2024, 1, 20),
                "metadata": b'{"key": "value"}',
            },
            {
                "id": uuid.UUID("b2c3d4e5-f6a7-8901-bcde-f23456789012"),
                "name": "Product B",
                "price": Decimal("0.01"),
                "created_at": dt.datetime(2024, 1, 16, 11, 45, 30),
                "updated_at": dt.date(2024, 1, 21),
                "metadata": None,
            },
        ]

        result = json_dumps({"success": True, "results": db_result})
        parsed = json.loads(result)

        assert parsed["success"] is True
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["price"] == 29.99
        assert parsed["results"][1]["price"] == 0.01
