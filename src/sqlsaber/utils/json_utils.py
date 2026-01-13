"""JSON utilities with enhanced serialization for database types."""

from __future__ import annotations

import base64
import datetime as dt
import json
import uuid
from decimal import Decimal
from typing import Any


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles common database types.

    Supports:
    - Decimal -> float
    - datetime/date/time -> ISO 8601 string
    - UUID -> string
    - bytes/bytearray/memoryview -> base64 string
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)

        if isinstance(o, dt.datetime):
            return o.isoformat()

        if isinstance(o, dt.date):
            return o.isoformat()

        if isinstance(o, dt.time):
            return o.isoformat()

        if isinstance(o, uuid.UUID):
            return str(o)

        if isinstance(o, (bytes, bytearray, memoryview)):
            return base64.b64encode(bytes(o)).decode("ascii")

        return super().default(o)


def json_dumps(obj: Any, **kwargs: Any) -> str:
    """Serialize object to JSON string with enhanced type support.

    Drop-in replacement for json.dumps that uses EnhancedJSONEncoder by default.
    Handles Decimal, datetime, UUID, bytes, and other common database types.

    Args:
        obj: Object to serialize
        **kwargs: Additional arguments passed to json.dumps

    Returns:
        JSON string representation
    """
    kwargs.setdefault("cls", EnhancedJSONEncoder)
    return json.dumps(obj, **kwargs)
