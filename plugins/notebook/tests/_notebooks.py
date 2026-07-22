from __future__ import annotations

import base64
import json
from typing import Any


def contract_notebook() -> bytes:
    sources = [
        """import json
import os
import socket
from pathlib import Path
values = json.loads(Path('../inputs/data.json').read_text())['values']
try:
    Path('../inputs/data.json').write_text('changed')
    print('input_mutable')
except OSError:
    print('input_read_only')
try:
    socket.create_connection(('1.1.1.1', 80), timeout=1)
    print('network_open')
except OSError:
    print('network_blocked')
print(f'uid={os.getuid()} sum={sum(values)}')
""",
        "raise RuntimeError('intentional contract error')",
        """from pathlib import Path
try:
    kernel_counter += 1
except NameError:
    kernel_counter = 1
Path('nested').mkdir(exist_ok=True)
Path('nested/summary.txt').write_text(f'sum={sum(values)} counter={kernel_counter}')
import matplotlib.pyplot as plt
plt.figure(figsize=(2, 1))
plt.plot(values)
plt.tight_layout()
plt.savefig('plot.png')
plt.close()
print(f'continued counter={kernel_counter}')
""",
    ]
    notebook: dict[str, Any] = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"cell-{index}",
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
            for index, source in enumerate(sources, start=1)
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook).encode()


def parse_notebook(data: bytes) -> dict[str, Any]:
    return json.loads(data)


def stream_text(cell: dict[str, Any]) -> str:
    chunks: list[str] = []
    for output in cell["outputs"]:
        if output.get("output_type") != "stream":
            continue
        text = output.get("text", "")
        chunks.append(text if isinstance(text, str) else "".join(text))
    return "".join(chunks)


def assert_contract_result(data: bytes, *, expected_uid: int) -> None:
    notebook = parse_notebook(data)
    first = stream_text(notebook["cells"][0])
    assert "input_read_only" in first
    assert "network_blocked" in first
    assert "network_open" not in first
    assert f"uid={expected_uid} sum=6" in first
    assert any(
        output.get("output_type") == "error" and output.get("ename") == "RuntimeError"
        for output in notebook["cells"][1]["outputs"]
    )
    assert "continued counter=1" in stream_text(notebook["cells"][2])
    image_outputs = [
        output
        for output in notebook["cells"][2]["outputs"]
        if "image/png" in output.get("data", {})
    ]
    # savefig creates a file; this also accepts models/backends that do not display it inline.
    for output in image_outputs:
        base64.b64decode(output["data"]["image/png"], validate=True)
