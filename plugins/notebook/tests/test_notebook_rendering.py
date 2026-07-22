from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image

from sqlsaber_notebook._shared import MAX_SNAPSHOT_CHARS
from sqlsaber_notebook.execution.fake import FakeNotebookBackend
from sqlsaber_notebook.rendering import (
    limit_output,
    render_snapshot_for_model,
    view_notebook,
)
from sqlsaber_notebook.result import Workspace
from sqlsaber_notebook.session import NotebookSession


def _png(width: int = 2, height: int = 2) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


def _session(*, include_images: bool) -> NotebookSession:
    return NotebookSession(
        workspace=Workspace(()),
        backend=FakeNotebookBackend(),
        image="unused",
        include_snapshot_images=include_images,
        cells=["print('hello')"],
        outputs=[[]],
    )


def test_limit_output_keeps_head_tail_and_strips_controls() -> None:
    rendered = limit_output("\x1b[31mHEAD\x1b[0m" + "x" * 100 + "TAIL\x00", 40)
    assert rendered.startswith("HEAD")
    assert rendered.endswith("TAIL")
    assert "output limited" in rendered
    assert "\x1b" not in rendered
    assert "\x00" not in rendered


def test_png_has_priority_over_plain_text_and_placeholder_is_correlated() -> None:
    image = _png()
    markdown, images = view_notebook(
        ["plot()"],
        [
            [
                {
                    "output_type": "display_data",
                    "data": {
                        "text/plain": "do not render this fallback",
                        "image/png": base64.b64encode(image).decode(),
                    },
                }
            ]
        ],
    )
    assert "<1>" in markdown
    assert "do not render" not in markdown
    assert images == [image]


def test_unsupported_and_duplicate_images_have_visible_omissions() -> None:
    image = _png()
    encoded = base64.b64encode(image).decode()
    session = _session(include_images=False)
    session.outputs = [
        [{"output_type": "display_data", "data": {"image/png": encoded}}]
    ]
    markdown, images = render_snapshot_for_model(session)
    assert images == []
    assert "[image omitted: unsupported]" in markdown
    assert "<1>" not in markdown

    session.include_snapshot_images = True
    first_markdown, first_images = render_snapshot_for_model(session)
    second_markdown, second_images = render_snapshot_for_model(session)
    assert "<1>" in first_markdown
    assert first_images == [image]
    assert second_images == []
    assert "[image omitted: already shown]" in second_markdown
    assert "<1>" not in second_markdown


def test_balanced_snapshot_keeps_middle_cells_and_prioritizes_errors() -> None:
    cells = [f"start_{index}\n" + "s" * 5_000 + f"\nend_{index}" for index in range(50)]
    outputs: list[list[dict[str, Any]]] = [
        [{"output_type": "stream", "text": "o" * 10_000}] for _ in cells
    ]
    outputs[25] = [
        {
            "output_type": "error",
            "ename": "RuntimeError",
            "evalue": "critical",
            "traceback": ["CRITICAL_MIDDLE_ERROR", "traceback tail"],
        }
    ]

    markdown, _ = view_notebook(cells, outputs)

    assert len(markdown) <= MAX_SNAPSHOT_CHARS
    assert "### Cell 0:" in markdown
    assert "### Cell 25:" in markdown
    assert "### Cell 49:" in markdown
    assert "CRITICAL_MIDDLE_ERROR" in markdown
    assert "cell source limited" in markdown
    assert "cell outputs limited" in markdown


def test_invalid_png_is_omitted_without_dangling_placeholder() -> None:
    markdown, images = view_notebook(
        ["plot()"],
        [[{"output_type": "display_data", "data": {"image/png": "not-base64"}}]],
    )
    assert images == []
    assert "[image omitted: invalid PNG]" in markdown
    assert "<1>" not in markdown
