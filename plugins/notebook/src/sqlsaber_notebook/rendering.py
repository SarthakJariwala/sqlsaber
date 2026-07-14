"""Bounded model-facing notebook rendering.

Adapted from Future-House/finch ``fhda.utils`` (Apache-2.0), modified for
provider-neutral nbconvert output, deterministic MIME handling, and hard budgets.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Callable
from typing import Any

from PIL import Image, UnidentifiedImageError

from ._shared import (
    MAX_HISTORY_IMAGE_BYTES,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_HEIGHT,
    MAX_IMAGE_WIDTH,
    MAX_OUTPUT_CHARS,
    MAX_SNAPSHOT_CHARS,
    MAX_SNAPSHOT_IMAGE_BYTES,
    MAX_SNAPSHOT_IMAGES,
)
from .session import NotebookSession

_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
type ImageSelector = Callable[[bytes], tuple[bytes | None, str]]


def limit_output(text: str | list[str], limit: int = MAX_OUTPUT_CHARS) -> str:
    """Strip controls and retain both ends of long output or tracebacks."""

    if isinstance(text, list):
        text = "".join(text)
    if limit <= 0:
        return ""
    clean = _strip_controls(text)
    if len(clean) <= limit:
        return clean
    marker = "\n<...output limited...>\n"
    if limit <= len(marker):
        return marker[:limit]
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    return f"{clean[:head]}{marker}{clean[-(remaining - head) :]}"


def process_cell_output(
    output: dict[str, Any],
    *,
    images: list[bytes],
    image_selector: ImageSelector,
) -> str | None:
    """Render one Jupyter output with deterministic PNG-over-text priority."""

    output_type = output.get("output_type")
    if output_type == "stream":
        return limit_output(_as_text(output.get("text", "")))
    if output_type == "error":
        traceback = output.get("traceback", [])
        if traceback:
            return limit_output(
                "\n".join(traceback) if isinstance(traceback, list) else str(traceback)
            )
        return limit_output(
            f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        )
    if output_type not in {"execute_result", "display_data"}:
        return None

    data = output.get("data", {})
    if not isinstance(data, dict):
        return None
    if "image/png" in data:
        normalized, error = _decode_and_bound_png(_as_text(data["image/png"]))
        if normalized is None:
            return f"[image omitted: {error}]"
        selected, reason = image_selector(normalized)
        if selected is None:
            return f"[image omitted: {reason}]"
        images.append(selected)
        return f"<{len(images)}>"
    if "text/plain" in data:
        return limit_output(_as_text(data["text/plain"]))
    # Deliberately ignore HTML/Markdown/LaTeX blobs rather than feeding large tables.
    return None


def view_notebook(
    cells: list[str],
    outputs: list[list[dict[str, Any]]],
    *,
    language: str = "python",
    image_selector: ImageSelector | None = None,
) -> tuple[str, list[bytes]]:
    """Render numbered code cells and bounded outputs as Markdown."""

    if image_selector is None:
        selected_count = 0
        selected_bytes = 0

        def selector(image: bytes) -> tuple[bytes | None, str]:
            nonlocal selected_count, selected_bytes
            if selected_count >= MAX_SNAPSHOT_IMAGES:
                return None, "snapshot image count budget"
            if selected_bytes + len(image) > MAX_SNAPSHOT_IMAGE_BYTES:
                return None, "snapshot byte budget"
            selected_count += 1
            selected_bytes += len(image)
            return image, ""

    else:
        selector = image_selector
    markdown: list[str] = []
    images: list[bytes] = []
    for index, source in enumerate(cells):
        markdown.extend((f"### Cell {index}:", f"```{language}", source, "```"))
        cell_outputs = outputs[index] if index < len(outputs) else []
        rendered = [
            text
            for output in cell_outputs
            if (
                text := process_cell_output(
                    output, images=images, image_selector=selector
                )
            )
        ]
        if rendered:
            markdown.extend((f"### Output {index}:", "```", "\n".join(rendered), "```"))
    return limit_output("\n".join(markdown), MAX_SNAPSHOT_CHARS), images


def render_snapshot_for_model(session: NotebookSession) -> tuple[str, list[bytes]]:
    """Render one snapshot while enforcing compatibility, dedupe, and binary budgets."""

    snapshot_bytes = 0
    snapshot_count = 0

    def select(image: bytes) -> tuple[bytes | None, str]:
        nonlocal snapshot_bytes, snapshot_count
        if not session.include_snapshot_images:
            return None, "unsupported"
        digest = hashlib.sha256(image).hexdigest()
        if digest in session.sent_image_hashes:
            return None, "already shown"
        if snapshot_count >= MAX_SNAPSHOT_IMAGES:
            return None, "snapshot image count budget"
        if snapshot_bytes + len(image) > MAX_SNAPSHOT_IMAGE_BYTES:
            return None, "snapshot byte budget"
        if session.sent_image_bytes + len(image) > MAX_HISTORY_IMAGE_BYTES:
            return None, "history byte budget"
        snapshot_count += 1
        snapshot_bytes += len(image)
        session.sent_image_hashes.add(digest)
        session.sent_image_bytes += len(image)
        return image, ""

    return view_notebook(session.cells, session.outputs, image_selector=select)


def extract_notebook_images(
    outputs: list[list[dict[str, Any]]],
) -> list[bytes]:
    """Harvest bounded, deduplicated PNG outputs for the standalone result."""

    images: list[bytes] = []
    hashes: set[str] = set()
    total = 0
    for cell_outputs in outputs:
        for output in cell_outputs:
            data = output.get("data", {})
            if not isinstance(data, dict) or "image/png" not in data:
                continue
            image, _ = _decode_and_bound_png(_as_text(data["image/png"]))
            if image is None:
                continue
            digest = hashlib.sha256(image).hexdigest()
            if digest in hashes or len(images) >= MAX_SNAPSHOT_IMAGES:
                continue
            if total + len(image) > MAX_SNAPSHOT_IMAGE_BYTES:
                return images
            hashes.add(digest)
            total += len(image)
            images.append(image)
    return images


def normalize_png_bytes(raw: bytes) -> bytes | None:
    """Validate/downsample raw PNG bytes to the model-facing image budget."""

    encoded = base64.b64encode(raw).decode()
    normalized, _ = _decode_and_bound_png(encoded)
    return normalized


def _decode_and_bound_png(encoded: str) -> tuple[bytes | None, str]:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG":
                return None, "invalid PNG"
            if source.width * source.height > 4 * MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT:
                return None, "image dimensions budget"
            source.load()
            if source.width > MAX_IMAGE_WIDTH or source.height > MAX_IMAGE_HEIGHT:
                resized = source.copy()
                resized.thumbnail(
                    (MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT), Image.Resampling.LANCZOS
                )
                buffer = io.BytesIO()
                resized.save(buffer, format="PNG", optimize=True)
                raw = buffer.getvalue()
    except (ValueError, Image.DecompressionBombError, UnidentifiedImageError, OSError):
        return None, "invalid PNG"
    if len(raw) > MAX_IMAGE_BYTES:
        return None, "image byte budget"
    return raw, ""


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def _strip_controls(text: str) -> str:
    without_ansi = _ANSI_ESCAPE.sub("", text)
    return "".join(
        character
        for character in without_ansi
        if character in "\n\r\t" or ord(character) >= 32
    )
