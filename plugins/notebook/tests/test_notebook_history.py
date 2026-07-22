from __future__ import annotations

from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from sqlsaber_notebook.history import SNAPSHOT_MARKER, collapse_old_snapshots


def _snapshot_request(index: int) -> ModelRequest:
    return ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="edit_cell",
                content=f"Edited cell {index}",
                tool_call_id=f"call-{index}",
            ),
            UserPromptPart(
                content=[
                    f"{SNAPSHOT_MARKER} cell {index}",
                    BinaryContent(data=f"png-{index}".encode(), media_type="image/png"),
                ]
            ),
        ]
    )


def _tool_call(index: int) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name="edit_cell",
                args={"contents": f"x = {index}"},
                tool_call_id=f"call-{index}",
            )
        ]
    )


async def test_history_collapse_preserves_tool_returns_and_latest_snapshot() -> None:
    messages = []
    for index in range(4):
        messages.extend((_tool_call(index), _snapshot_request(index)))
    messages.append(ModelResponse(parts=[TextPart(content="still analyzing")]))

    collapsed = await collapse_old_snapshots(messages, cache=False)

    full_snapshots = 0
    for message in collapsed:
        if not isinstance(message, ModelRequest):
            continue
        returns = [part for part in message.parts if isinstance(part, ToolReturnPart)]
        assert len(returns) == 1
        prompts = [part for part in message.parts if isinstance(part, UserPromptPart)]
        assert len(prompts) == 1
        content = prompts[0].content
        if isinstance(content, str):
            full_snapshots += SNAPSHOT_MARKER in content
        elif any(isinstance(item, str) and SNAPSHOT_MARKER in item for item in content):
            full_snapshots += 1
    assert full_snapshots == 1
    # ProcessHistory rewrites outbound copies; stored input messages remain full.
    original = messages[1]
    assert isinstance(original, ModelRequest)
    prompt = next(part for part in original.parts if isinstance(part, UserPromptPart))
    assert not isinstance(prompt.content, str)
    assert any(isinstance(item, BinaryContent) for item in prompt.content)


async def test_anthropic_stubs_use_at_most_two_explicit_cache_points() -> None:
    messages: list[ModelMessage] = [_snapshot_request(index) for index in range(6)]
    collapsed = await collapse_old_snapshots(messages, cache=True)

    cache_points = [
        item
        for message in collapsed
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, CachePoint)
    ]
    assert len(cache_points) == 2
    latest = collapsed[-1]
    assert isinstance(latest, ModelRequest)
    latest_prompt = next(
        part for part in latest.parts if isinstance(part, UserPromptPart)
    )
    assert not isinstance(latest_prompt.content, str)
    assert any(
        isinstance(item, str) and SNAPSHOT_MARKER in item
        for item in latest_prompt.content
    )


async def test_non_anthropic_stubs_do_not_include_cache_points() -> None:
    collapsed = await collapse_old_snapshots(
        [_snapshot_request(0), _snapshot_request(1)], cache=False
    )
    first = collapsed[0]
    assert isinstance(first, ModelRequest)
    prompt = next(part for part in first.parts if isinstance(part, UserPromptPart))
    assert not isinstance(prompt.content, str)
    assert prompt.content == ["[old notebook state omitted]"]
