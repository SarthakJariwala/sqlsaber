from __future__ import annotations

from datetime import datetime, timezone

from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from sqlsaber.query_results import QUERY_RESULT_MEDIA_TYPE, StoredQueryResult
from sqlsaber.rpc.protocol import (
    AssistantMessage,
    MessageEnd,
    MessageStart,
    MessageUpdate,
    SqlUpdate,
    TextBlock,
    TextDelta,
    TextEnd,
    ThinkingBlock,
    ThinkingDelta,
    ThinkingStart,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResultMessage,
    UserMessage,
)
from sqlsaber.rpc.transcript import StreamTranslator, last_assistant_text, transcript
from sqlsaber.utils.partial_json import partial_json_query

_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_partial_json_query_recovers_trailing_string() -> None:
    assert partial_json_query('{"query": "SELECT id FROM') == "SELECT id FROM"
    assert partial_json_query("not json") is None


def test_transcript_projects_user_assistant_and_tool_result() -> None:
    descriptor = StoredQueryResult(
        id="qr_" + "a" * 32,
        file="result_call_1.json",
        media_type=QUERY_RESULT_MEDIA_TYPE,
        size=10,
        sha256="b" * 64,
        row_count=1,
        columns=("n",),
    )
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart("hidden"),
                UserPromptPart("count them", timestamp=_TS),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart("plan"),
                ToolCallPart("execute_sql", {"query": "SELECT 1"}, "call_1"),
            ],
            model_name="claude-3-5-sonnet",
            timestamp=_TS,
            finish_reason="tool_call",
            usage=RequestUsage(input_tokens=3, output_tokens=2),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "execute_sql",
                    {"n": 1},
                    "call_1",
                    timestamp=_TS,
                    metadata={"query_result": descriptor.to_dict()},
                )
            ]
        ),
        ModelResponse(
            parts=[TextPart("one row")],
            model_name="claude-3-5-sonnet",
            timestamp=_TS,
            finish_reason="stop",
        ),
    ]
    projected = transcript(messages)
    assert isinstance(projected[0], UserMessage)
    assert projected[0].content == "count them"
    assert isinstance(projected[1], AssistantMessage)
    assert projected[1].stop_reason == "toolUse"
    assert projected[1].content[0] == ThinkingBlock("plan")
    assert isinstance(projected[2], ToolResultMessage)
    assert projected[2].query_result is not None
    assert projected[2].query_result.id == descriptor.id
    assert last_assistant_text(messages) == "one row"


def test_transcript_retry_prompt_is_an_error_tool_result() -> None:
    messages = [
        ModelRequest(
            parts=[
                RetryPromptPart("bad args", tool_name="execute_sql", tool_call_id="c1")
            ]
        )
    ]
    projected = transcript(messages)
    assert isinstance(projected[0], ToolResultMessage)
    assert projected[0].is_error is True
    assert "Fix the errors" in projected[0].content


def test_stream_translator_text_and_thinking_and_tool_sql() -> None:
    translator = StreamTranslator(model_name="claude-3-5-sonnet")
    events = []
    events.extend(
        translator.translate(PartStartEvent(index=0, part=ThinkingPart(content="Need")))
    )
    events.extend(
        translator.translate(
            PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=" joins"))
        )
    )
    events.extend(
        translator.translate(
            PartEndEvent(index=0, part=ThinkingPart(content="Need joins"))
        )
    )
    events.extend(
        translator.translate(
            PartStartEvent(
                index=1,
                part=ToolCallPart("execute_sql", '{"query": "SEL', "call_1"),
            )
        )
    )
    events.extend(
        translator.translate(
            PartDeltaEvent(
                index=1,
                delta=ToolCallPartDelta(args_delta='ECT 1"}'),
            )
        )
    )
    events.extend(
        translator.translate(
            PartEndEvent(
                index=1,
                part=ToolCallPart("execute_sql", {"query": "SELECT 1"}, "call_1"),
            )
        )
    )
    events.extend(translator.finish())

    assert isinstance(events[0], MessageStart)
    assert isinstance(events[1], MessageUpdate)
    assert isinstance(events[1].event, ThinkingStart)
    thinking_deltas = [
        event.event
        for event in events
        if isinstance(event, MessageUpdate) and isinstance(event.event, ThinkingDelta)
    ]
    assert thinking_deltas[0].delta == "Need"
    sql_updates = [event for event in events if isinstance(event, SqlUpdate)]
    assert sql_updates
    assert sql_updates[-1].sql == "SELECT 1"
    assert isinstance(events[-1], MessageEnd)
    assert events[-1].message.usage is None
    assert events[-1].message.stop_reason is None


def test_stream_translator_function_tool_events() -> None:
    translator = StreamTranslator(model_name=None)
    start = translator.translate(
        FunctionToolCallEvent(
            part=ToolCallPart("execute_sql", {"query": "SELECT 1"}, "c1")
        )
    )
    end = translator.translate(
        FunctionToolResultEvent(
            part=ToolReturnPart("execute_sql", {"ok": True}, "c1"),
        )
    )
    assert start == [
        ToolExecutionStart(
            tool_call_id="c1",
            tool_name="execute_sql",
            args={"query": "SELECT 1"},
        )
    ]
    assert isinstance(end[0], ToolExecutionEnd)
    assert end[0].is_error is False
    assert translator.finish() == []


def test_stream_translator_text_deltas() -> None:
    translator = StreamTranslator(model_name="m")
    events = []
    events.extend(translator.translate(PartStartEvent(index=0, part=TextPart("He"))))
    events.extend(
        translator.translate(
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="llo"))
        )
    )
    events.extend(translator.translate(PartEndEvent(index=0, part=TextPart("Hello"))))
    events.extend(translator.finish())
    kinds = []
    for event in events:
        if isinstance(event, MessageStart):
            kinds.append("start")
        elif isinstance(event, MessageUpdate):
            kinds.append(type(event.event).__name__)
        elif isinstance(event, MessageEnd):
            kinds.append("end")
    assert kinds == [
        "start",
        "TextStart",
        "TextDelta",
        "TextDelta",
        "TextEnd",
        "end",
    ]
    text_deltas = [
        event.event.delta
        for event in events
        if isinstance(event, MessageUpdate) and isinstance(event.event, TextDelta)
    ]
    assert text_deltas == ["He", "llo"]
    end_event = [
        event.event
        for event in events
        if isinstance(event, MessageUpdate) and isinstance(event.event, TextEnd)
    ][0]
    assert end_event.content == "Hello"
    assert isinstance(events[-1].message.content[0], TextBlock)
    assert events[-1].message.content[0].text == "Hello"
