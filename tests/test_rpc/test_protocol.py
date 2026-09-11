from __future__ import annotations

import json

from sqlsaber.rpc.protocol import (
    MAX_STDIN_LINE,
    Aborted,
    AgentEnd,
    AgentStart,
    Completed,
    Err,
    Failed,
    GetQueryResult,
    Invalid,
    MessageUpdate,
    Ok,
    Prompt,
    TextDelta,
    encode,
    parse_command,
)


def _load(record: bytes) -> dict:
    assert record.endswith(b"\n")
    assert b"\n" not in record[:-1]
    return json.loads(record.decode("utf-8"))


def test_parse_strips_crlf_and_echoes_id() -> None:
    command = parse_command(b'{"id":"q1","type":"abort"}\r\n')
    assert command == parse_command(b'{"id":"q1","type":"abort"}')
    encoded = _load(encode(Ok("abort", "q1", {"aborted": False})))
    assert encoded["id"] == "q1"
    assert encoded["success"] is True
    assert encoded["command"] == "abort"


def test_parse_ignores_blank_payload_as_invalid_object() -> None:
    command = parse_command(b"\n")
    assert isinstance(command, Invalid)
    assert command.command == "parse"


def test_parse_keeps_line_separator_inside_json_strings() -> None:
    raw = '{"type":"prompt","message":"a\u2028b"}'
    command = parse_command(raw.encode("utf-8"))
    assert isinstance(command, Prompt)
    assert command.message == "a\u2028b"


def test_parse_rejects_non_utf8() -> None:
    command = parse_command(b"\xff\xfe")
    assert isinstance(command, Invalid)
    assert command.command == "parse"
    assert "UTF-8" in command.error


def test_parse_rejects_invalid_json() -> None:
    command = parse_command(b"{not json}\n")
    assert isinstance(command, Invalid)
    assert command.error.startswith("Failed to parse command:")


def test_parse_rejects_non_object() -> None:
    command = parse_command(b'"prompt"\n')
    assert isinstance(command, Invalid)
    assert "JSON object" in command.error


def test_parse_rejects_bool_id() -> None:
    command = parse_command(b'{"type":"abort","id":true}')
    assert isinstance(command, Invalid)
    assert "id must be" in command.error


def test_parse_integer_id() -> None:
    parsed = parse_command(b'{"type":"get_state","id":7}')
    assert parsed.id == 7


def test_parse_unknown_command() -> None:
    command = parse_command(b'{"type":"compact"}')
    assert isinstance(command, Invalid)
    assert command.command == "compact"
    assert command.error == "Unknown command: compact"


def test_parse_rejects_streaming_behavior() -> None:
    command = parse_command(
        b'{"type":"prompt","message":"hi","streamingBehavior":"steer"}'
    )
    assert isinstance(command, Invalid)
    assert command.command == "prompt"
    assert "streamingBehavior" in command.error


def test_parse_rejects_images_and_parent_session() -> None:
    images = parse_command(b'{"type":"prompt","message":"hi","images":[]}')
    parent = parse_command(b'{"type":"prompt","message":"hi","parentSession":"x"}')
    assert isinstance(images, Invalid)
    assert "images" in images.error
    assert isinstance(parent, Invalid)
    assert "parentSession" in parent.error


def test_parse_rejects_empty_prompt() -> None:
    command = parse_command(b'{"type":"prompt","message":"   "}')
    assert isinstance(command, Invalid)
    assert command.error == "message must be a non-empty string"


def test_parse_ignores_unknown_fields() -> None:
    command = parse_command(b'{"type":"abort","extra":1}')
    assert command.__class__.__name__ == "Abort"


def test_parse_thinking_level() -> None:
    ok = parse_command(b'{"type":"set_thinking_level","level":"off"}')
    bad = parse_command(b'{"type":"set_thinking_level","level":"xhigh"}')
    assert ok.__class__.__name__ == "SetThinkingLevel"
    assert isinstance(bad, Invalid)
    assert "off, minimal" in bad.error


def test_parse_get_query_result_paging() -> None:
    command = parse_command(
        b'{"type":"get_query_result","resultId":"qr_abc","offset":10,"limit":2}'
    )
    assert isinstance(command, GetQueryResult)
    assert command.result_id == "qr_abc"
    assert command.offset == 10
    assert command.limit == 2
    default = parse_command(b'{"type":"get_query_result","resultId":"qr_abc"}')
    assert isinstance(default, GetQueryResult)
    assert default.offset == 0
    assert default.limit == 500
    too_big = parse_command(
        b'{"type":"get_query_result","resultId":"qr_abc","limit":5001}'
    )
    assert isinstance(too_big, Invalid)


def test_encode_ok_omits_data_and_err_has_string_error() -> None:
    ok = _load(encode(Ok("shutdown", None)))
    assert ok == {"type": "response", "command": "shutdown", "success": True}
    err = _load(encode(Err("parse", None, "nope")))
    assert err == {
        "type": "response",
        "command": "parse",
        "success": False,
        "error": "nope",
    }


def test_encode_agent_end_is_a_sum() -> None:
    completed = _load(
        encode(
            AgentEnd(
                Completed(
                    messages=(),
                    text="done",
                    usage=None,
                    query_results=(),
                    artifacts=(),
                    thread_id=None,
                )
            )
        )
    )
    assert completed["status"] == "completed"
    assert "error" not in completed
    aborted = _load(encode(AgentEnd(Aborted())))
    assert aborted == {"type": "agent_end", "status": "aborted"}
    failed = _load(encode(AgentEnd(Failed("boom"))))
    assert failed == {"type": "agent_end", "status": "error", "error": "boom"}


def test_encode_message_update_uses_pi_envelope() -> None:
    record = _load(encode(MessageUpdate(TextDelta(content_index=0, delta="Hello"))))
    assert record["type"] == "message_update"
    assert record["assistantMessageEvent"] == {
        "type": "text_delta",
        "contentIndex": 0,
        "delta": "Hello",
    }
    start = _load(encode(AgentStart(prompt_id="q1")))
    assert start == {"type": "agent_start", "promptId": "q1"}


def test_encode_sanitizes_non_finite_floats() -> None:
    record = _load(encode(Ok("get_state", None, {"n": float("nan")})))
    assert record["data"]["n"] == "nan"


def test_max_stdin_line_constant() -> None:
    assert MAX_STDIN_LINE == 1_048_576
