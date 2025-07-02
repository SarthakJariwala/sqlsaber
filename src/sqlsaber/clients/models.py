"""Data models for LLM client requests and responses."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Message roles in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ContentType(str, Enum):
    """Content block types."""

    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class ToolChoiceType(str, Enum):
    """Tool choice types."""

    AUTO = "auto"
    ANY = "any"
    TOOL = "tool"
    NONE = "none"


class StopReason(str, Enum):
    """Stop reasons for message completion."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


@dataclass
class ContentBlock:
    """A content block in a message."""

    type: ContentType
    content: str | dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        if self.type == ContentType.TEXT:
            return {"type": "text", "text": self.content}
        elif self.type == ContentType.TOOL_USE:
            return {
                "type": "tool_use",
                "id": self.content["id"],
                "name": self.content["name"],
                "input": self.content["input"],
            }
        elif self.type == ContentType.TOOL_RESULT:
            return {
                "type": "tool_result",
                "tool_use_id": self.content["tool_use_id"],
                "content": self.content["content"],
            }
        else:
            return {"type": self.type.value, **self.content}


@dataclass
class Message:
    """A message in a conversation."""

    role: MessageRole
    content: str | list[ContentBlock]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API requests."""
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}
        else:
            return {
                "role": self.role.value,
                "content": [block.to_dict() for block in self.content],
            }


@dataclass
class ToolDefinition:
    """Definition of a tool that can be called."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API requests."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolChoice:
    """Tool choice configuration."""

    type: ToolChoiceType
    name: str | None = None
    disable_parallel_tool_use: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API requests."""
        result = {"type": self.type.value}
        if self.name:
            result["name"] = self.name
        if self.disable_parallel_tool_use:
            result["disable_parallel_tool_use"] = True
        return result


@dataclass
class CreateMessageRequest:
    """Request to create a message."""

    model: str
    messages: list[Message]
    max_tokens: int
    system: str | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: ToolChoice | None = None
    temperature: float | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API requests."""
        data = {
            "model": self.model,
            "messages": [msg.to_dict() for msg in self.messages],
            "max_tokens": self.max_tokens,
        }

        if self.system:
            data["system"] = self.system
        if self.tools:
            data["tools"] = [tool.to_dict() for tool in self.tools]
        if self.tool_choice:
            data["tool_choice"] = self.tool_choice.to_dict()
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.stream:
            data["stream"] = True
        if self.stop_sequences:
            data["stop_sequences"] = self.stop_sequences

        return data


@dataclass
class Usage:
    """Token usage information."""

    input_tokens: int
    output_tokens: int


@dataclass
class MessageResponse:
    """Response from message creation."""

    id: str
    model: str
    role: MessageRole
    content: list[ContentBlock]
    stop_reason: StopReason
    stop_sequence: str | None
    usage: Usage

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MessageResponse":
        """Create from API response dictionary."""
        content_blocks = []
        for block_data in data["content"]:
            if block_data["type"] == "text":
                content_blocks.append(
                    ContentBlock(ContentType.TEXT, block_data["text"])
                )
            elif block_data["type"] == "tool_use":
                content_blocks.append(
                    ContentBlock(
                        ContentType.TOOL_USE,
                        {
                            "id": block_data["id"],
                            "name": block_data["name"],
                            "input": block_data["input"],
                        },
                    )
                )

        return cls(
            id=data["id"],
            model=data["model"],
            role=MessageRole(data["role"]),
            content=content_blocks,
            stop_reason=StopReason(data["stop_reason"]),
            stop_sequence=data.get("stop_sequence"),
            usage=Usage(
                input_tokens=data["usage"]["input_tokens"],
                output_tokens=data["usage"]["output_tokens"],
            ),
        )


# Stream event types and data models for streaming
@dataclass
class StreamEventData:
    """Base class for stream event data."""

    pass


@dataclass
class TextDeltaData(StreamEventData):
    """Text delta stream event data."""

    text: str


@dataclass
class ToolUseStartData(StreamEventData):
    """Tool use start stream event data."""

    id: str
    name: str


@dataclass
class ToolUseInputData(StreamEventData):
    """Tool use input stream event data."""

    partial_json: str


@dataclass
class MessageStartData(StreamEventData):
    """Message start stream event data."""

    message: dict[str, Any]


@dataclass
class MessageDeltaData(StreamEventData):
    """Message delta stream event data."""

    delta: dict[str, Any]
    usage: dict[str, Any] | None = None


@dataclass
class ContentBlockStartData(StreamEventData):
    """Content block start stream event data."""

    index: int
    content_block: dict[str, Any]


@dataclass
class ContentBlockDeltaData(StreamEventData):
    """Content block delta stream event data."""

    index: int
    delta: dict[str, Any]


@dataclass
class ContentBlockStopData(StreamEventData):
    """Content block stop stream event data."""

    index: int
