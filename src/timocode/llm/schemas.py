"""与服务商格式解耦的消息、工具调用和模型响应。"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class DataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class SystemMessage(DataModel):
    role: Literal["system"] = "system"
    content: str


class UserMessage(DataModel):
    role: Literal["user"] = "user"
    content: str = Field(min_length=1)


class ToolCall(DataModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue]


class AssistantMessage(DataModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolMessage(DataModel):
    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    content: str
    is_error: bool = False


Message = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class ToolDefinition(DataModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    description: str = Field(min_length=1)
    parameters: dict[str, JsonValue]

    @model_validator(mode="after")
    def require_object_schema(self) -> Self:
        if self.parameters.get("type") != "object":
            raise ValueError("工具参数 Schema 必须描述 object")
        return self


class Usage(DataModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class LLMRequest(DataModel):
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    tools: list[ToolDefinition] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class LLMResponse(DataModel):
    message: AssistantMessage
    stop_reason: Literal["stop", "tool_calls", "length"]
    usage: Usage = Field(default_factory=Usage)

    @model_validator(mode="after")
    def check_completion(self) -> Self:
        calls = self.message.tool_calls
        if len({call.id for call in calls}) != len(calls):
            raise ValueError("同一响应中的工具调用 ID 必须唯一")
        if self.stop_reason == "tool_calls" and not calls:
            raise ValueError("tool_calls 响应必须包含工具调用")
        if self.stop_reason == "stop" and (calls or not self.message.content):
            raise ValueError("stop 响应必须包含文本且不包含工具调用")
        return self
