"""通过 HTTPX 消费 OpenAI-compatible Chat Completions 的 SSE 流。"""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Literal, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field

from timocode.llm.base import (
    LLMEvent,
    ProviderError,
    ProviderConnectionError,
    ProviderHTTPError,
    ProviderProtocolError,
    ResponseComplete,
    TextDelta,
    ToolCallDelta,
)
from timocode.llm.config import LLMSettings
from timocode.llm.retry import parse_retry_after
from timocode.llm.schemas import (
    AssistantMessage,
    LLMRequest,
    LLMResponse,
    ToolCall,
    Usage,
)


class _ChunkModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)


class _FunctionDelta(_ChunkModel):
    name: str | None = None
    arguments: str | None = None


class _ToolDelta(_ChunkModel):
    index: int = Field(ge=0)
    id: str | None = None
    type: Literal["function"] | None = None
    function: _FunctionDelta = Field(default_factory=_FunctionDelta)


class _Delta(_ChunkModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[_ToolDelta] | None = None


class _Choice(_ChunkModel):
    index: Literal[0]
    delta: _Delta = Field(default_factory=_Delta)
    finish_reason: str | None = None


class _Usage(_ChunkModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class _Chunk(_ChunkModel):
    choices: list[_Choice]
    usage: _Usage | None = None


@dataclass
class _ToolParts:
    id: str = ""
    name: str = ""
    arguments: str = ""


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """按 SSE 空行边界组装 data 字段，支持注释、CRLF 和多行数据。"""
    data_lines: list[str] = []
    async for line in response.aiter_lines():
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
        elif line.startswith("data:"):
            value = line[5:]
            data_lines.append(value.removeprefix(" "))
    if data_lines:
        yield "\n".join(data_lines)


def _request_payload(request: LLMRequest) -> dict:
    messages = []
    for message in request.messages:
        item = {"role": message.role, "content": message.content}
        if isinstance(message, AssistantMessage) and message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False, allow_nan=False),
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == "tool":
            item["tool_call_id"] = message.tool_call_id
        messages.append(item)

    payload = {
        "model": request.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request.tools:
        payload["tools"] = [
            {"type": "function", "function": tool.model_dump()} for tool in request.tools
        ]
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    return payload


class OpenAICompatibleProvider:
    def __init__(self, settings: LLMSettings, *, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]:
        payload = _request_payload(request)
        policy = self.settings.retry_policy.model_copy(deep=True)
        for attempt in range(policy.max_retries + 1):
            emitted = False
            try:
                async with aclosing(self._stream_once(payload)) as stream:
                    async for event in stream:
                        emitted = True
                        yield event
                return
            except (ProviderHTTPError, ProviderConnectionError) as error:
                delay = policy.delay(attempt, error)
                if emitted or delay is None:
                    raise
            await asyncio.sleep(delay)

    async def _stream_once(self, request_payload: dict) -> AsyncGenerator[LLMEvent, None]:
        text_parts: list[str] = []
        tool_parts: dict[int, _ToolParts] = {}
        finish_reason: str | None = None
        usage = Usage()
        received_done = False
        received_response = False

        try:
            async with self._client.stream(
                "POST",
                self.settings.base_url + "/chat/completions",
                headers={
                    "Authorization": "Bearer " + self.settings.api_key.get_secret_value(),
                    "Accept": "text/event-stream",
                },
                json=request_payload,
                timeout=self.settings.timeout,
            ) as response:
                received_response = True
                if response.status_code >= 400:
                    raise ProviderHTTPError(
                        response.status_code,
                        retry_after=parse_retry_after(response.headers.get("retry-after")),
                    )
                if "text/event-stream" not in response.headers.get("content-type", "").lower():
                    raise ProviderProtocolError("模型服务未返回 SSE 事件流")

                async for data in _sse_data(response):
                    if data == "[DONE]":
                        received_done = True
                        break
                    try:
                        payload = json.loads(data)
                        if isinstance(payload, dict) and "error" in payload:
                            raise ProviderError("模型服务在事件流中返回错误")
                        chunk = _Chunk.model_validate(payload)
                    except (ValueError, TypeError):
                        raise ProviderProtocolError("模型返回了无效的 SSE 数据") from None

                    if chunk.usage is not None:
                        usage = Usage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) != 1 or finish_reason is not None:
                        raise ProviderProtocolError("模型返回了重复或多候选响应")
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta.content:
                        text_parts.append(delta.content)
                        yield TextDelta(delta=delta.content)
                    for fragment in delta.tool_calls or []:
                        parts = tool_parts.setdefault(fragment.index, _ToolParts())
                        if fragment.id:
                            if parts.id and parts.id != fragment.id:
                                raise ProviderProtocolError("工具调用 ID 在流式过程中发生变化")
                            parts.id = fragment.id
                        parts.name += fragment.function.name or ""
                        parts.arguments += fragment.function.arguments or ""
                        yield ToolCallDelta(
                            index=fragment.index,
                            id=fragment.id,
                            name=fragment.function.name,
                            arguments_delta=fragment.function.arguments or "",
                        )
                    finish_reason = choice.finish_reason
        except (httpx.ConnectTimeout, httpx.PoolTimeout):
            raise ProviderConnectionError("模型连接或等待连接池超时", retryable=not received_response) from None
        except httpx.ConnectError:
            raise ProviderConnectionError("模型请求连接失败", retryable=not received_response) from None
        except httpx.TimeoutException:
            raise ProviderError("模型请求超时") from None
        except httpx.HTTPError:
            raise ProviderError("模型请求连接或传输失败") from None

        if not received_done or finish_reason is None:
            raise ProviderProtocolError("模型响应流在完整结束前中断")
        if finish_reason not in {"stop", "tool_calls", "length"}:
            raise ProviderProtocolError(f"不支持的模型停止原因 {finish_reason}")

        try:
            calls = []
            if finish_reason != "length":
                if sorted(tool_parts) != list(range(len(tool_parts))):
                    raise ProviderProtocolError("工具调用索引不连续")
                for index in sorted(tool_parts):
                    parts = tool_parts[index]
                    calls.append(
                        ToolCall(id=parts.id, name=parts.name, arguments=json.loads(parts.arguments))
                    )
            message = AssistantMessage(content="".join(text_parts) or None, tool_calls=calls)
            result = LLMResponse(message=message, stop_reason=finish_reason, usage=usage)
        except (ValueError, TypeError):
            raise ProviderProtocolError("模型的完整消息或工具参数无效") from None
        yield ResponseComplete(response=result)
