"""流式 Provider 的最小协议。"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal, Protocol

from timocode.llm.schemas import LLMRequest, LLMResponse


class ProviderError(Exception):
    """服务请求或响应协议错误，可安全展示给调用方。"""


class ProviderProtocolError(ProviderError):
    """模型响应无法构成有效的内部响应。"""


class ProviderHTTPError(ProviderError):
    def __init__(self, status_code: int, *, retry_after: float | None = None):
        super().__init__(f"模型服务返回 HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderConnectionError(ProviderError):
    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class TextDelta:
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str = ""
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(frozen=True)
class ResponseComplete:
    response: LLMResponse
    type: Literal["response_complete"] = "response_complete"


LLMEvent = TextDelta | ToolCallDelta | ResponseComplete


class LLMProvider(Protocol):
    def stream(self, request: LLMRequest) -> AsyncGenerator[LLMEvent, None]: ...
