"""按异步执行上下文绑定 HTTP 追踪，Provider 保持独立于会话实现。"""

import asyncio
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

from timocode.llm.base import ProviderError, ProviderHTTPError
from timocode.llm.schemas import LLMResponse, Usage


class HTTPTraceSink(Protocol):
    error: str | None

    @property
    def http_parent_span_id(self) -> str | None: ...

    async def record(self, event: str, data: dict, *, span_id: str | None = None) -> None: ...


_CURRENT_SINK: ContextVar[HTTPTraceSink | None] = ContextVar("timocode_http_trace", default=None)


@contextmanager
def bind_http_trace(sink: HTTPTraceSink | None):
    token = _CURRENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_SINK.reset(token)


class HTTPRequestTrace:
    def __init__(self):
        self.sink = _CURRENT_SINK.get()
        self.request_id = uuid.uuid4().hex
        self.parent_span_id = None
        if self.sink is not None:
            try:
                self.parent_span_id = self.sink.http_parent_span_id
            except Exception as error:
                self.sink.error = f"HTTP 追踪处理失败（{type(error).__name__}）"
        self.attempt = 0
        self.attempt_id: str | None = None
        self.started = 0.0
        self.status_code: int | None = None
        self.server_request_id: str | None = None
        self.transport_error_type: str | None = None
        self.reported_usage: Usage | None = None
        self.reported_stop_reason: str | None = None
        self.client_stream_started = False

    async def _record(self, event: str, data: dict, span_id: str | None) -> None:
        if self.sink is None:
            return
        task = asyncio.current_task()
        cancellation_count = task.cancelling()
        try:
            await self.sink.record(event, {
                "request_id": self.request_id, "parent_span_id": self.parent_span_id,
                "attempt": self.attempt, **data,
            }, span_id=span_id)
        except Exception as error:
            self.sink.error = f"HTTP 追踪处理失败（{type(error).__name__}）"
            if task.cancelling() > cancellation_count:
                raise asyncio.CancelledError from None

    async def start_attempt(self, number: int, model: str, max_attempts: int) -> None:
        self.attempt = number
        self.attempt_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.status_code = None
        self.server_request_id = None
        self.transport_error_type = None
        self.reported_usage = None
        self.reported_stop_reason = None
        self.client_stream_started = False
        await self._record("http_attempt_start", {
            "model": model, "max_attempts": max_attempts,
            "method": "POST", "route": "/chat/completions",
        }, self.attempt_id)

    async def headers(self, status_code: int, server_request_id: str | None) -> None:
        self.status_code = status_code
        self.server_request_id = server_request_id
        await self._record("http_response_headers", {
            "status_code": status_code, "server_request_id": server_request_id,
        }, self.attempt_id)

    async def end_attempt(
        self, status: str, *, error: Exception | None, response: LLMResponse | None,
        emitted: bool, retry_delay: float | None,
    ) -> None:
        usage = response.usage if response is not None else self.reported_usage
        await self._record("http_attempt_end", {
            "status": status, "duration_ms": max(0.0, (time.perf_counter() - self.started) * 1000),
            "status_code": self.status_code, "server_request_id": self.server_request_id,
            "error_type": type(error).__name__ if error is not None else None,
            "transport_error_type": self.transport_error_type,
            "retry_after_seconds": error.retry_after if isinstance(error, ProviderHTTPError) else None,
            "will_retry": retry_delay is not None, "retry_delay_seconds": retry_delay,
            "stream_emitted": emitted,
            "usage": usage.model_dump() if usage is not None else None,
            "stop_reason": response.stop_reason if response is not None else self.reported_stop_reason,
            "response_complete": response is not None,
            "client_stream_started": self.client_stream_started,
        }, self.attempt_id)

    async def wait(self, delay: float, error: ProviderError) -> None:
        span_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = "error"
        try:
            await self._record("http_retry_start", {
                "attempt_id": self.attempt_id, "delay_seconds": delay,
                "source": "retry_after" if isinstance(error, ProviderHTTPError) and error.retry_after is not None else "backoff",
            }, span_id)
            await asyncio.sleep(delay)
            status = "completed"
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            await self._record("http_retry_end", {
                "attempt_id": self.attempt_id, "status": status, "delay_seconds": delay,
                "duration_ms": max(0.0, (time.perf_counter() - started) * 1000),
            }, span_id)
