"""每次执行独占的 JSONL 追踪，记录失败时保留主任务执行语义。"""

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from itertools import islice
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from timocode.agent.state import AgentEvent
from timocode.llm.schemas import AssistantMessage
from timocode.security.execution import complete_task
from timocode.security.workspace import Workspace


class TraceRecorder:
    def __init__(
        self, directory: str | Path, session_id: str, model: str, *,
        capture_content: bool = False, secrets: Sequence[str] = (),
    ):
        self.directory = Path(directory)
        self.session_id = session_id
        self.model = model
        self.capture_content = capture_content is True
        self._secrets = sorted({value for value in secrets if value}, key=len, reverse=True)
        self.run_id = uuid.uuid4().hex
        self.path: Path | None = None
        self.error: str | None = None
        self.status = "completed"
        self._stream = None
        self._sequence = 0
        self._started = time.perf_counter()
        self._lock = asyncio.Lock()
        self._spans: dict[str, tuple[str, float]] = {}
        self._closed = False

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", text)

    def _preview(self, value, *, depth=0, budget=None):
        budget = [256] if budget is None else budget
        budget[0] -= 1
        if budget[0] < 0 or depth > 8:
            return "[TRUNCATED]"
        if isinstance(value, str):
            text = self._redact(value)
            return text if len(text) <= 2048 else text[:2048] + "[TRUNCATED]"
        if isinstance(value, dict):
            result = {}
            for key, item in islice(value.items(), 20):
                normalized = key.casefold().replace("-", "_")
                sensitive = re.search(r"(?:^|_)(api_?key|authorization|password|passwd|secret|token|private_key)$", normalized)
                result[self._redact(key)[:128]] = "[REDACTED]" if sensitive else self._preview(item, depth=depth + 1, budget=budget)
            if len(value) > 20:
                result["_truncated_items"] = len(value) - 20
            return result
        if isinstance(value, list):
            result = [self._preview(item, depth=depth + 1, budget=budget) for item in value[:20]]
            if len(value) > 20:
                result.append({"_truncated_items": len(value) - 20})
            return result
        return value

    def content(self, value) -> dict:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        data = serialized.encode("utf-8")
        result = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        if isinstance(value, dict):
            result["keys"] = list(value)[:20]
        if self.capture_content:
            result["preview"] = self._preview(value)
        return result

    async def _io(self, operation) -> None:
        task = asyncio.current_task()
        cancellation_count = task.cancelling()
        try:
            await complete_task(asyncio.create_task(asyncio.to_thread(operation)))
        except Exception as error:
            self.error = f"执行追踪写入失败（{type(error).__name__}）"
            if task.cancelling() > cancellation_count:
                raise asyncio.CancelledError from None

    async def start(self, operation: str) -> None:
        def open_file():
            self.directory.mkdir(parents=True, exist_ok=True)
            workspace = Workspace(self.directory)
            path = workspace.resolve_path("trace-" + self.run_id + ".jsonl", must_exist=False)
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                self._stream = os.fdopen(fd, "w+b")
            except Exception:
                os.close(fd)
                raise
            self.path = path
        await self._io(open_file)
        await self.record("trace_start", {"operation": operation})

    async def record(self, event: str, data: dict, *, span_id: str | None = None) -> None:
        if self.error is not None or self._stream is None or self._closed:
            return
        try:
            prepared = self._preview(data)
        except Exception as error:
            self.error = f"执行追踪处理失败（{type(error).__name__}）"
            return
        async with self._lock:
            if self.error is not None or self._closed:
                return
            def append():
                entry = {
                    "version": 1, "run_id": self.run_id, "session_id": self._preview(self.session_id),
                    "model": self._preview(self.model), "sequence": self._sequence + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(), "event": self._redact(event)[:128],
                    "elapsed_ms": max(0.0, (time.perf_counter() - self._started) * 1000),
                    "span_id": span_id, "data": prepared,
                }
                encoded = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                position = self._stream.seek(0, os.SEEK_END)
                try:
                    if self._stream.write(encoded) != len(encoded):
                        raise OSError("partial trace write")
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                except OSError:
                    self._stream.seek(position)
                    self._stream.truncate()
                    raise
                self._sequence += 1
            await self._io(append)

    async def start_span(self, key: str, kind: str, data: dict) -> None:
        span_id = uuid.uuid4().hex
        self._spans[key] = (span_id, time.perf_counter())
        await self.record(kind + "_start", data, span_id=span_id)

    async def end_span(self, key: str, kind: str, data: dict) -> None:
        span = self._spans.pop(key, None)
        if span is None:
            return
        span_id, started = span
        await self.record(kind + "_end", {
            **data, "duration_ms": max(0.0, (time.perf_counter() - started) * 1000),
        }, span_id=span_id)

    async def observe(self, event: AgentEvent, *, session_sequence: int | None = None) -> None:
        if event.type in {"text_delta", "tool_call_delta"}:
            return
        if event.type == "model_request_start":
            request = event.request
            data = {} if request is None else {
                "model": request.model, "message_count": len(request.messages),
                "tool_names": [tool.name for tool in request.tools], "max_tokens": request.max_tokens,
                "temperature": request.temperature, "session_sequence": session_sequence,
            }
            if self.capture_content and request is not None:
                data["messages"] = [message.model_dump(mode="json") for message in request.messages[:20]]
                data["messages_omitted"] = max(0, len(request.messages) - 20)
            await self.start_span("model", "model", data)
        elif event.type == "message_complete":
            message = event.message
            if isinstance(message, AssistantMessage) and event.response is not None:
                await self.end_span("model", "model", {
                    "status": event.response.stop_reason, "usage": event.response.usage.model_dump(),
                    "tool_call_ids": [call.id for call in message.tool_calls],
                    "message": self.content(message.model_dump(mode="json")),
                })
            await self.record("message_saved", {
                "role": message.role, "session_sequence": session_sequence,
                "message": self.content(message.model_dump(mode="json")),
            })
        elif event.type == "tool_start":
            call = event.tool_call
            await self.start_span("tool:" + call.id, "tool", {"call_id": call.id, "name": call.name, "arguments": self.content(call.arguments)})
        elif event.type == "tool_end":
            call = event.tool_call
            await self.end_span("tool:" + call.id, "tool", {
                "call_id": call.id, "name": call.name, "status": "error" if event.message.is_error else "ok",
                "result": self.content(event.message.content),
            })
        elif event.type == "context_checked":
            report = event.context_report
            await self.record(event.type, {"estimated_input": report.estimate.total, "input_limit": report.input_limit, "fits": report.fits, "method": report.estimate.method})
        elif event.type == "compaction_start":
            await self.start_span("compaction", "compaction", {})
        elif event.type in {"compaction_end", "compaction_failed"}:
            await self.end_span("compaction", "compaction", {
                "status": "failed" if event.type == "compaction_failed" else event.reason,
                "error": self.content(event.reason) if event.type == "compaction_failed" else None,
            })
        elif event.type == "run_end":
            result = event.result
            self.status = result.status if result is not None else (event.reason or "unknown")
            await self._end_pending(self.status)
            await self.record(event.type, {
                "status": self.status, "iterations": result.iterations if result is not None else None,
                "error": self.content(result.error) if result is not None and result.error else None,
            })
        else:
            await self.record(event.type, {})

    async def _end_pending(self, status: str) -> None:
        for key in list(self._spans):
            await self.end_span(key, key.split(":", 1)[0], {"status": status, "interrupted": True})

    async def finish(self, error: BaseException | None = None) -> None:
        if self._closed:
            return
        async def close():
            try:
                await self._end_pending(self.status)
                await self.record("trace_end", {
                    "status": self.status, "exception_type": type(error).__name__ if error is not None else None,
                    "error": self.content(str(error)) if error is not None else None,
                })
            except Exception as failure:
                self.error = f"执行追踪收尾失败（{type(failure).__name__}）"
            finally:
                async with self._lock:
                    self._closed = True
                    if self._stream is not None:
                        await self._io(self._stream.close)
        await complete_task(asyncio.create_task(close()))
