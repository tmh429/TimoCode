"""线性 JSONL 会话存储，保留完整消息并检查工具调用关联。"""

import asyncio
import errno
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, BinaryIO, Iterator, Literal

from pydantic import AwareDatetime, Field, ValidationError, field_validator

from timocode.llm.schemas import (
    AssistantMessage, DataModel, LLMResponse, Message, ToolCall, ToolMessage, Usage,
)
from timocode.security.workspace import Workspace


Identifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
StopReason = Literal["stop", "tool_calls", "length"]
TailIssue = Literal["missing_newline", "invalid_json", "invalid_utf8"]


class SessionError(Exception):
    """会话存储错误。"""


class SessionCorruptionError(SessionError):
    """会话头、完整记录或消息关联无效。"""


class SessionConflictError(SessionError):
    """条目 ID 冲突或预期序号已经过期。"""


class SessionBusyError(SessionError):
    """会话正在被其他写入者访问。"""


class SessionStorageError(SessionError):
    """写入失败，需要根据报错重新检查磁盘记录。"""


class SessionRecoveryRequired(SessionError):
    """尾行需要显式修复后才能追加。"""


class SessionHeader(DataModel):
    type: Literal["session"]
    version: int
    id: Identifier
    created_at: AwareDatetime
    cwd: str = Field(min_length=1)
    model: str | None = None
    tool_names: list[str] = Field(default_factory=list)

    @field_validator("tool_names")
    @classmethod
    def unique_tool_names(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("会话工具名称必须唯一")
        return value


class SessionEntry(DataModel):
    type: Literal["message"]
    id: Identifier
    sequence: int = Field(ge=1)
    timestamp: AwareDatetime
    message: Message
    usage: Usage | None = None
    stop_reason: StopReason | None = None


class SessionSnapshot(DataModel):
    header: SessionHeader
    entries: list[SessionEntry]
    pending_tool_calls: list[ToolCall]
    file_sha256: str
    tail_issue: TailIssue | None = None
    backup_path: str | None = None

    @property
    def messages(self) -> list[Message]:
        return [entry.message.model_copy(deep=True) for entry in self.entries]

    @property
    def last_sequence(self) -> int:
        return len(self.entries)

    @property
    def needs_recovery(self) -> bool:
        return self.tail_issue is not None or bool(self.pending_tool_calls)


def _write_record(stream: BinaryIO, data: bytes) -> None:
    if stream.write(data) != len(data):
        raise OSError("记录写入不完整")
    stream.flush()
    os.fsync(stream.fileno())


def _encode_record(record: SessionHeader | SessionEntry) -> bytes:
    return (record.model_dump_json() + "\n").encode("utf-8")


def _check_entry(
    entry: SessionEntry,
    index: int,
    entry_ids: set[str],
    call_ids: set[str],
    pending: dict[str, ToolCall],
) -> None:
    if entry.sequence != index or entry.id in entry_ids:
        raise SessionCorruptionError(f"第 {index} 条消息的序号或 ID 无效")
    message = entry.message
    if entry.usage is not None or entry.stop_reason is not None:
        if not isinstance(message, AssistantMessage):
            raise SessionCorruptionError("仅 assistant 消息可携带模型响应元数据")
        if entry.stop_reason is not None:
            try:
                LLMResponse(message=message, stop_reason=entry.stop_reason)
            except ValidationError:
                raise SessionCorruptionError("响应停止原因与消息内容不一致") from None
    if isinstance(message, ToolMessage):
        call = pending.get(message.tool_call_id)
        if call is None or call.name != message.name:
            raise SessionCorruptionError("工具结果缺少对应调用、重复回传或工具名称不匹配")
        del pending[message.tool_call_id]
    else:
        if pending:
            raise SessionCorruptionError("工具调用尚缺结果，不能插入其他消息")
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if call.id in call_ids:
                    raise SessionCorruptionError("工具调用 ID 重复")
                call_ids.add(call.id)
                pending[call.id] = call
    entry_ids.add(entry.id)


class SessionStore:
    def __init__(self, directory: str | Path = ".timocode/sessions"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._workspace = Workspace(directory)
        self.directory = self._workspace.root
        self._mutex = asyncio.Lock()

    async def _run_io(self, operation: Callable[[], SessionSnapshot]) -> SessionSnapshot:
        async with self._mutex:
            worker = asyncio.create_task(asyncio.to_thread(operation))
            cancelled = False
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:
                    break
            if cancelled:
                # 事务可能已提交；调用方可用相同 entry_id 重试或重新加载确认。
                if not worker.cancelled():
                    worker.exception()
                raise asyncio.CancelledError
            return worker.result()

    async def create(
        self, cwd: str | Path, *, model: str | None = None, session_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> SessionSnapshot:
        tool_names = list(tool_names) if tool_names is not None else []
        return await self._run_io(lambda: self._create(
            cwd, model=model, session_id=session_id, tool_names=tool_names,
        ))

    async def load(self, session_id: str) -> SessionSnapshot:
        return await self._run_io(lambda: self._load(session_id))

    async def append(
        self, session_id: str, message: Message, *, entry_id: str | None = None,
        expected_sequence: int | None = None, usage: Usage | None = None,
        stop_reason: StopReason | None = None,
    ) -> SessionSnapshot:
        if expected_sequence is not None and (type(expected_sequence) is not int or expected_sequence < 0):
            raise ValueError("预期序号必须是非负整数")
        message = message.model_copy(deep=True)
        usage = usage.model_copy(deep=True) if usage is not None else None
        return await self._run_io(lambda: self._append(
            session_id, message, entry_id=entry_id, expected_sequence=expected_sequence,
            usage=usage, stop_reason=stop_reason,
        ))

    async def repair_tail(self, session_id: str, *, expected_sha256: str) -> SessionSnapshot:
        return await self._run_io(lambda: self._repair_tail(session_id, expected_sha256=expected_sha256))

    def _path(self, session_id: str, suffix: str = ".jsonl") -> Path:
        # 使用已验证的 ID 生成文件名，禁止调用者把任意路径当作会话 ID。
        if len(session_id) != 32 or any(char not in "0123456789abcdef" for char in session_id):
            raise ValueError("会话 ID 必须是 32 位小写十六进制字符串")
        return self._workspace.resolve_path(session_id + suffix, must_exist=False)

    @contextmanager
    def _locked(self, session_id: str) -> Iterator[None]:
        path = self._path(session_id, ".lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\x00")
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    raise SessionBusyError("会话正在被其他操作访问，请稍后重试") from None
                raise
            yield
        finally:
            # 关闭描述符会释放对应的 OS 文件锁，锁文件保留以避免锁目标切换。
            os.close(fd)

    def _read(self, session_id: str) -> tuple[SessionSnapshot, int]:
        entries: list[SessionEntry] = []
        entry_ids: set[str] = set()
        call_ids: set[str] = set()
        pending: dict[str, ToolCall] = {}
        digest = hashlib.sha256()
        valid_bytes = 0
        tail_issue = None
        header = None
        with self._path(session_id).open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                terminated = raw.endswith(b"\n")
                try:
                    text = raw.decode("utf-8")
                    json.loads(text)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    if terminated or header is None:
                        raise SessionCorruptionError(f"第 {line_number} 行 JSON 或 UTF-8 无效") from None
                    tail_issue = "invalid_utf8" if isinstance(error, UnicodeDecodeError) else "invalid_json"
                    break
                try:
                    if line_number == 1:
                        header = SessionHeader.model_validate_json(text)
                        if header.version != 1 or header.id != session_id:
                            raise SessionCorruptionError("会话格式版本或会话 ID 不匹配")
                    else:
                        entry = SessionEntry.model_validate_json(text)
                        _check_entry(entry, len(entries) + 1, entry_ids, call_ids, pending)
                        entries.append(entry)
                except ValidationError:
                    raise SessionCorruptionError(f"第 {line_number} 行记录字段无效") from None
                valid_bytes += len(raw)
                if not terminated:
                    tail_issue = "missing_newline"
        if header is None:
            raise SessionCorruptionError("文件缺少有效会话头")
        return SessionSnapshot(
            header=header, entries=entries,
            pending_tool_calls=[call.model_copy(deep=True) for call in pending.values()],
            file_sha256=digest.hexdigest(), tail_issue=tail_issue,
        ), valid_bytes

    def _create(
        self, cwd: str | Path, *, model: str | None = None, session_id: str | None = None,
        tool_names: list[str] | None = None,
    ) -> SessionSnapshot:
        session_id = session_id if session_id is not None else uuid.uuid4().hex
        working_directory = Path(cwd).resolve(strict=True)
        if not working_directory.is_dir():
            raise ValueError("工作目录必须是现有目录")
        header = SessionHeader(
            type="session", version=1, id=session_id,
            created_at=datetime.now(timezone.utc), cwd=str(working_directory), model=model,
            tool_names=tool_names if tool_names is not None else [],
        )
        with self._locked(session_id):
            path = self._path(session_id)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as stream:
                    _write_record(stream, _encode_record(header))
            except OSError:
                path.unlink(missing_ok=True)
                raise SessionStorageError("创建会话失败，未保留不完整会话文件") from None
            return self._read(session_id)[0]

    def _load(self, session_id: str) -> SessionSnapshot:
        with self._locked(session_id):
            return self._read(session_id)[0]

    def _append(
        self, session_id: str, message: Message, *, entry_id: str | None = None,
        expected_sequence: int | None = None, usage: Usage | None = None,
        stop_reason: StopReason | None = None,
    ) -> SessionSnapshot:
        with self._locked(session_id):
            snapshot, _ = self._read(session_id)
            if snapshot.tail_issue is not None:
                raise SessionRecoveryRequired("会话尾行异常，请先显式修复")
            entry = SessionEntry(
                type="message", id=entry_id if entry_id is not None else uuid.uuid4().hex,
                sequence=snapshot.last_sequence + 1, timestamp=datetime.now(timezone.utc),
                message=message, usage=usage, stop_reason=stop_reason,
            )
            # 重新校验可变的 Pydantic 对象，并创建与调用方隔离的快照。
            entry = SessionEntry.model_validate_json(entry.model_dump_json())
            for existing in snapshot.entries:
                if existing.id == entry.id:
                    if (existing.message, existing.usage, existing.stop_reason) != (entry.message, entry.usage, entry.stop_reason):
                        raise SessionConflictError("条目 ID 已被不同内容使用")
                    return snapshot
            if expected_sequence is not None and expected_sequence != snapshot.last_sequence:
                raise SessionConflictError("预期序号已过期，请重新加载会话")
            entry_ids = {item.id for item in snapshot.entries}
            call_ids = {
                call.id for item in snapshot.entries if isinstance(item.message, AssistantMessage)
                for call in item.message.tool_calls
            }
            pending = {call.id: call for call in snapshot.pending_tool_calls}
            _check_entry(entry, snapshot.last_sequence + 1, entry_ids, call_ids, pending)
            path = self._path(session_id)
            with path.open("r+b") as stream:
                original_size = stream.seek(0, os.SEEK_END)
                try:
                    _write_record(stream, _encode_record(entry))
                except OSError:
                    try:
                        stream.seek(original_size)
                        stream.truncate()
                        stream.flush()
                        os.fsync(stream.fileno())
                    except OSError:
                        raise SessionStorageError("追加失败且回退失败，磁盘状态需重新检查") from None
                    raise SessionStorageError("追加失败，文件已回退到追加前状态") from None
            return self._read(session_id)[0]

    def _repair_tail(self, session_id: str, *, expected_sha256: str) -> SessionSnapshot:
        with self._locked(session_id):
            snapshot, valid_bytes = self._read(session_id)
            if snapshot.file_sha256 != expected_sha256:
                raise SessionConflictError("文件已变化，请重新加载后修复")
            if snapshot.tail_issue is None:
                return snapshot
            path = self._path(session_id)
            backup = self._path(session_id, f".{uuid.uuid4().hex}.bak")
            temporary = self._path(session_id, f".{uuid.uuid4().hex}.tmp")
            try:
                backup_fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(backup_fd, "wb") as target, path.open("rb") as source:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                temporary_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(temporary_fd, "wb") as target, path.open("rb") as source:
                    remaining = valid_bytes
                    while remaining:
                        chunk = source.read(min(65536, remaining))
                        if not chunk:
                            raise OSError("修复期间原文件长度发生变化")
                        target.write(chunk)
                        remaining -= len(chunk)
                    if snapshot.tail_issue == "missing_newline":
                        target.write(b"\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, path)
            except OSError:
                raise SessionStorageError("尾行修复失败，请保留并检查原文件与备份") from None
            finally:
                temporary.unlink(missing_ok=True)
            repaired = self._read(session_id)[0]
            repaired.backup_path = str(backup)
            return repaired
