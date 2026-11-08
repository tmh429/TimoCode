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

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError, field_validator, model_validator

from timocode.agent.compaction import AutoCompactionPolicy
from timocode.agent.context import ContextBudget
from timocode.agent.history import check_tool_association, split_history
from timocode.agent.summarization import HistorySummary
from timocode.llm.schemas import (
    AssistantMessage, DataModel, LLMResponse, Message, SystemMessage, ToolCall, Usage, UserMessage,
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
    context_budget: ContextBudget | None = None
    auto_compaction: AutoCompactionPolicy | None = None

    @model_validator(mode="after")
    def require_compaction_budget(self) -> "SessionHeader":
        if self.auto_compaction is not None and self.context_budget is None:
            raise ValueError("自动压缩必须配置任务上下文预算")
        return self

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


class CompactionEntry(DataModel):
    type: Literal["compaction"]
    id: Identifier
    sequence: int = Field(ge=1)
    timestamp: AwareDatetime
    source_sequence: int = Field(ge=1)
    first_kept_entry_id: Identifier
    summary: HistorySummary
    usage: Usage | None = None


class ModelChangeEntry(DataModel):
    type: Literal["model_change"]
    id: Identifier
    sequence: int = Field(ge=1)
    timestamp: AwareDatetime
    model: str = Field(min_length=1)

    @field_validator("model")
    @classmethod
    def require_model_name(cls, value: str) -> str:
        if any(character.isspace() for character in value):
            raise ValueError("模型名称不能包含空白字符")
        return value


SessionRecord = Annotated[SessionEntry | CompactionEntry | ModelChangeEntry, Field(discriminator="type")]
RECORD_ADAPTER = TypeAdapter(SessionRecord)


class SessionSnapshot(DataModel):
    header: SessionHeader
    entries: list[SessionRecord]
    pending_tool_calls: list[ToolCall]
    file_sha256: str
    tail_issue: TailIssue | None = None
    backup_path: str | None = None

    @property
    def model(self) -> str | None:
        for entry in reversed(self.entries):
            if isinstance(entry, ModelChangeEntry):
                return entry.model
        return self.header.model

    @property
    def messages(self) -> list[Message]:
        return [
            entry.message.model_copy(deep=True) for entry in self.entries
            if isinstance(entry, SessionEntry)
        ]

    @property
    def latest_compaction(self) -> CompactionEntry | None:
        for entry in reversed(self.entries):
            if isinstance(entry, CompactionEntry):
                return entry.model_copy(deep=True)
        return None

    @property
    def context_messages(self) -> list[Message]:
        compaction = self.latest_compaction
        if compaction is None:
            return self.messages
        first_kept = next(entry for entry in self.entries if entry.id == compaction.first_kept_entry_id)
        messages: list[Message] = []
        for message in self.messages:
            if not isinstance(message, SystemMessage):
                break
            messages.append(message)
        messages.append(UserMessage(
            content="以下 JSON 为较早历史的摘要，供继续任务时参考。\n" + compaction.summary.model_dump_json(),
        ))
        messages.extend(
            entry.message.model_copy(deep=True) for entry in self.entries
            if isinstance(entry, SessionEntry) and entry.sequence >= first_kept.sequence
        )
        return messages

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


def _encode_record(record: SessionHeader | SessionRecord) -> bytes:
    return (record.model_dump_json() + "\n").encode("utf-8")


def _check_entry(
    entry: SessionRecord,
    index: int,
    entry_ids: set[str],
    call_ids: set[str],
    pending: dict[str, ToolCall],
    entries: list[SessionRecord],
) -> None:
    if entry.sequence != index or entry.id in entry_ids:
        raise SessionCorruptionError(f"第 {index} 条消息的序号或 ID 无效")
    if isinstance(entry, ModelChangeEntry):
        if pending:
            raise SessionCorruptionError("模型切换之前存在未完成工具调用")
        entry_ids.add(entry.id)
        return
    if isinstance(entry, CompactionEntry):
        _check_compaction(entry, entries, pending)
        entry_ids.add(entry.id)
        return
    message = entry.message
    if entry.usage is not None or entry.stop_reason is not None:
        if not isinstance(message, AssistantMessage):
            raise SessionCorruptionError("仅 assistant 消息可携带模型响应元数据")
        if entry.stop_reason is not None:
            try:
                LLMResponse(message=message, stop_reason=entry.stop_reason)
            except ValidationError:
                raise SessionCorruptionError("响应停止原因与消息内容不一致") from None
    try:
        check_tool_association(message, pending, call_ids)
    except ValueError as error:
        raise SessionCorruptionError(str(error)) from None
    entry_ids.add(entry.id)


def _check_compaction(
    entry: CompactionEntry, entries: list[SessionRecord], pending: dict[str, ToolCall],
) -> None:
    if pending:
        raise SessionCorruptionError("压缩记录之前存在未完成工具调用")
    if entry.source_sequence != entry.sequence - 1:
        raise SessionCorruptionError("压缩来源序号与记录位置不一致")
    messages = [item for item in entries if isinstance(item, SessionEntry)]
    first_kept = next((item for item in messages if item.id == entry.first_kept_entry_id), None)
    if first_kept is None or not isinstance(first_kept.message, UserMessage):
        raise SessionCorruptionError("压缩保留边界必须指向已有用户消息")
    if not any(
        isinstance(item.message, UserMessage) and item.sequence < first_kept.sequence
        for item in messages
    ):
        raise SessionCorruptionError("压缩边界之前缺少可摘要的用户回合")
    try:
        split_history([item.message for item in messages])
    except ValueError as error:
        raise SessionCorruptionError(str(error)) from None
    previous = next((item for item in reversed(entries) if isinstance(item, CompactionEntry)), None)
    if previous is not None:
        previous_kept = next(item for item in messages if item.id == previous.first_kept_entry_id)
        if first_kept.sequence <= previous_kept.sequence:
            raise SessionCorruptionError("新的压缩边界必须向前推进")


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
        tool_names: list[str] | None = None, context_budget: ContextBudget | None = None,
        auto_compaction: AutoCompactionPolicy | None = None,
    ) -> SessionSnapshot:
        context_budget = context_budget.model_copy(deep=True) if context_budget is not None else None
        auto_compaction = auto_compaction.model_copy(deep=True) if auto_compaction is not None else None
        tool_names = list(tool_names) if tool_names is not None else []
        return await self._run_io(lambda: self._create(
            cwd, model=model, session_id=session_id, tool_names=tool_names, context_budget=context_budget,
            auto_compaction=auto_compaction,
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

    async def append_compaction(
        self, session_id: str, summary: HistorySummary, *, first_kept_entry_id: str,
        expected_sequence: int, entry_id: str | None = None, usage: Usage | None = None,
    ) -> SessionSnapshot:
        if type(expected_sequence) is not int or expected_sequence < 1:
            raise ValueError("压缩预期序号必须为正整数")
        summary = summary.model_copy(deep=True)
        usage = usage.model_copy(deep=True) if usage is not None else None
        return await self._run_io(lambda: self._append_compaction(
            session_id, summary, first_kept_entry_id=first_kept_entry_id,
            expected_sequence=expected_sequence, entry_id=entry_id, usage=usage,
        ))

    async def append_model_change(
        self, session_id: str, model: str, *, expected_sequence: int, entry_id: str | None = None,
    ) -> SessionSnapshot:
        if type(expected_sequence) is not int or expected_sequence < 1:
            raise ValueError("模型切换预期序号必须为正整数")
        return await self._run_io(lambda: self._append_model_change(
            session_id, model, expected_sequence=expected_sequence, entry_id=entry_id,
        ))

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
        entries: list[SessionRecord] = []
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
                        entry = RECORD_ADAPTER.validate_json(text)
                        _check_entry(entry, len(entries) + 1, entry_ids, call_ids, pending, entries)
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
        tool_names: list[str] | None = None, context_budget: ContextBudget | None = None,
        auto_compaction: AutoCompactionPolicy | None = None,
    ) -> SessionSnapshot:
        session_id = session_id if session_id is not None else uuid.uuid4().hex
        working_directory = Path(cwd).resolve(strict=True)
        if not working_directory.is_dir():
            raise ValueError("工作目录必须是现有目录")
        header = SessionHeader(
            type="session", version=1, id=session_id,
            created_at=datetime.now(timezone.utc), cwd=str(working_directory), model=model,
            tool_names=tool_names if tool_names is not None else [], context_budget=context_budget,
            auto_compaction=auto_compaction,
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
                    if not isinstance(existing, SessionEntry) or (
                        existing.message, existing.usage, existing.stop_reason
                    ) != (entry.message, entry.usage, entry.stop_reason):
                        raise SessionConflictError("条目 ID 已被不同内容使用")
                    return snapshot
            if expected_sequence is not None and expected_sequence != snapshot.last_sequence:
                raise SessionConflictError("预期序号已过期，请重新加载会话")
            entry_ids = {item.id for item in snapshot.entries}
            call_ids = {
                call.id for item in snapshot.entries
                if isinstance(item, SessionEntry) and isinstance(item.message, AssistantMessage)
                for call in item.message.tool_calls
            }
            pending = {call.id: call for call in snapshot.pending_tool_calls}
            _check_entry(entry, snapshot.last_sequence + 1, entry_ids, call_ids, pending, snapshot.entries)
            self._write_entry(session_id, entry)
            return self._read(session_id)[0]

    def _append_compaction(
        self, session_id: str, summary: HistorySummary, *, first_kept_entry_id: str,
        expected_sequence: int, entry_id: str | None, usage: Usage | None,
    ) -> SessionSnapshot:
        with self._locked(session_id):
            snapshot, _ = self._read(session_id)
            if snapshot.tail_issue is not None:
                raise SessionRecoveryRequired("会话尾行异常，请先显式修复")
            entry = CompactionEntry(
                type="compaction", id=entry_id if entry_id is not None else uuid.uuid4().hex,
                sequence=snapshot.last_sequence + 1, timestamp=datetime.now(timezone.utc),
                source_sequence=expected_sequence, first_kept_entry_id=first_kept_entry_id,
                summary=summary, usage=usage,
            )
            entry = CompactionEntry.model_validate_json(entry.model_dump_json())
            for existing in snapshot.entries:
                if existing.id == entry.id:
                    fields = {"sequence", "timestamp"}
                    if not isinstance(existing, CompactionEntry) or (
                        existing.model_dump(exclude=fields) != entry.model_dump(exclude=fields)
                    ):
                        raise SessionConflictError("条目 ID 已被不同内容使用")
                    return snapshot
            if expected_sequence != snapshot.last_sequence:
                raise SessionConflictError("摘要生成期间会话已变化，请重新加载并生成摘要")
            if snapshot.pending_tool_calls:
                raise SessionRecoveryRequired("存在未完成工具调用，不能压缩历史")
            _check_entry(
                entry, snapshot.last_sequence + 1, {item.id for item in snapshot.entries},
                set(), {call.id: call for call in snapshot.pending_tool_calls}, snapshot.entries,
            )
            self._write_entry(session_id, entry)
            return self._read(session_id)[0]

    def _append_model_change(
        self, session_id: str, model: str, *, expected_sequence: int, entry_id: str | None,
    ) -> SessionSnapshot:
        with self._locked(session_id):
            snapshot, _ = self._read(session_id)
            if snapshot.needs_recovery:
                raise SessionRecoveryRequired("会话需要修复或存在未完成工具调用，请先核查恢复")
            entry = ModelChangeEntry(
                type="model_change", id=entry_id if entry_id is not None else uuid.uuid4().hex,
                sequence=snapshot.last_sequence + 1, timestamp=datetime.now(timezone.utc), model=model,
            )
            for existing in snapshot.entries:
                if existing.id == entry.id:
                    if not isinstance(existing, ModelChangeEntry) or existing.model != entry.model:
                        raise SessionConflictError("条目 ID 已被不同内容使用")
                    return snapshot
            if expected_sequence != snapshot.last_sequence:
                raise SessionConflictError("预期序号已过期，请重新加载会话")
            if snapshot.model == model:
                return snapshot
            self._write_entry(session_id, entry)
            return self._read(session_id)[0]

    def _write_entry(self, session_id: str, entry: SessionRecord) -> None:
        with self._path(session_id).open("r+b") as stream:
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
