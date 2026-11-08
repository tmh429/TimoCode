"""有界读取、摘要检查与原子文件修改。"""

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import Field, JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition
from timocode.security.permissions import PermissionManager, PermissionRequest
from timocode.security.workspace import Workspace
from timocode.tools.base import ToolResult


Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class ReadArguments(DataModel):
    path: str = Field(min_length=1)
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=200, ge=1, le=2000)
    max_chars: int = Field(default=20000, ge=1, le=100000)


class WriteArguments(DataModel):
    path: str = Field(min_length=1)
    content: str
    expected_sha256: Digest | None = None


class EditArguments(DataModel):
    path: str = Field(min_length=1)
    old_text: str = Field(min_length=1)
    new_text: str
    expected_sha256: Digest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_target(workspace: Workspace, value: str, expected: str | None) -> Path:
    path = workspace.resolve_path(value, must_exist=False)
    if path.exists():
        _, current = workspace.read_bytes(value)
        if expected is None or _digest(current) != expected:
            raise ValueError("修改冲突，请先 read_file 并提供当前 expected_sha256")
    elif expected is not None:
        raise ValueError("目标已不存在，修改前检查失败")
    return path


async def _write(
    workspace: Workspace,
    permissions: PermissionManager,
    value: str,
    content: str,
    expected: str | None,
    tool: str,
) -> ToolResult:
    data = content.encode("utf-8")
    if len(data) > workspace.max_file_bytes:
        raise ValueError("写入内容超过文件大小限制")
    path = _check_target(workspace, value, expected)
    await permissions.require(PermissionRequest(tool, {
        "path": str(path), "bytes": len(data), "sha256": _digest(data),
        "content_preview": content[:4000], "preview_truncated": int(len(content) > 4000),
    }))
    async with workspace.mutation_lock:
        path = _check_target(workspace, value, expected)
        path.parent.mkdir(parents=True, exist_ok=True)
        path = workspace.resolve_path(value, must_exist=False)
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
        fd, temporary_name = tempfile.mkstemp(prefix=".timo-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if mode is not None:
                os.chmod(temporary, mode)
            _check_target(workspace, value, expected)
            if expected is None:
                os.link(temporary, path)
            else:
                os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return ToolResult(content=json.dumps({
        "path": workspace.relative(path), "bytes": len(data), "sha256": _digest(data),
    }, ensure_ascii=False))


class ReadFileTool:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_file",
            description="读取授权目录内 UTF-8 文件的指定行，返回内容和修改前检查所需的 sha256。",
            parameters=ReadArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = ReadArguments.model_validate(arguments)
        path, data = self.workspace.read_bytes(args.path)
        lines = data.decode("utf-8").splitlines(keepends=True)
        selected = "".join(lines[args.start_line - 1:args.start_line - 1 + args.max_lines])
        truncated = len(selected) > args.max_chars or args.start_line - 1 + args.max_lines < len(lines)
        return ToolResult(content=json.dumps({
            "path": self.workspace.relative(path), "sha256": _digest(data),
            "total_lines": len(lines), "start_line": args.start_line,
            "content": selected[:args.max_chars], "truncated": truncated,
        }, ensure_ascii=False))


class WriteFileTool:
    def __init__(self, workspace: Workspace, permissions: PermissionManager):
        self.workspace = workspace
        self.permissions = permissions

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="write_file",
            description="创建 UTF-8 文件或覆盖文件，需要用户批准。覆盖必须提供 read_file 返回的 expected_sha256，创建新文件时省略。",
            parameters=WriteArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = WriteArguments.model_validate(arguments)
        return await _write(self.workspace, self.permissions, args.path, args.content, args.expected_sha256, "write_file")


class EditFileTool:
    def __init__(self, workspace: Workspace, permissions: PermissionManager):
        self.workspace = workspace
        self.permissions = permissions

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="edit_file",
            description="精确替换文件中唯一出现的 old_text，需要 read_file 返回的 expected_sha256 和用户批准。重复匹配会拒绝修改。",
            parameters=EditArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = EditArguments.model_validate(arguments)
        _, data = self.workspace.read_bytes(args.path)
        if _digest(data) != args.expected_sha256:
            raise ValueError("文件已变化，请重新读取后编辑")
        text = data.decode("utf-8")
        if text.count(args.old_text) != 1:
            raise ValueError("old_text 必须在文件中唯一匹配")
        updated = text.replace(args.old_text, args.new_text, 1)
        return await _write(self.workspace, self.permissions, args.path, updated, args.expected_sha256, "edit_file")
