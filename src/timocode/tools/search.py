"""有界目录遍历、文件名搜索和文本搜索。"""

import fnmatch
import json
import os
from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition
from timocode.security.workspace import Workspace
from timocode.tools.base import ToolResult


SKIP_RECURSION = {".git", ".venv", ".timocode", "__pycache__", ".pytest_cache"}


class ListArguments(DataModel):
    path: str = "."
    recursive: bool = False
    max_depth: int = Field(default=3, ge=0, le=20)
    max_entries: int = Field(default=200, ge=1, le=2000)


class SearchArguments(DataModel):
    query: str = Field(min_length=1, max_length=200)
    path: str = "."
    mode: Literal["name", "content"] = "content"
    glob: str = "*"
    case_sensitive: bool = False
    max_results: int = Field(default=100, ge=1, le=500)


def _scan(workspace: Workspace, start: Path, depth: int, limit: int) -> tuple[list[tuple[Path, bool]], bool]:
    if not start.is_dir():
        raise NotADirectoryError(str(start))
    queue = deque([(start, 0)])
    entries = []
    examined = 0
    while queue:
        directory, level = queue.popleft()
        workspace.resolve_path(str(directory))
        with os.scandir(directory) as children:
            for child in children:
                if examined >= limit:
                    return entries, True
                examined += 1
                try:
                    path = workspace.resolve_path(child.path)
                    is_dir = path.is_dir()
                except (OSError, ValueError):
                    continue
                entries.append((path, is_dir))
                if is_dir and level < depth and path.name not in SKIP_RECURSION:
                    queue.append((path, level + 1))
    return entries, False


class ListFilesTool:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="list_files",
            description="列出授权目录结构。递归时跳过 .git、.venv、.timocode 和 Python 缓存目录，结果数量和深度受限。",
            parameters=ListArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = ListArguments.model_validate(arguments)
        root = self.workspace.resolve_path(args.path)
        entries, truncated = _scan(self.workspace, root, args.max_depth if args.recursive else 0, args.max_entries)
        result = [{"path": self.workspace.relative(path), "type": "directory" if is_dir else "file"} for path, is_dir in entries]
        result.sort(key=lambda entry: entry["path"])
        return ToolResult(content=json.dumps({"entries": result, "truncated": truncated}, ensure_ascii=False))


class SearchFilesTool:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_files",
            description="在授权目录搜索文件名或逐行搜索 UTF-8 文本中的字面字符串。支持 glob、大小写选项和结果上限，返回行号。",
            parameters=SearchArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = SearchArguments.model_validate(arguments)
        start = self.workspace.resolve_path(args.path)
        if start.is_dir():
            entries, truncated = _scan(self.workspace, start, 20, 5000)
        else:
            entries, truncated = [(start, False)], False
        matches = []
        scanned = 0
        skipped = 0
        needle = args.query if args.case_sensitive else args.query.casefold()
        for path, is_dir in entries:
            relative = self.workspace.relative(path)
            if is_dir or not (fnmatch.fnmatch(path.name, args.glob) or fnmatch.fnmatch(relative, args.glob)):
                continue
            scanned += 1
            if args.mode == "name":
                name = path.name if args.case_sensitive else path.name.casefold()
                if needle in name:
                    matches.append({"path": relative})
            else:
                try:
                    _, data = self.workspace.read_bytes(str(path))
                    text = data.decode("utf-8")
                    if "\x00" in text:
                        raise ValueError("二进制文件")
                except (OSError, ValueError):
                    skipped += 1
                    continue
                for line_number, line in enumerate(text.splitlines(), 1):
                    compared = line if args.case_sensitive else line.casefold()
                    if needle in compared:
                        matches.append({"path": relative, "line": line_number, "text": line[:500]})
                        if len(matches) >= args.max_results:
                            break
            if len(matches) >= args.max_results:
                truncated = True
                break
        return ToolResult(content=json.dumps({
            "matches": matches, "truncated": truncated, "scanned_files": scanned, "skipped_files": skipped,
        }, ensure_ascii=False))
