"""为沙箱执行准备有界、过滤本地配置的临时项目副本。"""

import asyncio
import os
import stat
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from timocode.llm.schemas import DataModel
from timocode.security.workspace import Workspace


EXCLUDED_NAMES = frozenset({".git", ".venv", ".timocode", "__pycache__", ".pytest_cache"})
T = TypeVar("T")


class SnapshotLimits(DataModel):
    max_files: int = Field(default=2000, gt=0)
    max_total_bytes: int = Field(default=67_108_864, gt=0)
    max_entries: int = Field(default=10000, gt=0)
    max_depth: int = Field(default=20, ge=0)


class SnapshotLimitError(ValueError):
    """快照超出限制，调用方应缩小授权目录或调整限制。"""


@dataclass(frozen=True)
class SandboxSnapshot:
    root: Path
    file_count: int
    total_bytes: int
    skipped_entries: int


async def _complete_io(operation: Callable[[], T]) -> T:
    """取消时等待正在运行的线程收尾，避免删除仍在写入的目录。"""
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
        worker.exception()
        raise asyncio.CancelledError
    return worker.result()


def _copy_workspace(workspace: Workspace, destination: Path, limits: SnapshotLimits) -> SandboxSnapshot:
    destination.mkdir()
    queue = deque([(workspace.root, 0)])
    file_count = 0
    total_bytes = 0
    examined = 0
    skipped = 0
    while queue:
        directory, depth = queue.popleft()
        workspace.resolve_path(str(directory))
        with os.scandir(directory) as children:
            for child in children:
                examined += 1
                if examined > limits.max_entries:
                    raise SnapshotLimitError("快照目录条目数超过限制")
                name = child.name.casefold()
                if name in EXCLUDED_NAMES or name == ".env" or name.startswith(".env."):
                    skipped += 1
                    continue
                source = workspace.resolve_path(child.path)
                target = destination / source.relative_to(workspace.root)
                info = source.stat()
                if stat.S_ISDIR(info.st_mode):
                    if depth >= limits.max_depth:
                        raise SnapshotLimitError("快照目录深度超过限制")
                    target.mkdir()
                    queue.append((source, depth + 1))
                elif stat.S_ISREG(info.st_mode):
                    if file_count >= limits.max_files:
                        raise SnapshotLimitError("快照文件数超过限制")
                    if total_bytes + info.st_size > limits.max_total_bytes:
                        raise SnapshotLimitError("快照总字节数超过限制")
                    _, data = workspace.read_bytes(str(source))
                    if total_bytes + len(data) > limits.max_total_bytes:
                        raise SnapshotLimitError("快照总字节数超过限制")
                    target.write_bytes(data)
                    target.chmod(stat.S_IMODE(info.st_mode) & 0o777)
                    file_count += 1
                    total_bytes += len(data)
                else:
                    raise ValueError("快照仅支持普通文件和目录")
    return SandboxSnapshot(destination, file_count, total_bytes, skipped)


@asynccontextmanager
async def sandbox_snapshot(
    workspace: Workspace, *, limits: SnapshotLimits | None = None,
) -> AsyncIterator[SandboxSnapshot]:
    """在上下文存续期间提供快照，退出、失败或取消时删除临时副本。"""
    limits = SnapshotLimits() if limits is None else SnapshotLimits.model_validate_json(limits.model_dump_json())
    temporary = tempfile.TemporaryDirectory(prefix="timocode-sandbox-")
    try:
        temporary_root = Path(temporary.name).resolve(strict=True)
        if temporary_root.is_relative_to(workspace.root):
            raise ValueError("快照临时目录必须位于授权工作目录之外")
        async with workspace.mutation_lock:
            snapshot = await _complete_io(lambda: _copy_workspace(workspace, temporary_root / "project", limits))
        yield snapshot
    finally:
        await _complete_io(temporary.cleanup)
