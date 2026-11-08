"""文件工具共用的工作目录边界。"""

import asyncio
import os
import stat
from pathlib import Path


class Workspace:
    def __init__(self, root: str | Path, *, max_file_bytes: int = 1_048_576):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("工作目录必须是现有目录")
        if max_file_bytes < 1:
            raise ValueError("文件大小上限必须为正数")
        self.max_file_bytes = max_file_bytes
        self.mutation_lock = asyncio.Lock()

    def resolve_path(self, value: str, *, must_exist: bool = True) -> Path:
        if not value or "\x00" in value:
            raise ValueError("路径不能为空或包含空字符")
        path = Path(value)
        if os.name == "nt" and (bool(path.drive) != bool(path.root)):
            raise PermissionError("拒绝盘符相对路径和不明确的根路径")
        if os.name == "nt":
            for part in path.parts:
                if part in {path.anchor, ".", ".."}:
                    continue
                if ":" in part or part.endswith((".", " ")) or Path(part).is_reserved():
                    raise PermissionError("拒绝设备名、备用数据流或不明确的 Windows 路径")
        target = Path(os.path.abspath(self.root / path))
        try:
            relative = target.relative_to(self.root)
        except ValueError:
            raise PermissionError("路径越出授权工作目录") from None
        current = self.root
        for part in ("", *relative.parts):
            if part:
                current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                if must_exist:
                    raise
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise PermissionError("拒绝符号链接和重解析点")
            if current != target and not stat.S_ISDIR(info.st_mode):
                raise NotADirectoryError(str(current))
        return target

    def relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def read_bytes(self, value: str) -> tuple[Path, bytes]:
        path = self.resolve_path(value)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("目标必须是普通文件")
        with path.open("rb") as stream:
            data = stream.read(self.max_file_bytes + 1)
        if len(data) > self.max_file_bytes:
            raise ValueError(f"文件超过 {self.max_file_bytes} 字节限制")
        return path, data
