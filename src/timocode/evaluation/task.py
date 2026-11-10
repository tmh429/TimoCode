"""声明式评测任务。工作区只包含初始文件，验收脚本由评测器持有。"""

from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Iterator, Literal, Self

from pydantic import Field, field_validator, model_validator

from timocode.llm.schemas import DataModel
from timocode.security.workspace import Workspace


def _relative_path(value: str) -> str:
    """任务路径采用可移植的 POSIX 相对路径。"""
    parts = value.split("/")
    if (
        not value or "\\" in value or "\x00" in value
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in parts)
        or any(any(ord(char) < 32 or char in '<>:"|?*' for char in part)
               or part.endswith((".", " "))
               or PureWindowsPath(part).is_reserved() for part in parts)
    ):
        raise ValueError("任务文件必须使用明确的可移植相对路径")
    return value


class TaskLimits(DataModel):
    max_iterations: int = Field(default=20, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)
    max_total_tokens: int = Field(default=32000, gt=0)
    verification_timeout_seconds: int = Field(default=30, gt=0)


class EvaluationTask(DataModel):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")
    category: Literal["retrieval", "code_change", "testing", "bug_fix", "multi_file"]
    instruction: str = Field(min_length=1)
    initial_files: dict[str, str] = Field(min_length=1)
    verifier: str = Field(min_length=1)
    limits: TaskLimits = Field(default_factory=TaskLimits)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("schema_version 必须是整数")
        return value

    @field_validator("instruction")
    @classmethod
    def nonblank_instruction(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("任务指令不能为空白")
        return value

    @field_validator("verifier")
    @classmethod
    def verifier_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        names = set()
        for name in self.initial_files:
            normalized = _relative_path(name).casefold()
            if normalized in names:
                raise ValueError("初始文件存在大小写冲突")
            names.add(normalized)
        for name in names:
            parents = Path(name).parents
            if any(parent.as_posix() in names for parent in parents):
                raise ValueError("初始文件存在文件与目录冲突")
        return self


def load_task(path: str | Path) -> EvaluationTask:
    """读取可信的本地任务包，并验证外部验收脚本存在。"""
    manifest = Path(path).resolve(strict=True)
    task = EvaluationTask.model_validate_json(manifest.read_text(encoding="utf-8"))
    verifier = Workspace(manifest.parent).resolve_path(task.verifier)
    if not verifier.is_file() or verifier.suffix != ".py":
        raise ValueError("验收入口必须是现有 Python 文件")
    return task


@contextmanager
def task_workspace(task: EvaluationTask) -> Iterator[Path]:
    """每次从声明的初始文件创建独立临时目录，退出时清理。"""
    # 重新验证，兼容调用者修改过的 Pydantic 对象。
    validated = EvaluationTask.model_validate(task.model_dump())
    with TemporaryDirectory(prefix=f"timo-eval-{validated.id}-") as directory:
        root = Path(directory)
        workspace = Workspace(root)
        for name, content in validated.initial_files.items():
            target = workspace.resolve_path(name, must_exist=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="\n")
        yield root
