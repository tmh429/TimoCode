"""逐次批准的命令执行，支持超时、有界输出和进程树清理。"""

import json
import os
import shutil

from pydantic import Field, JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition
from timocode.security.permissions import PermissionManager, PermissionRequest
from timocode.security.execution import run_process
from timocode.security.workspace import Workspace
from timocode.tools.base import ToolResult


class ShellArguments(DataModel):
    command: str = Field(min_length=1, max_length=20000)
    cwd: str = "."
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_output_bytes: int = Field(default=16384, ge=1, le=131072)


class ShellTool:
    def __init__(self, workspace: Workspace, permissions: PermissionManager):
        self.workspace = workspace
        self.permissions = permissions
        if os.name == "nt":
            self.executable = shutil.which("pwsh") or shutil.which("powershell")
            self.backend = "PowerShell"
        else:
            self.executable = shutil.which("bash")
            self.backend = "Bash"
        if self.executable is None:
            raise ValueError("找不到命令执行后端")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description=f"经用户逐次确认执行 {self.backend} 命令。cwd 在授权目录内，命令本身拥有宿主用户权限。返回退出码、stdout、stderr、超时和截断状态。当前 Python 为 {os.sys.executable}。",
            parameters=ShellArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = ShellArguments.model_validate(arguments)
        cwd = self.workspace.resolve_path(args.cwd)
        if not cwd.is_dir():
            raise NotADirectoryError(str(cwd))
        await self.permissions.require(PermissionRequest("bash", {
            "command": args.command, "cwd": str(cwd), "shell": self.backend,
        }))
        cwd = self.workspace.resolve_path(args.cwd)
        if os.name == "nt":
            prefix = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $OutputEncoding=[Console]::OutputEncoding; $ErrorActionPreference='Stop'; $LASTEXITCODE=$null; "
            script = prefix + args.command + "\nif ($null -ne $LASTEXITCODE) { exit $LASTEXITCODE }"
            command = [self.executable, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script]
        else:
            command = [self.executable, "--noprofile", "--norc", "-c", args.command]
        result = await run_process(
            command, cwd=cwd, timeout_seconds=args.timeout_seconds,
            max_output_bytes=args.max_output_bytes,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
        )
        return ToolResult(
            content=json.dumps({"shell": self.backend, **result.as_dict()}, ensure_ascii=False),
            is_error=result.timed_out or result.exit_code != 0,
        )
