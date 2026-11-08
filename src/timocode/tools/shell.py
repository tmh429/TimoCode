"""逐次批准的命令执行，支持超时、有界输出和进程树清理。"""

import asyncio
import json
import os
import shutil
import signal

from pydantic import Field, JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition
from timocode.security.permissions import PermissionManager, PermissionRequest
from timocode.security.processes import WindowsJob
from timocode.security.workspace import Workspace
from timocode.tools.base import ToolResult


class ShellArguments(DataModel):
    command: str = Field(min_length=1, max_length=20000)
    cwd: str = "."
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_output_bytes: int = Field(default=16384, ge=1, le=131072)


async def _capture(stream: asyncio.StreamReader, limit: int) -> tuple[str, bool]:
    kept = bytearray()
    size = 0
    while chunk := await stream.read(65536):
        size += len(chunk)
        kept.extend(chunk[:max(0, limit - len(kept))])
    return kept.decode("utf-8", errors="replace"), size > limit


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
        job = WindowsJob() if os.name == "nt" else None
        process = None
        readers = []
        timed_out = False
        try:
            spawning = asyncio.create_task(asyncio.create_subprocess_exec(
                *command, cwd=cwd, stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
                creationflags=0x4 if os.name == "nt" else 0,
            ))
            try:
                process = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                process = await spawning
                raise
            if job is not None:
                job.attach(process.pid)
            readers = [
                asyncio.create_task(_capture(process.stdout, args.max_output_bytes)),
                asyncio.create_task(_capture(process.stderr, args.max_output_bytes)),
            ]
            try:
                async with asyncio.timeout(args.timeout_seconds):
                    # shield 保留输出读取任务，超时后仍可收集已截断的输出。
                    await asyncio.shield(asyncio.gather(*readers))
                    await process.wait()
            except TimeoutError:
                timed_out = True
                await self._terminate(process, job)
            stdout, stderr = await asyncio.gather(*readers)
            return ToolResult(content=json.dumps({
                "shell": self.backend, "exit_code": process.returncode,
                "stdout": stdout[0], "stderr": stderr[0],
                "stdout_truncated": stdout[1], "stderr_truncated": stderr[1],
                "timed_out": timed_out,
            }, ensure_ascii=False), is_error=timed_out or process.returncode != 0)
        finally:
            try:
                if process is not None:
                    await self._terminate(process, job)
            finally:
                if job is not None:
                    job.close()
                if readers:
                    await asyncio.gather(*readers, return_exceptions=True)

    async def _terminate(self, process: asyncio.subprocess.Process, job: WindowsJob | None) -> None:
        if job is not None:
            job.terminate()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()
