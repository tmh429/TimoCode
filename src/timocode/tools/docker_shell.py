"""在受限 Linux 容器的临时项目副本中执行经批准的命令。"""

import asyncio
import json
import re
import shutil
import uuid
from pathlib import Path

from pydantic import JsonValue

from timocode.llm.schemas import ToolDefinition
from timocode.security.execution import ProcessResult, complete_task, run_process
from timocode.security.permissions import PermissionManager, PermissionRequest
from timocode.security.snapshot import sandbox_snapshot
from timocode.security.workspace import Workspace
from timocode.tools.base import ToolResult
from timocode.tools.shell import ShellArguments


BOOTSTRAP = """import os, shutil, signal, subprocess, sys
signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
signal.setitimer(signal.ITIMER_REAL, float(sys.argv[3]))
shutil.copytree('/input', '/work/project')
os.chdir(os.path.join('/work/project', sys.argv[1]))
result = subprocess.run(['/bin/sh', '-c', sys.argv[2]])
sys.exit(result.returncode if result.returncode >= 0 else 128 - result.returncode)
"""


class DockerShellTool:
    def __init__(self, workspace: Workspace, permissions: PermissionManager, *, image: str):
        if not re.fullmatch(r"[a-z0-9][a-zA-Z0-9._:/@-]*", image):
            raise ValueError("Docker 镜像引用无效")
        self.workspace = workspace
        self.permissions = permissions
        self.image = image
        self.executable = shutil.which("docker")
        if self.executable is None:
            raise ValueError("找不到 Docker CLI")

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="docker_shell",
            description=(
                f"逐次批准后在 Linux Docker 镜像 {self.image} 的 /bin/sh 中执行命令。"
                "项目使用过滤 .env、Git、虚拟环境和会话目录后的临时副本。"
                "容器采用非 root、禁网、只读根文件系统和资源限制。"
                "cwd 为项目相对目录，Python 命令使用容器内 python。"
                "生成文件随容器退出清理，持久项目修改使用 write_file/edit_file。"
            ),
            parameters=ShellArguments.model_json_schema(),
        )

    async def _cli(self, arguments: list[str], *, timeout: float = 10.0, limit: int = 4096) -> ProcessResult:
        return await run_process(
            [self.executable, *arguments], cwd=self.workspace.root,
            timeout_seconds=timeout, max_output_bytes=limit,
        )

    def _create_arguments(self, name: str, root: Path, image_id: str, args: ShellArguments, cwd: str) -> list[str]:
        if "," in str(root):
            raise ValueError("Docker 快照挂载路径包含不支持的逗号")
        command = [
            "create", "--name", name, "--pull=never", "--init",
            "--label", "timocode.sandbox=true", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
            "--cpus=1", "--memory=512m", "--memory-swap=512m", "--pids-limit=64",
            "--ipc=private", "--shm-size=16m", "--log-driver=none", "--restart=no", "--no-healthcheck",
            "--mount", f"type=bind,source={root},target=/input,readonly",
            "--tmpfs", "/work:rw,nosuid,nodev,size=128m,mode=1777",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m,mode=1777",
            "--env", "HOME=/tmp", "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
        ]
        # 覆盖 Docker 客户端配置可能自动注入的代理环境变量。
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY"):
            command.extend(["--env", key + "=", "--env", key.lower() + "="])
        return command + [
            "--entrypoint=python", image_id, "-I", "-c", BOOTSTRAP,
            cwd, args.command, str(args.timeout_seconds),
        ]

    async def _remove(self, name: str) -> None:
        if not re.fullmatch(r"timocode-[0-9a-f]{32}", name):
            raise ValueError("拒绝回收无效的沙箱容器名")
        result = await self._cli(["rm", "--force", "--volumes", name])
        if result.timed_out or (result.exit_code != 0 and "no such container" not in result.stderr.casefold()):
            raise RuntimeError(f"无法确认容器已回收，请核查 {name}。{result.stderr}")

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        args = ShellArguments.model_validate(arguments)
        cwd = self.workspace.resolve_path(args.cwd)
        if not cwd.is_dir():
            raise NotADirectoryError(str(cwd))
        inspected = await self._cli([
            "image", "inspect", "--format",
            '{{.Os}} {{.Id}} {{json (index .Config "Volumes")}}', self.image,
        ])
        if inspected.exit_code != 0 or inspected.timed_out:
            raise RuntimeError(f"无法检查本地 Docker 镜像，请确认 Linux 引擎和镜像已就绪。{inspected.stderr}")
        parts = inspected.stdout.strip().split(maxsplit=2)
        if len(parts) != 3 or parts[0] != "linux" or not re.fullmatch(r"sha256:[0-9a-f]{64}", parts[1]):
            raise ValueError("沙箱需要有效的本地 Linux 镜像 ID")
        if json.loads(parts[2]):
            raise ValueError("沙箱镜像不能声明额外 VOLUME")
        image_id = parts[1]
        await self.permissions.require(PermissionRequest("docker_shell", {
            "command": args.command, "cwd": str(cwd), "image": self.image,
            "image_id": image_id, "network": "none", "memory_mb": 512, "pids_limit": 64,
        }))
        cwd = self.workspace.resolve_path(args.cwd)
        relative = cwd.relative_to(self.workspace.root).as_posix()
        name = "timocode-" + uuid.uuid4().hex
        deadline = asyncio.get_running_loop().time() + args.timeout_seconds
        async with sandbox_snapshot(self.workspace) as snapshot:
            if not (snapshot.root / relative).is_dir():
                raise ValueError("cwd 已被快照过滤")
            try:
                remaining = max(0.001, deadline - asyncio.get_running_loop().time())
                created = await complete_task(asyncio.create_task(self._cli(
                    self._create_arguments(name, snapshot.root, image_id, args, relative),
                    timeout=min(10.0, remaining), limit=args.max_output_bytes,
                )))
                if created.timed_out or created.exit_code != 0:
                    result = created
                else:
                    remaining = max(0.001, deadline - asyncio.get_running_loop().time())
                    result = await self._cli(
                        ["start", "--attach", name], timeout=remaining, limit=args.max_output_bytes,
                    )
            finally:
                await complete_task(asyncio.create_task(self._remove(name)))
        return ToolResult(
            content=json.dumps({
                "shell": "Docker /bin/sh", "image_id": image_id,
                "container": name, **result.as_dict(),
            }, ensure_ascii=False),
            is_error=result.timed_out or result.exit_code != 0,
        )
