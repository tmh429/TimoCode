"""单任务评测，复用会话持久化、Docker 隔离和安全工作区快照。"""

import asyncio
from contextlib import aclosing
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Literal
import uuid

from pydantic import Field, JsonValue

from timocode.agent.loop import DEFAULT_SYSTEM_PROMPT
from timocode.agent.session import SessionStore
from timocode.agent.session_runtime import AgentSession
from timocode.agent.state import RunResult
from timocode.evaluation.metrics import RunMetrics, ToolMetrics, measure_run, read_tool_metrics
from timocode.evaluation.task import EvaluationTask, load_task, task_workspace
from timocode.llm.base import LLMProvider, ProviderError, ResponseComplete
from timocode.llm.schemas import DataModel, LLMRequest, Usage
from timocode.security.permissions import PermissionManager, PermissionRequest
from timocode.security.snapshot import sandbox_snapshot
from timocode.security.workspace import Workspace
from timocode.tools.coding import create_coding_tools


EvaluationStatus = Literal[
    "passed", "failed", "agent_error", "task_timeout", "token_limit",
    "usage_unavailable", "verification_timeout", "verification_error",
    "runner_error", "cancelled",
]


class EvaluationResult(DataModel):
    schema_version: Literal[1] = 1
    run_id: str
    task_id: str
    model: str
    docker_image: str
    image_id: str | None = None
    started_at: str
    status: EvaluationStatus = "runner_error"
    elapsed_seconds: float = Field(default=0.0, ge=0)
    session_id: str | None = None
    iterations: int = Field(default=0, ge=0)
    usage: list[Usage] = Field(default_factory=list)
    agent_result: RunResult | None = None
    verification: dict[str, JsonValue] | None = None
    error: str | None = None
    artifact_error: str | None = None
    artifact_directory: str
    metrics: RunMetrics | None = None


class _BudgetProvider:
    """在完整响应边界累计服务端用量，超限响应不进入工具调度。"""

    def __init__(self, provider: LLMProvider, maximum: int):
        self.provider = provider
        self.maximum = maximum
        self.total = 0
        self.iterations = 0
        self.usage: list[Usage] = []
        self.failure: EvaluationStatus | None = None

    async def stream(self, request: LLMRequest):
        remaining = self.maximum - self.total
        if remaining <= 0:
            self.failure = "token_limit"
            raise ProviderError("已达到累计 Token 上限")
        request = request.model_copy(deep=True)
        request.max_tokens = min(request.max_tokens or remaining, remaining)
        self.iterations += 1
        async with aclosing(self.provider.stream(request)) as events:
            async for event in events:
                if isinstance(event, ResponseComplete):
                    usage = event.response.usage.model_copy(deep=True)
                    self.usage.append(usage)
                    if usage.input_tokens is None or usage.output_tokens is None:
                        self.failure = "usage_unavailable"
                        raise ProviderError("服务端未提供完整 usage，无法核查累计 Token 预算")
                    self.total += usage.input_tokens + usage.output_tokens
                    if self.total > self.maximum or (
                        self.total == self.maximum and event.response.message.tool_calls
                    ):
                        self.failure = "token_limit"
                        raise ProviderError("已达到累计 Token 上限，停止后续工具执行")
                yield event


async def _approve_evaluation(request: PermissionRequest) -> bool:
    # 自动批准范围仅限 Runner 创建的临时工作区与现有 Docker 工具。
    return request.tool in {"write_file", "edit_file", "docker_shell"}


def _save_json(path: Path, value: DataModel) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(value.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_task_inputs(manifest: str | Path) -> tuple[EvaluationTask, bytes]:
    task = load_task(manifest)
    if task.limits.verification_timeout_seconds > 300:
        raise ValueError("当前 Docker 验收后端支持最长 300 秒的验收超时")
    _, verifier_bytes = Workspace(Path(manifest).resolve().parent).read_bytes(task.verifier)
    verifier_bytes.decode("utf-8")
    return task, verifier_bytes


def _verification_command(verifier_source: str, run_id: str) -> tuple[str, str]:
    completion_marker = f"TIMOCODE_VERIFIER_FINISHED_{run_id}"
    # 完成标记用于区分验收失败和 Docker 创建、启动失败。
    verification_program = (
        "try:\n"
        f"    exec(compile({verifier_source!r}, '<verifier>', 'exec'), globals())\n"
        "finally:\n"
        f"    print('\\n' + {completion_marker!r}, flush=True)\n"
    )
    # 在模型执行前固定验收程序，Agent 只能访问独立候选工作区。
    command = "python -I -c " + shlex.quote(verification_program) + " ."
    if len(command) > 20000:
        raise ValueError("当前验收脚本超出 Docker 命令长度上限")
    return command, completion_marker


async def run_evaluation(
    manifest: str | Path, provider: LLMProvider, model: str, *,
    output_directory: str | Path = ".timocode/evaluations",
    docker_image: str = "python:3.12-slim", trace_secrets: tuple[str, ...] = (),
) -> EvaluationResult:
    """独立运行一次任务。调用方管理 Provider 的生命周期。"""
    if not model.strip():
        raise ValueError("模型名称不能为空")
    task, verifier_bytes = _load_task_inputs(manifest)
    run_id = uuid.uuid4().hex
    command, completion_marker = _verification_command(verifier_bytes.decode("utf-8"), run_id)
    started = monotonic()
    artifacts = Path(output_directory).resolve() / f"{task.id}-{run_id}"
    artifacts.mkdir(parents=True, exist_ok=False)
    result = EvaluationResult(
        run_id=run_id, task_id=task.id, model=model, docker_image=docker_image,
        started_at=datetime.now(timezone.utc).isoformat(), artifact_directory=str(artifacts),
    )
    budget = _BudgetProvider(provider, task.limits.max_total_tokens)
    session = None
    try:
        task_directory = artifacts / "task"
        task_directory.mkdir()
        # 归一化验收路径，使嵌套入口也能作为自包含任务快照重新加载。
        saved_task = task.model_copy(update={"verifier": "verify.py"})
        _save_json(task_directory / "task.json", saved_task)
        (task_directory / "verify.py").write_bytes(verifier_bytes)
        with task_workspace(task) as root:
            workspace = Workspace(root)
            try:
                async with asyncio.timeout(task.limits.timeout_seconds):
                    tools = create_coding_tools(
                        workspace, PermissionManager(_approve_evaluation), docker_image=docker_image,
                    )
                    docker = tools[-1]
                    preflight = await docker.execute({
                        "command": "python -I -c 'pass'", "timeout_seconds": 15.0,
                    })
                    environment = json.loads(preflight.content)
                    if preflight.is_error:
                        raise RuntimeError("Docker 环境预检失败。" + preflight.content)
                    result.image_id = environment["image_id"]
                    docker.image = result.image_id
                    session = await AgentSession.create(
                        budget, model, root, store=SessionStore(artifacts / "sessions"), tools=tools,
                        system_prompt=DEFAULT_SYSTEM_PROMPT + (
                            "你正在独立临时项目中完成评测任务。文件修改前先读取 sha256。"
                            "使用 write_file/edit_file 持久修改代码。docker_shell 中的修改随容器退出清理。"
                            "无需请求人工确认。完成任务后简要说明修改。"
                        ),
                        max_iterations=task.limits.max_iterations,
                        max_tokens=min(4096, task.limits.max_total_tokens),
                        trace_directory=artifacts / "traces", trace_content=True,
                        trace_secrets=trace_secrets,
                    )
                    result.session_id = session.session_id
                    result.agent_result = await session.run(task.instruction)
                    if budget.failure is not None:
                        result.status = budget.failure
                        result.error = result.agent_result.error
                    elif result.agent_result.status != "completed":
                        result.status = "agent_error"
                        result.error = result.agent_result.error
                    else:
                        result.status = "verification_error"
                        verification = await docker.execute({
                            "command": command,
                            "timeout_seconds": float(task.limits.verification_timeout_seconds),
                        })
                        result.verification = json.loads(verification.content)
                        if result.verification["timed_out"] or result.verification["exit_code"] == 124:
                            result.status = "verification_timeout"
                        elif completion_marker not in result.verification["stdout"].splitlines():
                            result.error = "验收程序未确认完成，请检查 Docker 状态和截断输出"
                        elif result.verification["exit_code"] == 0:
                            result.status = "passed"
                        elif result.verification["exit_code"] == 1:
                            result.status = "failed"
                        else:
                            result.error = "验收进程异常退出，请检查 verification 输出"
            except TimeoutError:
                result.status = "task_timeout"
                result.error = "已达到任务总耗时上限"
            finally:
                # 只从已有的有界安全快照导出文件，避免跟随候选路径中的链接。
                try:
                    async with sandbox_snapshot(workspace) as snapshot:
                        shutil.copytree(snapshot.root, artifacts / "workspace")
                except Exception as error:
                    result.artifact_error = f"工作区导出失败（{type(error).__name__}）。{error}"
    except asyncio.CancelledError:
        result.status = "cancelled"
        result.error = "运行已取消"
        raise
    except Exception as error:
        # 验收阶段异常与执行基础设施异常分开记录。
        if result.status != "verification_error":
            result.status = "runner_error"
        result.error = f"{type(error).__name__}。{error}"
    finally:
        result.iterations = budget.iterations
        result.usage = budget.usage
        if session is not None and session.trace_error:
            result.artifact_error = "。".join(filter(None, [result.artifact_error, session.trace_error]))
        result.elapsed_seconds = monotonic() - started
        tools = ToolMetrics() if budget.iterations == 0 else None
        if session is not None and session.trace_path is not None:
            tools = read_tool_metrics(session.trace_path, session.session_id)
        result.metrics = measure_run(result, tools)
        _save_json(artifacts / "result.json", result)
    return result
