"""Typer 单次任务入口，复用 AgentSession 和现有代码工具。"""

import asyncio
import sys
from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from timocode.agent.context import ContextBudget
from timocode.agent.loop import DEFAULT_SYSTEM_PROMPT
from timocode.agent.session import SessionError, SessionStore
from timocode.agent.session_runtime import AgentSession
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.security.permissions import PermissionManager, confirm_console
from timocode.security.workspace import Workspace
from timocode.terminal import TerminalRenderer
from timocode.tools.coding import create_coding_tools


app = typer.Typer(help="TimoCode 终端编程助手", pretty_exceptions_enable=False)


async def run_task(
    prompt: str, *, workspace: Path | None, docker_image: str | None,
    sessions_dir: Path, max_iterations: int, max_tokens: int | None,
    context_window: int | None, trace: bool, trace_dir: Path | None,
    renderer: TerminalRenderer,
) -> int:
    if not prompt.strip():
        raise ValueError("任务内容不能为空")
    if docker_image is not None and workspace is None:
        raise ValueError("--docker-image 需要同时指定 --workspace")
    if trace_dir is not None and not trace:
        raise ValueError("--trace-dir 需要同时指定 --trace")
    budget = None if context_window is None else ContextBudget(context_window=context_window)
    root = Path.cwd() if workspace is None else workspace.resolve(strict=True)
    tools = [] if workspace is None else create_coding_tools(
        Workspace(root), PermissionManager(partial(confirm_console, output=sys.stderr)), docker_image=docker_image,
    )
    settings = load_settings()
    rules = DEFAULT_SYSTEM_PROMPT
    if workspace is not None:
        rules += "文件修改前先读取 sha256。收到 permission_denied 时停止对应操作并向用户说明。"
    async with OpenAICompatibleProvider(settings) as provider:
        session = await AgentSession.create(
            provider, settings.model, root, store=SessionStore(sessions_dir), tools=tools,
            system_prompt=rules, max_iterations=max_iterations, max_tokens=max_tokens,
            context_budget=budget, on_event=renderer,
            trace_directory=(trace_dir or Path(".timocode/traces")) if trace else None,
            trace_secrets=(settings.api_key.get_secret_value(),),
        )
        renderer.info(f"[session] {session.session_id}")
        renderer.info(f"[model] {settings.model}")
        renderer.info(f"[workspace] {root}" if workspace is not None else "[tools] 未授权工作目录")
        try:
            result = await session.run(prompt)
        finally:
            renderer.end_text()
            if session.trace_path is not None:
                renderer.info(f"[trace] {session.trace_path}")
            if session.trace_error is not None:
                renderer.error(session.trace_error)
    renderer.finish(result)
    return 0 if result.status == "completed" else 1


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="交给模型的自然语言任务")],
    workspace: Annotated[Path | None, typer.Option(
        "--workspace", exists=True, file_okay=False, readable=True, help="授权代码工具访问的目录",
    )] = None,
    docker_image: Annotated[str | None, typer.Option(help="用于隔离执行的本地 Linux Python 镜像")] = None,
    sessions_dir: Annotated[Path, typer.Option(help="会话保存目录")] = Path(".timocode/sessions"),
    max_iterations: Annotated[int, typer.Option(min=1, help="最大模型请求次数")] = 8,
    max_tokens: Annotated[int | None, typer.Option(min=1, help="单次模型输出 Token 上限")] = None,
    context_window: Annotated[int | None, typer.Option(min=1, help="启用上下文窗口预算")] = None,
    trace: Annotated[bool, typer.Option("--trace", help="保存结构化执行追踪")] = False,
    trace_dir: Annotated[Path | None, typer.Option(help="追踪保存目录，需要 --trace")] = None,
) -> None:
    renderer = TerminalRenderer()
    try:
        code = asyncio.run(run_task(
            prompt, workspace=workspace, docker_image=docker_image,
            sessions_dir=sessions_dir, max_iterations=max_iterations, max_tokens=max_tokens,
            context_window=context_window, trace=trace, trace_dir=trace_dir, renderer=renderer,
        ))
    except (KeyboardInterrupt, asyncio.CancelledError):
        renderer.error("运行已取消。")
        code = 130
    except (ValueError, OSError, SessionError) as error:
        renderer.error(str(error))
        code = 2
    raise typer.Exit(code)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
