"""Typer 任务与交互入口，复用 AgentSession 和现有代码工具。"""

import asyncio
import sys
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from timocode.agent.context import ContextBudget
from timocode.agent.loop import DEFAULT_SYSTEM_PROMPT
from timocode.agent.session import SessionError, SessionStore
from timocode.agent.session_runtime import AgentSession
from timocode.interactive import interact, run_turn
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.security.permissions import PermissionManager, confirm_console
from timocode.security.workspace import Workspace
from timocode.terminal import TerminalRenderer
from timocode.tools.coding import create_coding_tools


app = typer.Typer(help="TimoCode 终端编程助手", pretty_exceptions_enable=False)


async def resume_recent(
    store: SessionStore, restore: Callable[[str], Awaitable[AgentSession]], renderer: TerminalRenderer,
) -> AgentSession | None:
    candidates = []
    for path in store.directory.glob("*.jsonl"):
        try:
            candidates.append((path.stat().st_mtime_ns, path.stem))
        except OSError:
            continue
    for _, session_id in sorted(candidates, reverse=True):
        try:
            session = await restore(session_id)
        except (SessionError, OSError, ValueError) as error:
            renderer.info(f"[resume skipped] {session_id}。{error}")
            continue
        renderer.info("[resume] 已自动恢复当前目录最近的可用会话。")
        return session
    return None


async def run_task(
    prompt: str | None, *, workspace: Path | None, docker_image: str | None,
    sessions_dir: Path, max_iterations: int, max_tokens: int | None,
    context_window: int | None, trace: bool, trace_dir: Path | None,
    renderer: TerminalRenderer, env_file: Path | None = None,
    summary_context_window: int = 32000, summary_output_tokens: int = 2048,
    auto_resume: bool = False,
) -> int:
    if prompt is not None and not prompt.strip():
        raise ValueError("任务内容不能为空")
    if prompt is None and not sys.stdin.isatty():
        raise ValueError("交互模式需要终端输入，请提供任务文本，例如 timo \"解释 Agent Loop\"")
    if docker_image is not None and workspace is None:
        raise ValueError("--docker-image 需要同时指定 --workspace")
    if trace_dir is not None and not trace:
        raise ValueError("--trace-dir 需要同时指定 --trace")
    budget = None if context_window is None else ContextBudget(context_window=context_window)
    summary_budget = ContextBudget(
        context_window=summary_context_window, reserve_output_tokens=summary_output_tokens,
    )
    root = Path.cwd() if workspace is None else workspace.resolve(strict=True)
    tools = [] if workspace is None else create_coding_tools(
        Workspace(root), PermissionManager(partial(confirm_console, output=sys.stderr)), docker_image=docker_image,
    )
    settings = load_settings() if env_file is None else load_settings(env_file)
    rules = DEFAULT_SYSTEM_PROMPT
    if workspace is not None:
        rules += "文件修改前先读取 sha256。收到 permission_denied 时停止对应操作并向用户说明。"
    store = SessionStore(sessions_dir)
    async with OpenAICompatibleProvider(settings) as provider:
        new_session = partial(
            AgentSession.create,
            provider, settings.model, root, store=store, tools=tools,
            system_prompt=rules, max_iterations=max_iterations, max_tokens=max_tokens,
            context_budget=budget, on_event=renderer,
            trace_directory=(trace_dir or Path(".timocode/traces")) if trace else None,
            trace_secrets=(settings.api_key.get_secret_value(),),
        )

        async def resume_session(session_id: str) -> AgentSession:
            restored = await AgentSession.resume(
                provider, session_id, store=store, tools=tools, expected_cwd=root,
                max_iterations=max_iterations, max_tokens=max_tokens, on_event=renderer,
                trace_directory=(trace_dir or Path(".timocode/traces")) if trace else None,
                trace_secrets=(settings.api_key.get_secret_value(),),
            )
            if budget is not None and budget != restored.snapshot.header.context_budget:
                raise ValueError("上下文预算与保存配置不一致，请使用原配置或省略 --context-window")
            return restored

        session = await resume_recent(store, resume_session, renderer) if auto_resume else None
        if session is None:
            session = await new_session()
        renderer.info(f"[session] {session.session_id}")
        renderer.info(f"[model] {session.model}")
        renderer.info(f"[workspace] {root}" if workspace is not None else "[tools] 未授权工作目录")
        if prompt is None:
            return await interact(
                session, new_session=new_session, resume_session=resume_session, renderer=renderer,
                summary_budget=summary_budget,
            )
        return await run_turn(session, prompt, renderer)


@app.command()
def run(
    prompt: Annotated[str | None, typer.Argument(help="自然语言任务，省略时进入交互并自动恢复最近会话")] = None,
    workspace: Annotated[Path | None, typer.Option(
        "--workspace", exists=True, file_okay=False, readable=True, help="授权代码工具访问的目录，默认当前目录",
    )] = None,
    docker_image: Annotated[str | None, typer.Option(help="用于隔离执行的本地 Linux Python 镜像")] = None,
    sessions_dir: Annotated[Path | None, typer.Option(help="会话保存目录，默认工作目录下 .timocode/sessions")] = None,
    max_iterations: Annotated[int, typer.Option(min=1, help="最大模型请求次数")] = 8,
    max_tokens: Annotated[int | None, typer.Option(min=1, help="单次模型输出 Token 上限")] = None,
    context_window: Annotated[int | None, typer.Option(min=1, help="启用上下文窗口预算")] = None,
    trace: Annotated[bool, typer.Option("--trace", help="保存结构化执行追踪")] = False,
    trace_dir: Annotated[Path | None, typer.Option(help="追踪保存目录，需要 --trace")] = None,
    env_file: Annotated[Path | None, typer.Option(
        "--env-file", exists=True, dir_okay=False, readable=True, help="模型配置文件，默认合并用户配置与当前目录 .env",
    )] = None,
    summary_context_window: Annotated[int, typer.Option(min=1, help="独立摘要请求的上下文窗口")] = 32000,
    summary_output_tokens: Annotated[int, typer.Option(min=1, help="独立摘要请求的输出预留")] = 2048,
) -> None:
    renderer = TerminalRenderer()
    try:
        workspace = Path.cwd() if workspace is None else workspace
        sessions_dir = workspace / ".timocode/sessions" if sessions_dir is None else sessions_dir
        code = asyncio.run(run_task(
            prompt, workspace=workspace, docker_image=docker_image,
            sessions_dir=sessions_dir, max_iterations=max_iterations, max_tokens=max_tokens,
            context_window=context_window, trace=trace, trace_dir=trace_dir, renderer=renderer,
            env_file=env_file,
            summary_context_window=summary_context_window, summary_output_tokens=summary_output_tokens,
            auto_resume=prompt is None,
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
