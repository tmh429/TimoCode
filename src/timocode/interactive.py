"""连续对话循环及单轮执行，输入与模型请求顺序进行。"""

import sys
from collections.abc import Awaitable, Callable
from contextlib import closing

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_input
from prompt_toolkit.output import create_output

from timocode.agent.session import SessionError
from timocode.agent.session_runtime import AgentSession
from timocode.terminal import TerminalRenderer


async def run_turn(session: AgentSession, prompt: str, renderer: TerminalRenderer) -> int:
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


async def conversation_loop(
    session: AgentSession, *, new_session: Callable[[], Awaitable[AgentSession]],
    read_line: Callable[[], Awaitable[str]], renderer: TerminalRenderer,
    resume_session: Callable[[str], Awaitable[AgentSession]] | None = None,
) -> int:
    renderer.info("输入任务开始对话，/new 新建会话，/resume <会话ID> 恢复会话，/model [名称] 查看或切换模型，/status 查看状态，/exit 退出。")
    while True:
        try:
            line = await read_line()
        except EOFError:
            return 0
        except KeyboardInterrupt:
            renderer.info("输入已取消。")
            continue
        text = line.strip()
        if not text:
            continue
        if text == "/exit":
            return 0
        if text == "/status":
            try:
                renderer.show_status(session)
            except (SessionError, OSError, ValueError) as error:
                renderer.error(f"状态读取失败。{error}")
            continue
        if text == "/new":
            try:
                replacement = await new_session()
            except (SessionError, OSError, ValueError) as error:
                renderer.error(f"新建会话失败，当前会话为 {session.session_id}。{error}")
                continue
            session = replacement
            renderer.info(f"[session] {session.session_id}")
            renderer.info(f"[model] {session.model}")
            continue
        parts = text.split()
        if parts[0] == "/model":
            if len(parts) > 2:
                renderer.error("用法为 /model 或 /model <模型名称>。")
                continue
            if len(parts) == 2:
                try:
                    await session.set_model(parts[1])
                except (SessionError, OSError, ValueError) as error:
                    renderer.error(f"模型切换失败，当前模型为 {session.model}。{error}")
                    continue
            renderer.info(f"[model] {session.model}")
            continue
        if parts[0] == "/resume":
            if len(parts) != 2:
                renderer.error("用法为 /resume <会话ID>，会话 ID 可从 [session] 输出获取。")
                continue
            if resume_session is None:
                renderer.error("当前交互入口未配置会话恢复。")
                continue
            try:
                replacement = await resume_session(parts[1])
            except (SessionError, OSError, ValueError) as error:
                renderer.error(f"恢复失败，当前会话为 {session.session_id}。{error}")
                continue
            session = replacement
            renderer.info(f"[session] {session.session_id}")
            renderer.info(f"[model] {session.model}")
            continue
        if text.startswith("/"):
            renderer.error("未知命令。可用命令为 /new、/resume <会话ID>、/model [名称]、/status 和 /exit，命令单独占一行。")
            continue
        # 输入文本保留原始缩进，strip 仅用于空白判断与命令识别。
        await run_turn(session, line, renderer)


async def interact(
    session: AgentSession, *, new_session: Callable[[], Awaitable[AgentSession]],
    renderer: TerminalRenderer,
    resume_session: Callable[[str], Awaitable[AgentSession]] | None = None,
) -> int:
    with closing(create_input(stdin=sys.stdin)) as terminal_input:
        prompt = PromptSession(input=terminal_input, output=create_output(stdout=sys.stderr))
        return await conversation_loop(
            session, new_session=new_session,
            read_line=lambda: prompt.prompt_async("timo> "), renderer=renderer,
            resume_session=resume_session,
        )
