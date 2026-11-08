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
) -> int:
    renderer.info("输入任务开始对话，/new 新建会话，/exit 退出。")
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
        if text == "/new":
            try:
                replacement = await new_session()
            except (SessionError, OSError, ValueError) as error:
                renderer.error(f"新建会话失败，当前会话为 {session.session_id}。{error}")
                continue
            session = replacement
            renderer.info(f"[session] {session.session_id}")
            continue
        if text.startswith("/"):
            renderer.error("未知命令。可用命令为 /new 和 /exit，命令单独占一行。")
            continue
        # 输入文本保留原始缩进，strip 仅用于空白判断与命令识别。
        await run_turn(session, line, renderer)


async def interact(
    session: AgentSession, *, new_session: Callable[[], Awaitable[AgentSession]],
    renderer: TerminalRenderer,
) -> int:
    with closing(create_input(stdin=sys.stdin)) as terminal_input:
        prompt = PromptSession(input=terminal_input, output=create_output(stdout=sys.stderr))
        return await conversation_loop(
            session, new_session=new_session,
            read_line=lambda: prompt.prompt_async("timo> "), renderer=renderer,
        )
