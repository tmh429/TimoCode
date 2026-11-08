"""基础运行入口，可选择加法演示或指定工作目录的 Coding Tools。"""

import argparse
import asyncio
import sys
from pathlib import Path

from timocode.agent.loop import Agent
from timocode.agent.state import AgentEvent
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.tools.demo import AddTool
from timocode.tools.coding import create_coding_tools
from timocode.security.permissions import PermissionManager, confirm_console
from timocode.security.workspace import Workspace


async def render(event: AgentEvent) -> None:
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_start":
        print(f"\n[tool] {event.tool_call.name}", flush=True)
    elif event.type == "tool_end":
        state = "error" if event.message.is_error else "ok"
        print(f"[tool {state}] {event.message.content}", flush=True)


async def run_demo(prompt: str, max_iterations: int, max_tokens: int | None, workspace: Path | None = None) -> int:
    settings = load_settings()
    tools = [AddTool()] if workspace is None else create_coding_tools(
        Workspace(workspace), PermissionManager(confirm_console),
    )
    prompt_rules = "你是编程助手，请依据工具的实际结果回答。"
    if workspace is not None:
        prompt_rules += "文件修改前先读取 sha256。收到 permission_denied 时停止对应操作并向用户说明。"
    async with OpenAICompatibleProvider(settings) as provider:
        agent = Agent(
            provider,
            settings.model,
            tools=tools,
            system_prompt=prompt_rules,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            on_event=render,
        )
        result = await agent.run(prompt)
    print(f"\n[{result.status}] model requests={result.iterations}")
    if result.error:
        print(result.error, file=sys.stderr)
    return 0 if result.status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="TimoCode 工具调用演示")
    parser.add_argument("prompt", help="交给模型的自然语言任务")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--workspace", type=Path, help="启用 Coding Tools 并授权指定工作目录")
    args = parser.parse_args()
    try:
        return asyncio.run(run_demo(args.prompt, args.max_iterations, args.max_tokens, args.workspace))
    except KeyboardInterrupt:
        print("\n运行已取消。", file=sys.stderr)
        return 130
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
