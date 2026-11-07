"""Phase 1 演示入口，通过自然语言触发加法工具。"""

import argparse
import asyncio
import sys

from timocode.agent.loop import Agent
from timocode.agent.state import AgentEvent
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.tools.demo import AddTool


async def render(event: AgentEvent) -> None:
    if event.type == "text_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_start":
        print(f"\n[tool] {event.tool_call.name}", flush=True)
    elif event.type == "tool_end":
        state = "error" if event.message.is_error else "ok"
        print(f"[tool {state}] {event.message.content}", flush=True)


async def run_demo(prompt: str, max_iterations: int, max_tokens: int | None) -> int:
    settings = load_settings()
    async with OpenAICompatibleProvider(settings) as provider:
        agent = Agent(
            provider,
            settings.model,
            tools=[AddTool()],
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
    parser = argparse.ArgumentParser(description="TimoCode Phase 1 工具调用演示")
    parser.add_argument("prompt", help="交给模型的自然语言任务")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args()
    try:
        return asyncio.run(run_demo(args.prompt, args.max_iterations, args.max_tokens))
    except KeyboardInterrupt:
        print("\n运行已取消。", file=sys.stderr)
        return 130
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
