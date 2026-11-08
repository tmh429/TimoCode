"""基础运行入口，可选择加法演示或指定工作目录的 Coding Tools。"""

import argparse
import asyncio
import sys
from pathlib import Path

from timocode.agent.compaction import AutoCompactionPolicy
from timocode.agent.context import ContextBudget
from timocode.agent.loop import DEFAULT_SYSTEM_PROMPT
from timocode.agent.session import SessionError, SessionStore
from timocode.agent.session_runtime import AgentSession
from timocode.agent.state import AgentEvent
from timocode.llm.config import load_settings
from timocode.llm.provider import OpenAICompatibleProvider
from timocode.tools.demo import AddTool
from timocode.tools.coding import create_coding_tools
from timocode.security.permissions import PermissionManager, confirm_console
from timocode.security.workspace import Workspace


async def render(event: AgentEvent) -> None:
    if event.type == "context_checked":
        report = event.context_report
        limit = "未设上限" if report.input_limit is None else str(report.input_limit)
        print(f"[context estimate] input~{report.estimate.total}, limit={limit}, method={report.estimate.method}", flush=True)
    elif event.type == "text_delta":
        print(event.delta, end="", flush=True)
    elif event.type == "tool_start":
        print(f"\n[tool] {event.tool_call.name}", flush=True)
    elif event.type == "tool_end":
        state = "error" if event.message.is_error else "ok"
        print(f"[tool {state}] {event.message.content}", flush=True)
    elif event.type == "compaction_start":
        print("\n[compaction] 正在生成历史摘要", flush=True)
    elif event.type == "compaction_end":
        print(f"[compaction {event.reason}] input~{event.context_report.estimate.total}", flush=True)
    elif event.type == "compaction_failed":
        print(f"[compaction failed] {event.reason}", flush=True)


async def run_demo(
    prompt: str | None, max_iterations: int, max_tokens: int | None,
    workspace: Path | None = None, resume_id: str | None = None,
    sessions_dir: Path = Path(".timocode/sessions"),
    context_window: int | None = None, reserve_output_tokens: int | None = None,
    safety_margin_tokens: int | None = None,
    auto_compact: bool = False, compact_at: float | None = None,
    keep_recent_turns: int | None = None, summary_context_window: int | None = None,
    summary_output_tokens: int | None = None,
) -> int:
    if not auto_compact and any(value is not None for value in (
        compact_at, keep_recent_turns, summary_context_window, summary_output_tokens,
    )):
        raise ValueError("设置压缩参数时必须指定 --auto-compact")
    if context_window is None and (reserve_output_tokens is not None or safety_margin_tokens is not None):
        raise ValueError("设置输出预留或安全余量时必须指定 --context-window")
    budget = None
    if context_window is not None:
        budget = ContextBudget(
            context_window=context_window,
            reserve_output_tokens=1024 if reserve_output_tokens is None else reserve_output_tokens,
            safety_margin_tokens=256 if safety_margin_tokens is None else safety_margin_tokens,
        )
    settings = load_settings()
    store = SessionStore(sessions_dir)
    snapshot = await store.load(resume_id) if resume_id is not None else None
    policy = None
    if auto_compact:
        saved = snapshot.header.auto_compaction if snapshot is not None else None
        task_budget = snapshot.header.context_budget if snapshot is not None else budget
        if task_budget is None:
            raise ValueError("自动压缩需要 --context-window 或会话已保存的上下文预算")
        if snapshot is not None and saved is None:
            raise ValueError("会话创建时未启用自动压缩，请创建启用该策略的新会话")
        values = saved.model_dump() if saved is not None else {
            "summary_budget": {
                "context_window": task_budget.context_window,
                "reserve_output_tokens": 2048,
            },
        }
        for key, value in (("trigger_ratio", compact_at), ("keep_recent_turns", keep_recent_turns)):
            if value is not None:
                values[key] = value
        for key, value in (
            ("context_window", summary_context_window), ("reserve_output_tokens", summary_output_tokens),
        ):
            if value is not None:
                values["summary_budget"][key] = value
        policy = AutoCompactionPolicy.model_validate(values)
        if snapshot is not None and policy != saved:
            raise ValueError("指定压缩策略与会话保存配置不一致，恢复时请省略压缩参数或使用原配置")
    if snapshot is not None:
        if budget is not None and budget != snapshot.header.context_budget:
            raise ValueError("指定预算与会话保存配置不一致，恢复时请使用原配置或省略预算参数")
        root = Path(snapshot.header.cwd)
        if workspace is not None and workspace.resolve(strict=True) != root:
            raise ValueError("指定工作目录与会话保存目录不一致")
        names = snapshot.header.tool_names
        if names == ["add"]:
            tools = [AddTool()]
        elif names == ["read_file", "write_file", "edit_file", "list_files", "search_files", "bash"]:
            tools = create_coding_tools(Workspace(root), PermissionManager(confirm_console))
        else:
            raise ValueError("此入口无法重建保存的工具集合，请通过 Python API 明确传入工具")
    else:
        root = workspace.resolve(strict=True) if workspace is not None else Path.cwd()
        tools = [AddTool()] if workspace is None else create_coding_tools(
            Workspace(root), PermissionManager(confirm_console),
        )
    prompt_rules = DEFAULT_SYSTEM_PROMPT
    if snapshot is None and workspace is not None:
        prompt_rules += "文件修改前先读取 sha256。收到 permission_denied 时停止对应操作并向用户说明。"
    async with OpenAICompatibleProvider(settings) as provider:
        if resume_id is None:
            session = await AgentSession.create(
                provider, settings.model, root, store=store, tools=tools,
                system_prompt=prompt_rules, max_iterations=max_iterations, context_budget=budget,
                auto_compaction=policy,
                max_tokens=max_tokens, on_event=render,
            )
        else:
            session = await AgentSession.resume(
                provider, resume_id, store=store, tools=tools, expected_cwd=root,
                max_iterations=max_iterations, max_tokens=max_tokens, on_event=render,
            )
        print(f"[session] {session.session_id}", flush=True)
        print(f"[model] {session.snapshot.header.model}", flush=True)
        print(f"[workspace] {root}", flush=True)
        result = await session.continue_run() if prompt is None else await session.run(prompt)
    print(f"\n[{result.status}] model requests={result.iterations}")
    if result.error:
        print(result.error, file=sys.stderr)
    return 0 if result.status == "completed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="TimoCode 工具调用演示")
    parser.add_argument("prompt", nargs="?", help="交给模型的自然语言任务，恢复待回答会话时可省略")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--workspace", type=Path, help="启用 Coding Tools 并授权指定工作目录")
    parser.add_argument("--resume", dest="resume_id", help="恢复指定会话 ID")
    parser.add_argument("--sessions-dir", type=Path, default=Path(".timocode/sessions"), help="会话文件目录")
    parser.add_argument("--context-window", type=int, help="上下文窗口 Token 数，显式指定后启用预算限制")
    parser.add_argument("--reserve-output-tokens", type=int, help="预留输出 Token，默认 1024")
    parser.add_argument("--safety-margin-tokens", type=int, help="安全余量 Token，默认 256")
    parser.add_argument("--auto-compact", action="store_true", help="启用自动压缩，恢复时沿用保存配置")
    parser.add_argument("--compact-at", type=float, help="可用输入预算触发比例，默认 0.8")
    parser.add_argument("--keep-recent-turns", type=int, help="压缩保留用户回合数，默认 2")
    parser.add_argument("--summary-context-window", type=int, help="摘要请求窗口，默认使用任务窗口")
    parser.add_argument("--summary-output-tokens", type=int, help="摘要输出预留，默认 2048")
    args = parser.parse_args()
    if args.prompt is None and args.resume_id is None:
        parser.error("需要任务内容或 --resume")
    try:
        return asyncio.run(run_demo(
            args.prompt, args.max_iterations, args.max_tokens, args.workspace,
            args.resume_id, args.sessions_dir, args.context_window,
            args.reserve_output_tokens, args.safety_margin_tokens,
            args.auto_compact, args.compact_at, args.keep_recent_turns,
            args.summary_context_window, args.summary_output_tokens,
        ))
    except KeyboardInterrupt:
        print("\n运行已取消。", file=sys.stderr)
        return 130
    except (ValueError, OSError, SessionError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
