"""消费 Agent 事件，用 Rich 展示流式文本和运行诊断。"""

import json

from rich.console import Console
from rich.text import Text

from timocode.agent.state import AgentEvent, RunResult
from timocode.agent.session import CompactionEntry, SessionEntry
from timocode.agent.session_runtime import AgentSession
from timocode.llm.schemas import AssistantMessage, Usage


# 模型和工具文本按字面显示，过滤能控制终端的字符，保留换行与制表。
_CONTROL_CHARACTERS = dict.fromkeys(
    [value for value in range(32) if value not in (9, 10)] + list(range(127, 160)),
)


def _token_total(values: list[int | None]) -> str:
    known = [value for value in values if value is not None]
    if not known:
        return "未知"
    total = str(sum(known))
    return total + "（部分已报告）" if len(known) != len(values) else total


class TerminalRenderer:
    def __init__(self, output: Console | None = None, diagnostics: Console | None = None):
        self.output = output if output is not None else Console()
        self.diagnostics = diagnostics if diagnostics is not None else Console(stderr=True)
        self._line_open = False

    def _print(self, console: Console, value: str, *, style: str = "", end: str = "\n") -> None:
        console.print(Text(value.translate(_CONTROL_CHARACTERS), style=style), end=end, soft_wrap=True)

    def end_text(self) -> None:
        if self._line_open:
            self.output.print()
            self._line_open = False

    def info(self, value: str) -> None:
        self.end_text()
        self._print(self.diagnostics, value, style="dim")

    def error(self, value: str) -> None:
        self.end_text()
        self._print(self.diagnostics, value, style="red")

    async def __call__(self, event: AgentEvent) -> None:
        if event.type == "text_delta":
            text = event.delta or ""
            self._print(self.output, text, end="")
            if text:
                self._line_open = not text.endswith("\n")
        elif event.type == "tool_start":
            call = event.tool_call
            self.info(f"[tool] {call.name} {json.dumps(call.arguments, ensure_ascii=False)}")
        elif event.type == "tool_end":
            state = "error" if event.message.is_error else "ok"
            self.info(f"[tool {state}] {event.tool_call.name}")
            self._print(self.diagnostics, event.message.content)
        elif event.type == "context_checked":
            report = event.context_report
            limit = "未设置" if report.input_limit is None else str(report.input_limit)
            self.info(f"[context] input~{report.estimate.total}, limit={limit}")
        elif event.type == "compaction_start":
            self.info("[compaction] 正在生成历史摘要或检查可压缩范围")
            if event.context_report is not None:
                self.info(f"[context before compaction] input~{event.context_report.estimate.total}")
        elif event.type == "compaction_end":
            self.info(f"[compaction] {event.reason}")
            if event.context_report is not None:
                self.info(f"[context after compaction] input~{event.context_report.estimate.total}")
        elif event.type == "compaction_failed":
            self.error(f"[compaction failed] {event.reason}")
        elif event.type == "run_end":
            self.end_text()

    def finish(self, result: RunResult) -> None:
        input_tokens = _token_total([usage.input_tokens for usage in result.usage])
        output_tokens = _token_total([usage.output_tokens for usage in result.usage])
        self.info(
            f"[{result.status}] model requests={result.iterations}, "
            f"reported tokens in={input_tokens}, out={output_tokens}"
        )
        if result.error:
            self.error(result.error)

    def show_status(self, session: AgentSession) -> None:
        snapshot = session.snapshot
        state = session.state
        report = session.context_report
        responses = [
            entry.usage or Usage() for entry in snapshot.entries
            if isinstance(entry, SessionEntry) and isinstance(entry.message, AssistantMessage)
        ]
        summaries = [entry.usage or Usage() for entry in snapshot.entries if isinstance(entry, CompactionEntry)]
        status = "idle"
        if state.is_running:
            status = "running"
        elif state.needs_recovery:
            status = "recovery_required"
        self.info(f"[status] {status}")
        self.info(f"[session] {snapshot.header.id}")
        self.info(f"[model] {snapshot.model}")
        self.info(f"[workspace] {snapshot.header.cwd}")
        self.info(f"[tools] {', '.join(snapshot.header.tool_names) or '无'}")
        self.info(
            f"[history] saved messages={len(snapshot.messages)}, "
            f"context messages={len(snapshot.context_messages)}, sequence={snapshot.last_sequence}"
        )
        limit = "未设置" if report.input_limit is None else str(report.input_limit)
        window = "未设置" if report.context_window is None else str(report.context_window)
        self.info(
            f"[context] 当前投影 input~{report.estimate.total}, limit={limit}, window={window}, "
            f"reserved output={report.reserved_output_tokens}, margin={report.safety_margin_tokens}, "
            f"fits={report.fits}, method={report.estimate.method}（下一条输入加入前）"
        )
        self.info(
            f"[usage task] saved responses={len(responses)}, "
            f"reported tokens in={_token_total([usage.input_tokens for usage in responses])}, "
            f"out={_token_total([usage.output_tokens for usage in responses])}"
        )
        self.info(
            f"[usage summary] saved summaries={len(summaries)}, "
            f"reported tokens in={_token_total([usage.input_tokens for usage in summaries])}, "
            f"out={_token_total([usage.output_tokens for usage in summaries])}"
        )
        self.info(f"[recovery] pending tools={len(snapshot.pending_tool_calls)}, tail issue={snapshot.tail_issue or '无'}")
        if session.trace_path is not None:
            self.info(f"[trace] {session.trace_path}")
        if session.trace_error is not None:
            self.error(session.trace_error)
