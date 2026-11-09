"""消费 Agent 事件，用 Rich 展示流式文本和运行诊断。"""

from contextlib import contextmanager

from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.text import Text

from timocode.agent.state import AgentEvent, RunResult
from timocode.agent.session import CompactionEntry, SessionEntry
from timocode.agent.session_runtime import AgentSession
from timocode.llm.schemas import AssistantMessage, Usage


# 模型和工具文本按字面显示，过滤能控制终端的字符，保留换行与制表。
_CONTROL_CHARACTERS = dict.fromkeys(
    [value for value in range(32) if value not in (9, 10)] + list(range(127, 160)),
)


def literal_text(value: str, *, style: str = "") -> Text:
    return Text(value.translate(_CONTROL_CHARACTERS), style=style)


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
        self._activity: Status | None = None
        self._activity_text: str | None = None

    def _print(self, console: Console, value: str, *, style: str = "", end: str = "\n") -> None:
        console.print(literal_text(value, style=style), end=end, soft_wrap=True)

    def activity(self, value: str) -> None:
        self.end_text()
        self._activity_text = value
        if not self.diagnostics.is_terminal:
            self._print(self.diagnostics, value, style="cyan")
            return
        if self._activity is None:
            self._activity = self.diagnostics.status(literal_text(value), spinner="dots")
            self._activity.start()
        else:
            self._activity.update(literal_text(value))

    def stop_activity(self) -> None:
        if self._activity is not None:
            self._activity.stop()
            self._activity = None
        self._activity_text = None

    @contextmanager
    def suspend_activity(self):
        previous = self._activity_text
        self.stop_activity()
        try:
            yield
        finally:
            if previous is not None:
                self.activity(previous)

    def welcome(self, session: AgentSession) -> None:
        mascot = Text("   .----.\n  [ o  o ]\n  [  --  ]\n   '----'\n   /|  |\\", style="bold cyan")
        details = literal_text(
            f"TimoCode\n\n模型  {session.model}\n目录  {session.snapshot.header.cwd}\n会话  {session.session_id}"
        )
        header = Table.grid(padding=(0, 2), expand=True)
        header.add_column(width=12, no_wrap=True)
        header.add_column(ratio=1, overflow="fold")
        header.add_row(mascot, details)
        body = Group(
            header,
            Text("\n输入任务开始对话 · ↑/↓ 选择审批 · Enter 确认", style="dim"),
            Text("/new  /resume  /model  /compact  /status  /exit", style="dim"),
        )
        self.diagnostics.print(Panel(body, title="timo", border_style="cyan", padding=(1, 2)))

    def end_text(self) -> None:
        if self._line_open:
            self.output.print()
            self._line_open = False

    def info(self, value: str) -> None:
        self.stop_activity()
        self.end_text()
        self._print(self.diagnostics, value, style="dim")

    def error(self, value: str) -> None:
        self.stop_activity()
        self.end_text()
        self._print(self.diagnostics, value, style="red")

    async def __call__(self, event: AgentEvent) -> None:
        if event.type == "model_request_start":
            self.activity("思考中…")
        elif event.type == "message_complete" and isinstance(event.message, AssistantMessage):
            message = event.message
            if not message.tool_calls and message.content:
                self.stop_activity()
                self._print(self.output, message.content, end="")
                self._line_open = not message.content.endswith("\n")
                self.end_text()
        elif event.type == "tool_start":
            call = event.tool_call
            labels = {
                "read_file": "正在读取文件", "list_files": "正在查看目录", "search_files": "正在搜索代码",
                "write_file": "正在写入文件", "edit_file": "正在修改文件",
                "bash": "正在执行命令", "docker_shell": "正在执行隔离命令",
            }
            self.activity(labels.get(call.name, "正在使用工具") + "…")
        elif event.type == "tool_end":
            self.activity("工具执行失败，正在调整…" if event.message.is_error else "正在整理结果…")
        elif event.type == "compaction_start":
            self.activity("正在生成历史摘要…")
        elif event.type == "compaction_end":
            self.info("上下文压缩完成" if event.reason == "compacted" else "当前历史无需压缩")
        elif event.type == "compaction_failed":
            self.error(f"[compaction failed] {event.reason}")
        elif event.type == "run_end":
            self.stop_activity()
            self.end_text()

    def finish(self, result: RunResult) -> None:
        self.info(f"[{result.status}] " + ("已完成" if result.status == "completed" else "任务已结束"))
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
