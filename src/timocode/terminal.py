"""消费 Agent 事件，用 Rich 展示流式文本和运行诊断。"""

import json

from rich.console import Console
from rich.text import Text

from timocode.agent.state import AgentEvent, RunResult


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
            self.info("[compaction] 正在生成历史摘要")
        elif event.type == "compaction_end":
            self.info(f"[compaction] {event.reason}")
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
