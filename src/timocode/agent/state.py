"""单个 Agent 的运行状态、返回结果和观察事件。"""

from dataclasses import dataclass, field
from typing import Literal

from timocode.agent.context import ContextReport
from timocode.llm.base import ToolCallDelta
from timocode.llm.schemas import DataModel, LLMResponse, Message, ToolCall, Usage


RunStatus = Literal["completed", "max_iterations", "provider_error", "invalid_response", "storage_error", "context_limit"]


class RunResult(DataModel):
    status: RunStatus
    messages: list[Message]
    iterations: int
    usage: list[Usage]
    error: str | None = None
    context_report: ContextReport | None = None


@dataclass
class AgentState:
    messages: list[Message] = field(default_factory=list)
    iterations: int = 0
    is_running: bool = False
    needs_recovery: bool = False
    context_report: ContextReport | None = None


@dataclass(frozen=True)
class AgentEvent:
    type: Literal[
        "run_start", "context_checked", "model_request_start", "text_delta", "tool_call_delta",
        "message_complete", "tool_start", "tool_end", "run_end",
    ]
    message: Message | None = None
    delta: str | None = None
    tool_call: ToolCall | None = None
    tool_delta: ToolCallDelta | None = None
    result: RunResult | None = None
    reason: str | None = None
    response: LLMResponse | None = None
    context_report: ContextReport | None = None
