"""单次运行指标，缺失用量和不完整追踪保持未知。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from timocode.llm.schemas import DataModel

if TYPE_CHECKING:
    from timocode.evaluation.runner import EvaluationResult


FailureType = Literal[
    "acceptance_failed", "iteration_limit", "provider_error", "invalid_response",
    "storage_error", "context_limit", "compaction_error", "agent_error",
    "task_timeout", "token_limit", "usage_unavailable", "verification_timeout",
    "verification_error", "runner_error", "cancelled",
]


class ToolCounts(DataModel):
    attempted: int = Field(default=0, ge=0)
    succeeded: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    interrupted: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def consistent_counts(self) -> Self:
        if self.attempted != self.succeeded + self.failed + self.interrupted:
            raise ValueError("工具调用次数必须等于成功、失败与中断次数之和")
        return self


class ToolMetrics(DataModel):
    total: ToolCounts = Field(default_factory=ToolCounts)
    by_name: dict[str, ToolCounts] = Field(default_factory=dict)

    @model_validator(mode="after")
    def consistent_totals(self) -> Self:
        for field in ToolCounts.model_fields:
            if getattr(self.total, field) != sum(getattr(counts, field) for counts in self.by_name.values()):
                raise ValueError("工具总计必须与各工具分项相符")
        return self


class RunMetrics(DataModel):
    failure_type: FailureType | None = None
    tools: ToolMetrics | None = None
    tool_success_rate: float | None = Field(default=None, ge=0, le=1)
    reported_input_tokens: int = Field(default=0, ge=0)
    reported_output_tokens: int = Field(default=0, ge=0)
    token_usage_complete: bool
    total_tokens: int | None = Field(default=None, ge=0)


def classify_failure(result: EvaluationResult) -> FailureType | None:
    if result.status == "passed":
        return None
    if result.status == "failed":
        return "acceptance_failed"
    if result.status == "agent_error":
        status = result.agent_result.status if result.agent_result is not None else None
        if status == "max_iterations":
            return "iteration_limit"
        if status in {"provider_error", "invalid_response", "storage_error", "context_limit", "compaction_error"}:
            return status
        return "agent_error"
    return result.status


def measure_run(result: EvaluationResult, tools: ToolMetrics | None) -> RunMetrics:
    input_tokens = sum(usage.input_tokens or 0 for usage in result.usage)
    output_tokens = sum(usage.output_tokens or 0 for usage in result.usage)
    complete = len(result.usage) == result.iterations and all(
        usage.input_tokens is not None and usage.output_tokens is not None for usage in result.usage
    )
    return RunMetrics(
        failure_type=classify_failure(result), tools=tools,
        tool_success_rate=(
            tools.total.succeeded / tools.total.attempted
            if tools is not None and tools.total.attempted else None
        ),
        reported_input_tokens=input_tokens, reported_output_tokens=output_tokens,
        token_usage_complete=complete, total_tokens=input_tokens + output_tokens if complete else None,
    )


def read_tool_metrics(path: Path, session_id: str) -> ToolMetrics | None:
    """只采纳同一执行的完整追踪，以 span_id 关联工具开始、完成和中断。"""
    calls = {}
    ended = set()
    run_id = None
    finished = False
    sequence = 0
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if not isinstance(entry, dict):
                    return None
                if finished or entry.get("session_id") != session_id or entry.get("sequence") != sequence + 1:
                    return None
                if sequence == 0:
                    run_id = entry.get("run_id")
                    if not isinstance(run_id, str) or entry.get("event") != "trace_start":
                        return None
                if entry.get("run_id") != run_id:
                    return None
                sequence += 1
                event = entry.get("event")
                if sequence > 1 and event == "trace_start":
                    return None
                if event in {"tool_start", "tool_end"}:
                    span = entry.get("span_id")
                    data = entry.get("data")
                    if not isinstance(span, str) or not isinstance(data, dict):
                        return None
                    if event == "tool_start":
                        name = data.get("name")
                        if span in calls or not isinstance(name, str) or not name:
                            return None
                        calls[span] = (name, "interrupted")
                    else:
                        if span not in calls or span in ended:
                            return None
                        if data.get("interrupted") is True:
                            outcome = "interrupted"
                        elif data.get("status") in {"ok", "error"}:
                            outcome = "succeeded" if data["status"] == "ok" else "failed"
                        else:
                            return None
                        calls[span] = (calls[span][0], outcome)
                        ended.add(span)
                elif event == "trace_end":
                    finished = True
        if not finished or len(ended) != len(calls):
            return None
    except (OSError, ValueError, TypeError):
        return None
    total = ToolCounts()
    by_name = {}
    for name, outcome in calls.values():
        counts = by_name.setdefault(name, ToolCounts())
        counts.attempted += 1
        setattr(counts, outcome, getattr(counts, outcome) + 1)
        total.attempted += 1
        setattr(total, outcome, getattr(total, outcome) + 1)
    return ToolMetrics(total=total, by_name=by_name)
