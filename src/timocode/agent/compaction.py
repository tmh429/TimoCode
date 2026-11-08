"""可持久化的自动压缩策略与运行时失败类型。"""

from pydantic import Field

from timocode.agent.context import ContextBudget, ContextReport
from timocode.llm.schemas import DataModel


class AutoCompactionPolicy(DataModel):
    trigger_ratio: float = Field(default=0.8, gt=0, le=1)
    keep_recent_turns: int = Field(default=2, gt=0)
    summary_budget: ContextBudget

    def should_compact(self, report: ContextReport) -> bool:
        return report.input_limit is not None and (
            not report.fits or report.estimate.total >= report.input_limit * self.trigger_ratio
        )


class CompactionError(Exception):
    """自动压缩失败，当前任务停止并保留已保存历史。"""
