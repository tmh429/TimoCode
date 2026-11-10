"""评测运行配置，默认值保持现有单次和批量入口行为。"""

from typing import Literal

from pydantic import Field

from timocode.agent.context import ContextBudget
from timocode.llm.schemas import DataModel


class EvaluationOptions(DataModel):
    toolset: Literal["full", "files_only"] = "full"
    context_budget: ContextBudget | None = None
    max_output_tokens: int = Field(default=4096, gt=0)
