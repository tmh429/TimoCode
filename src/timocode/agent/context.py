"""请求前的近似 Token 计数与上下文预算，保持完整历史。"""

import json
from typing import Protocol, Self

from pydantic import Field, model_validator

from timocode.llm.schemas import DataModel, LLMRequest


class TextTokenCounter(Protocol):
    method: str

    def count(self, text: str) -> int: ...


class UTF8ByteCounter:
    """用字节数近似 Token，通常偏大，无法保证覆盖所有服务端模板。"""

    method = "utf8_bytes_v1"

    def count(self, text: str) -> int:
        return len(text.encode("utf-8"))


class TokenEstimate(DataModel):
    method: str
    system_tokens: int = Field(ge=0)
    history_tokens: int = Field(ge=0)
    tool_tokens: int = Field(ge=0)
    framing_tokens: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.system_tokens + self.history_tokens + self.tool_tokens + self.framing_tokens


class TokenEstimator:
    def __init__(self, counter: TextTokenCounter | None = None):
        self.counter = counter if counter is not None else UTF8ByteCounter()

    def _count(self, value: dict) -> int:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        count = self.counter.count(text)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Token 计数器必须返回非负整数")
        return count

    def estimate(self, request: LLMRequest) -> TokenEstimate:
        system = 0
        history = 0
        for message in request.messages:
            count = self._count(message.model_dump(exclude_none=True))
            if message.role == "system":
                system += count
            else:
                history += count
        return TokenEstimate(
            method=self.counter.method, system_tokens=system, history_tokens=history,
            tool_tokens=sum(self._count(tool.model_dump()) for tool in request.tools),
            # 工程启发式余量，未依据具体模型聊天模板标定。
            framing_tokens=16 + 12 * len(request.messages) + 16 * len(request.tools),
        )


class ContextBudget(DataModel):
    context_window: int = Field(gt=0)
    reserve_output_tokens: int = Field(default=1024, gt=0)
    safety_margin_tokens: int = Field(default=256, ge=0)

    @model_validator(mode="after")
    def require_input_room(self) -> Self:
        if self.reserve_output_tokens + self.safety_margin_tokens >= self.context_window:
            raise ValueError("上下文窗口必须大于输出预留与安全余量之和")
        return self


class ContextReport(DataModel):
    estimate: TokenEstimate
    context_window: int | None = None
    reserved_output_tokens: int = Field(ge=0)
    safety_margin_tokens: int = Field(ge=0)
    input_limit: int | None = None
    fits: bool


class ContextBuilder:
    def __init__(self, budget: ContextBudget | None = None, estimator: TokenEstimator | None = None):
        self.budget = budget.model_copy(deep=True) if budget is not None else None
        self.estimator = estimator if estimator is not None else TokenEstimator()

    def build(self, request: LLMRequest) -> tuple[LLMRequest, ContextReport]:
        prepared = request.model_copy(deep=True)
        estimate = self.estimator.estimate(prepared)
        if self.budget is None:
            return prepared, ContextReport(
                estimate=estimate, reserved_output_tokens=prepared.max_tokens or 0,
                safety_margin_tokens=0, fits=True,
            )
        budget = self.budget
        if prepared.max_tokens is None:
            prepared.max_tokens = budget.reserve_output_tokens
        reserved = max(budget.reserve_output_tokens, prepared.max_tokens)
        limit = budget.context_window - reserved - budget.safety_margin_tokens
        return prepared, ContextReport(
            estimate=estimate, context_window=budget.context_window,
            reserved_output_tokens=reserved, safety_margin_tokens=budget.safety_margin_tokens,
            input_limit=limit, fits=estimate.total <= limit,
        )
