"""连接与 HTTP 暂时失败的有界重试策略。"""

import math
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Self

from pydantic import Field, model_validator

from timocode.llm.base import ProviderConnectionError, ProviderError, ProviderHTTPError
from timocode.llm.schemas import DataModel


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - (now if now is not None else datetime.now(timezone.utc))).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


class RetryPolicy(DataModel):
    max_retries: int = Field(default=2, ge=0, le=5)
    base_delay: float = Field(default=0.5, ge=0)
    max_delay: float = Field(default=30.0, ge=0)

    @model_validator(mode="after")
    def check_delays(self) -> Self:
        if self.base_delay > self.max_delay:
            raise ValueError("重试基础等待时间不能超过等待上限")
        return self

    def delay(self, retry_index: int, error: ProviderError) -> float | None:
        retryable = (
            isinstance(error, ProviderHTTPError) and error.status_code in {429, 500, 502, 503, 504}
        ) or (isinstance(error, ProviderConnectionError) and error.retryable)
        if not retryable or retry_index >= self.max_retries:
            return None
        if isinstance(error, ProviderHTTPError) and error.retry_after is not None:
            return error.retry_after if error.retry_after <= self.max_delay else None
        return random.uniform(0, min(self.base_delay * 2 ** retry_index, self.max_delay))
