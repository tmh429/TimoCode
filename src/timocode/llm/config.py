"""读取用户、项目与显式指定的模型配置。"""

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import Field, SecretStr, ValidationError, field_validator

from timocode.llm.schemas import DataModel
from timocode.llm.retry import RetryPolicy


DEFAULT_BASE_URL = "https://api.openai.com/v1"


class LLMSettings(DataModel):
    api_key: SecretStr
    base_url: str = DEFAULT_BASE_URL
    model: str = Field(min_length=1)
    timeout: float = Field(default=60.0, gt=0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)

    @field_validator("api_key")
    @classmethod
    def require_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API key 不能为空")
        return value

    @field_validator("model")
    @classmethod
    def require_model(cls, value: str) -> str:
        value = value.strip()
        if not value or value == "your-model-name":
            raise ValueError("请填写实际模型名称")
        return value

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        parsed.port  # 触发非法端口校验，由 Pydantic 转换为配置错误。
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Base URL 必须为不含认证信息、查询和片段的 HTTP(S) 地址")
        return value


def user_config_file() -> Path:
    return Path.home() / ".timocode" / "config.env"


def load_settings(env_file: str | Path | None = None) -> LLMSettings:
    if env_file is None:
        local = dict(dotenv_values(user_config_file(), interpolate=False))
        project = dotenv_values(".env", interpolate=False)
        user_url = local.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        project_url = project.get("OPENAI_BASE_URL")
        if (
            project_url is not None
            and project_url.strip().rstrip("/") != user_url.strip().rstrip("/")
            and not os.environ.get("OPENAI_BASE_URL")
        ):
            # 项目切换服务时，从项目配置或显式进程环境获取对应凭据。
            local.pop("OPENAI_API_KEY", None)
        local.update(project)
    else:
        local = dotenv_values(env_file, interpolate=False)
    names = {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
    }
    values = {}
    for field, name in names.items():
        value = os.environ.get(name, local.get(name))
        if value is not None:
            values[field] = value
    timeout = os.environ.get("OPENAI_TIMEOUT", local.get("OPENAI_TIMEOUT"))
    retries = os.environ.get("OPENAI_MAX_RETRIES", local.get("OPENAI_MAX_RETRIES"))
    try:
        if timeout is not None:
            values["timeout"] = float(timeout)
        if retries is not None:
            values["retry_policy"] = {"max_retries": int(retries)}
    except ValueError:
        raise ValueError("请检查 OPENAI_TIMEOUT 和 OPENAI_MAX_RETRIES 的数值配置") from None
    try:
        return LLMSettings(**values)
    except ValidationError as error:
        fields = ", ".join(str(item["loc"][0]) for item in error.errors())
        raise ValueError(f"模型配置缺失或无效，请检查字段 {fields}") from None
