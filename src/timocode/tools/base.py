"""工具描述、执行入口与统一结果，供 Registry 和具体工具使用。"""

from typing import Protocol

from pydantic import JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition


class ToolResult(DataModel):
    content: str
    is_error: bool = False


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult: ...
