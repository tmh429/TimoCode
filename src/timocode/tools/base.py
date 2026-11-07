"""工具描述与执行入口，完整注册和权限系统在 Phase 2 扩展。"""

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
