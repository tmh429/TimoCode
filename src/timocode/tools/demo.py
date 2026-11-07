"""无文件或命令副作用的加法工具，用于验证工具调用闭环。"""

import json

from pydantic import JsonValue

from timocode.llm.schemas import DataModel, ToolDefinition
from timocode.tools.base import ToolResult


class AddArguments(DataModel):
    a: int | float
    b: int | float


class AddTool:
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="add",
            description="计算两个有限数值的和，并返回真实计算结果。",
            parameters=AddArguments.model_json_schema(),
        )

    async def execute(self, arguments: dict[str, JsonValue]) -> ToolResult:
        params = AddArguments.model_validate(arguments)
        content = json.dumps({"result": params.a + params.b}, allow_nan=False)
        return ToolResult(content=content)
