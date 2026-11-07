"""统一工具注册、Schema 校验和执行结果规范化。"""

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError as SchemaValidationError
from pydantic import ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable

from timocode.llm.schemas import ToolCall, ToolDefinition
from timocode.tools.base import Tool, ToolResult


@dataclass(frozen=True)
class _RegisteredTool:
    tool: Tool
    definition: ToolDefinition
    validator: Draft202012Validator


def _error_result(code: str, message: str) -> ToolResult:
    return ToolResult(
        content=json.dumps(
            {"error": {"code": code, "message": message}},
            ensure_ascii=False,
        ),
        is_error=True,
    )


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()):
        self._entries: dict[str, _RegisteredTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        definition = ToolDefinition.model_validate(tool.definition.model_dump())
        if definition.name in self._entries:
            raise ValueError(f"工具名称重复 {definition.name}")
        dialect = definition.parameters.get("$schema")
        if dialect not in (
            None,
            "https://json-schema.org/draft/2020-12/schema",
            "https://json-schema.org/draft/2020-12/schema#",
        ):
            raise ValueError("工具 Schema 需使用 JSON Schema Draft 2020-12")
        try:
            Draft202012Validator.check_schema(definition.parameters)
        except SchemaError as error:
            raise ValueError(f"工具 {definition.name} 的 Schema 无效 {error.message}") from None
        definition = definition.model_copy(deep=True)
        # 空资源库允许当前 Schema 内部引用，外部引用不会触发网络请求。
        validator = Draft202012Validator(definition.parameters, registry=Registry())
        self._entries[definition.name] = _RegisteredTool(tool, definition, validator)

    def definitions(self) -> list[ToolDefinition]:
        return [entry.definition.model_copy(deep=True) for entry in self._entries.values()]

    def get(self, name: str) -> Tool | None:
        entry = self._entries.get(name)
        return entry.tool if entry is not None else None

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            call = ToolCall.model_validate(call.model_dump())
        except ValidationError:
            return _error_result("invalid_arguments", "工具调用结构或参数不是有效的 JSON 数据")
        entry = self._entries.get(call.name)
        if entry is None:
            return _error_result("unknown_tool", f"工具 {call.name} 未注册")
        arguments = copy.deepcopy(call.arguments)
        try:
            entry.validator.validate(arguments)
        except SchemaValidationError as error:
            return _error_result("invalid_arguments", error.message)
        except Unresolvable:
            return _error_result("invalid_schema", "工具 Schema 引用了无法解析的资源")

        # 后续权限检查将在参数校验之后、具体工具执行之前接入。
        try:
            result = await entry.tool.execute(arguments)
        except Exception as error:
            return _error_result("execution_error", f"{type(error).__name__}: {error}")
        if not isinstance(result, ToolResult):
            return _error_result("invalid_result", "工具必须返回 ToolResult")
        try:
            return ToolResult.model_validate(result.model_dump())
        except ValidationError:
            return _error_result("invalid_result", "工具返回结果不符合 ToolResult 结构")
