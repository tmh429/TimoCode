"""顺序工具调度、迭代限制和协作式取消的最小 Agent Loop。"""

import asyncio
import copy
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import aclosing

from pydantic import ValidationError

from timocode.agent.compaction import CompactionError
from timocode.agent.context import ContextBudget, ContextBuilder, ContextReport, TokenEstimator
from timocode.agent.state import AgentEvent, AgentState, RunResult, RunStatus
from timocode.llm.base import (
    LLMProvider,
    ProviderError,
    ProviderProtocolError,
    ResponseComplete,
    TextDelta,
    ToolCallDelta,
)
from timocode.llm.schemas import (
    AssistantMessage,
    LLMRequest,
    LLMResponse,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from timocode.tools.base import Tool
from timocode.tools.registry import ToolRegistry


EventHandler = Callable[[AgentEvent], Awaitable[None]]
ContextHandler = Callable[[ContextReport], Awaitable[Sequence[Message] | None]]
DEFAULT_SYSTEM_PROMPT = "你是编程助手，请依据工具的实际结果回答。"


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        tools: Sequence[Tool] = (),
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_iterations: int = 8,
        max_tokens: int | None = None,
        on_event: EventHandler | None = None,
        initial_messages: Sequence[Message] | None = None,
        context_budget: ContextBudget | None = None,
        token_estimator: TokenEstimator | None = None,
        historical_tool_call_ids: Sequence[str] = (),
        on_context: ContextHandler | None = None,
    ):
        if not model.strip():
            raise ValueError("模型名称不能为空")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations 必须为正整数")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
            raise ValueError("max_tokens 必须为正整数")
        self._context = ContextBuilder(context_budget, token_estimator)
        self.provider = provider
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.on_event = on_event
        self.on_context = on_context
        self._registry = ToolRegistry(tools)
        self._historical_tool_call_ids = set(historical_tool_call_ids)
        messages = [SystemMessage(content=system_prompt)] if initial_messages is None else [
            message.model_copy(deep=True) for message in initial_messages
        ]
        if not messages:
            raise ValueError("初始历史不能为空")
        self._state = AgentState(messages=messages)
        self._executing_call_id: str | None = None

    @property
    def state(self) -> AgentState:
        return copy.deepcopy(self._state)

    async def _emit(self, event: AgentEvent) -> None:
        if self.on_event is not None:
            await self.on_event(copy.deepcopy(event))

    async def _record(self, message: Message, *, response: LLMResponse | None = None) -> None:
        self._state.messages.append(message)
        await self._emit(AgentEvent(type="message_complete", message=message, response=response))

    async def _request_response(self, request: LLMRequest) -> LLMResponse:
        response = None
        async with aclosing(self.provider.stream(request)) as stream:
            async for event in stream:
                if response is not None:
                    raise ProviderProtocolError("完整响应后仍收到模型事件")
                if isinstance(event, TextDelta):
                    await self._emit(AgentEvent(type="text_delta", delta=event.delta))
                elif isinstance(event, ToolCallDelta):
                    await self._emit(AgentEvent(type="tool_call_delta", tool_delta=event))
                elif isinstance(event, ResponseComplete):
                    try:
                        response = LLMResponse.model_validate(event.response.model_dump())
                    except ValidationError:
                        raise ProviderProtocolError("Provider 返回的完整响应无效") from None
                else:
                    raise ProviderProtocolError("收到未知的模型事件")
        if response is None:
            raise ProviderProtocolError("模型流缺少完整响应")
        used_ids = {
            call.id
            for message in self._state.messages
            if isinstance(message, AssistantMessage)
            for call in message.tool_calls
        }
        used_ids.update(self._historical_tool_call_ids)
        if any(call.id in used_ids for call in response.message.tool_calls):
            raise ProviderProtocolError("模型复用了历史工具调用 ID")
        return response

    def _prepare_request(self) -> tuple[LLMRequest, ContextReport]:
        return self._context.build(LLMRequest(
            model=self.model, messages=self._state.messages,
            tools=self._registry.definitions(), max_tokens=self.max_tokens,
        ))

    async def _execute(self, call: ToolCall) -> None:
        await self._emit(AgentEvent(type="tool_start", tool_call=call))
        self._executing_call_id = call.id
        result = await self._registry.execute(call)
        self._executing_call_id = None
        message = ToolMessage(
            tool_call_id=call.id,
            name=call.name,
            content=result.content,
            is_error=result.is_error,
        )
        await self._record(message)
        await self._emit(AgentEvent(type="tool_end", tool_call=call, message=message))

    def _close_pending_calls(self) -> None:
        """中断后补齐诊断结果，执行中断的工具要求调用方核查副作用。"""
        if self._executing_call_id is not None:
            self._state.needs_recovery = True
        completed = {
            message.tool_call_id
            for message in self._state.messages
            if isinstance(message, ToolMessage)
        }
        for message in list(self._state.messages):
            if isinstance(message, AssistantMessage):
                for call in message.tool_calls:
                    if call.id not in completed:
                        self._state.messages.append(
                            ToolMessage(
                                tool_call_id=call.id,
                                name=call.name,
                                content=json.dumps({"error": "运行已中断，请核查工具执行状态后再继续。"}, ensure_ascii=False),
                                is_error=True,
                            )
                        )
                        completed.add(call.id)
        self._executing_call_id = None

    async def _finish(self, status: RunStatus, usage: list[Usage], error: str | None = None) -> RunResult:
        result = RunResult(
            status=status,
            messages=copy.deepcopy(self._state.messages),
            iterations=self._state.iterations,
            usage=copy.deepcopy(usage),
            error=error,
            context_report=copy.deepcopy(self._state.context_report),
        )
        await self._emit(AgentEvent(type="run_end", result=result, reason=status))
        return result

    async def run(self, user_input: str) -> RunResult:
        message = UserMessage(content=user_input)
        if not user_input.strip():
            raise ValueError("用户输入不能为空白")
        return await self._run(message)

    async def continue_run(self) -> RunResult:
        if not isinstance(self._state.messages[-1], (UserMessage, ToolMessage)):
            raise ValueError("续跑需要以用户消息或工具结果结束的历史")
        return await self._run(None)

    async def _run(self, message: UserMessage | None) -> RunResult:
        if self._state.is_running:
            raise RuntimeError("同一个 Agent 已有任务运行中")
        if self._state.needs_recovery:
            raise RuntimeError("上次工具执行中断，需要核查副作用并建立新的 Agent")
        self._state.is_running = True
        self._state.iterations = 0
        self._state.context_report = None
        usage: list[Usage] = []
        try:
            await self._emit(AgentEvent(type="run_start"))
            if message is not None:
                await self._record(message)
            for _ in range(self.max_iterations):
                request, report = self._prepare_request()
                self._state.context_report = report
                await self._emit(AgentEvent(type="context_checked", context_report=report))
                if self.on_context is not None:
                    replacement = await self.on_context(report.model_copy(deep=True))
                    if replacement is not None:
                        self._historical_tool_call_ids.update(
                            call.id for item in self._state.messages if isinstance(item, AssistantMessage)
                            for call in item.tool_calls
                        )
                        self._state.messages = [item.model_copy(deep=True) for item in replacement]
                        request, report = self._prepare_request()
                        self._state.context_report = report
                        await self._emit(AgentEvent(type="context_checked", context_report=report))
                if not report.fits:
                    return await self._finish(
                        "context_limit", usage, "估算输入超过上下文预算，请检查上下文配置",
                    )
                await self._emit(AgentEvent(type="model_request_start", request=request))
                self._state.iterations += 1
                response = await self._request_response(request)
                usage.append(response.usage)
                await self._record(response.message, response=response)
                if response.stop_reason == "length":
                    for call in response.message.tool_calls:
                        await self._record(
                            ToolMessage(
                                tool_call_id=call.id,
                                name=call.name,
                                content=json.dumps({"error": "模型输出被截断，工具未执行。"}, ensure_ascii=False),
                                is_error=True,
                            )
                        )
                    return await self._finish("invalid_response", usage, "模型输出被截断，工具调用未执行")
                if not response.message.tool_calls:
                    return await self._finish("completed", usage)
                for call in response.message.tool_calls:
                    await self._execute(call)
            return await self._finish("max_iterations", usage, "已达到最大模型请求次数")
        except asyncio.CancelledError:
            self._close_pending_calls()
            await self._emit(AgentEvent(type="run_end", reason="cancelled"))
            raise
        except ProviderProtocolError as error:
            return await self._finish("invalid_response", usage, str(error))
        except CompactionError as error:
            return await self._finish("compaction_error", usage, str(error))
        except ProviderError as error:
            return await self._finish("provider_error", usage, str(error))
        finally:
            self._close_pending_calls()
            self._state.is_running = False
