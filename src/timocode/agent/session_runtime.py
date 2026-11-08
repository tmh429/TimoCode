"""连接 Agent 的完整消息事件、会话保存和重启恢复。"""

import asyncio
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from timocode.agent.context import ContextBudget, ContextBuilder, ContextReport, TokenEstimator
from timocode.agent.history import split_history
from timocode.agent.loop import Agent, DEFAULT_SYSTEM_PROMPT, EventHandler
from timocode.agent.session import (
    SessionConflictError, SessionError, SessionRecoveryRequired,
    CompactionEntry, SessionEntry, SessionSnapshot, SessionStore,
)
from timocode.agent.summarization import SummaryError, SummaryGenerator
from timocode.agent.state import AgentEvent, AgentState, RunResult
from timocode.llm.base import LLMProvider
from timocode.llm.schemas import AssistantMessage, LLMRequest, Message, SystemMessage, Usage
from timocode.tools.base import Tool


class AgentSession:
    def __init__(
        self, provider: LLMProvider, store: SessionStore, snapshot: SessionSnapshot,
        *, tools: Sequence[Tool] = (), max_iterations: int = 8,
        max_tokens: int | None = None, on_event: EventHandler | None = None,
        token_estimator: TokenEstimator | None = None,
    ):
        if snapshot.header.model is None:
            raise ValueError("会话缺少模型名称")
        if snapshot.header.tool_names != [tool.definition.name for tool in tools]:
            raise ValueError("恢复会话时的工具集合或顺序与保存配置不一致")
        if not snapshot.messages or not isinstance(snapshot.messages[0], SystemMessage):
            raise ValueError("会话缺少初始系统消息")
        self.store = store
        self._snapshot = snapshot.model_copy(deep=True)
        self._header = snapshot.header.model_copy(deep=True)
        self._token_estimator = token_estimator
        self._provider = provider
        self._tools = tuple(tools)
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._on_event = on_event
        self._running = False
        self._storage_unavailable = False
        self._observed_usage: list[Usage] = []
        self._agent = self._build_agent()

    @classmethod
    async def create(
        cls, provider: LLMProvider, model: str, cwd: str | Path,
        *, store: SessionStore | None = None, tools: Sequence[Tool] = (),
        system_prompt: str = DEFAULT_SYSTEM_PROMPT, max_iterations: int = 8,
        max_tokens: int | None = None, on_event: EventHandler | None = None,
        token_estimator: TokenEstimator | None = None,
        context_budget: ContextBudget | None = None,
    ) -> "AgentSession":
        store = store if store is not None else SessionStore()
        snapshot = await store.create(
            cwd, model=model, tool_names=[tool.definition.name for tool in tools],
            context_budget=context_budget,
        )
        snapshot = await store.append(snapshot.header.id, SystemMessage(content=system_prompt), expected_sequence=0)
        return cls(
            provider, store, snapshot, tools=tools, max_iterations=max_iterations,
            max_tokens=max_tokens, on_event=on_event, token_estimator=token_estimator,
        )

    @classmethod
    async def resume(
        cls, provider: LLMProvider, session_id: str,
        *, store: SessionStore | None = None, tools: Sequence[Tool] = (),
        expected_cwd: str | Path | None = None, max_iterations: int = 8,
        max_tokens: int | None = None, on_event: EventHandler | None = None,
        token_estimator: TokenEstimator | None = None,
    ) -> "AgentSession":
        store = store if store is not None else SessionStore()
        snapshot = await store.load(session_id)
        cls._require_ready(snapshot)
        if expected_cwd is not None and Path(expected_cwd).resolve(strict=True) != Path(snapshot.header.cwd):
            raise ValueError("授权工作目录与会话保存目录不一致")
        return cls(
            provider, store, snapshot, tools=tools, max_iterations=max_iterations,
            max_tokens=max_tokens, on_event=on_event, token_estimator=token_estimator,
        )

    @staticmethod
    def _require_ready(snapshot: SessionSnapshot) -> None:
        if snapshot.tail_issue is not None:
            raise SessionRecoveryRequired("会话尾行异常，请先使用 SessionStore.repair_tail 核查并修复")
        if snapshot.pending_tool_calls:
            identifiers = ", ".join(call.id for call in snapshot.pending_tool_calls)
            raise SessionRecoveryRequired(f"会话存在未完成工具调用 {identifiers}，请核查执行状态并补齐结果后再恢复")

    @property
    def session_id(self) -> str:
        return self._header.id

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot.model_copy(deep=True)

    @property
    def state(self) -> AgentState:
        state = self._agent.state
        if self._storage_unavailable:
            state.messages = self._snapshot.context_messages
        state.is_running = self._running
        state.needs_recovery |= self._snapshot.needs_recovery or self._storage_unavailable
        return state

    def _build_agent(self) -> Agent:
        return Agent(
            self._provider, self._header.model, tools=self._tools,
            initial_messages=self._snapshot.context_messages, max_iterations=self._max_iterations,
            max_tokens=self._max_tokens, on_event=self._handle_event,
            context_budget=self._header.context_budget, token_estimator=self._token_estimator,
            historical_tool_call_ids=[
                call.id for message in self._snapshot.messages if isinstance(message, AssistantMessage)
                for call in message.tool_calls
            ],
        )

    def _context_report(self, messages: Sequence[Message]) -> ContextReport:
        _, report = ContextBuilder(self._header.context_budget, self._token_estimator).build(LLMRequest(
            model=self._header.model, messages=list(messages),
            tools=[tool.definition for tool in self._tools], max_tokens=self._max_tokens,
        ))
        return report

    async def compact(
        self, summarizer: SummaryGenerator, *, keep_recent_turns: int = 2,
    ) -> ContextReport | None:
        """手动压缩完整旧前缀，采纳前检查长度与预算，提交后重建 Agent。"""
        if self._running:
            raise RuntimeError("同一个会话已有任务运行中")
        self._running = True
        try:
            await self._reload()
            self._require_ready(self._snapshot)
            parts = split_history(self._snapshot.messages, keep_recent_turns=keep_recent_turns)
            if not parts.can_compact:
                return None
            messages = [entry for entry in self._snapshot.entries if isinstance(entry, SessionEntry)]
            first_kept = messages[parts.first_kept_index]
            previous = self._snapshot.latest_compaction
            if previous is not None:
                previous_kept = next(entry for entry in messages if entry.id == previous.first_kept_entry_id)
                if first_kept.sequence <= previous_kept.sequence:
                    return None
            before = self._context_report(self._snapshot.context_messages)
            result = await summarizer.summarize(parts.to_summarize)
            entry = CompactionEntry(
                type="compaction", id=uuid.uuid4().hex,
                sequence=self._snapshot.last_sequence + 1, timestamp=datetime.now(timezone.utc),
                source_sequence=self._snapshot.last_sequence, first_kept_entry_id=first_kept.id,
                summary=result.summary, usage=result.usage,
            )
            candidate = self._snapshot.model_copy(deep=True)
            candidate.entries.append(entry)
            after = self._context_report(candidate.context_messages)
            if not after.fits:
                raise SummaryError("压缩后的上下文仍超过预算")
            if after.estimate.total >= before.estimate.total:
                raise SummaryError("摘要未缩短上下文")
            self._snapshot = await self.store.append_compaction(
                self.session_id, entry.summary, first_kept_entry_id=first_kept.id,
                expected_sequence=entry.source_sequence, entry_id=entry.id, usage=entry.usage,
            )
            self._agent = self._build_agent()
            return after
        except (Exception, asyncio.CancelledError):
            await self._reload_after_failure()
            raise
        finally:
            self._running = False

    async def _reload(self) -> None:
        snapshot = await self.store.load(self.session_id)
        if snapshot.header != self._header:
            raise SessionConflictError("会话头发生变化，请核查配置后重新打开会话")
        self._snapshot = snapshot
        self._agent = self._build_agent()
        self._storage_unavailable = False

    async def _reload_after_failure(self) -> None:
        try:
            await self._reload()
        except (SessionError, OSError, ValueError):
            self._storage_unavailable = True

    async def _handle_event(self, event: AgentEvent) -> None:
        if event.type == "message_complete":
            response = event.response
            if response is not None:
                self._observed_usage.append(response.usage.model_copy(deep=True))
            self._snapshot = await self.store.append(
                self.session_id, event.message, expected_sequence=self._snapshot.last_sequence,
                usage=response.usage if response is not None else None,
                stop_reason=response.stop_reason if response is not None else None,
            )
        if self._on_event is not None:
            await self._on_event(event)

    async def run(self, user_input: str) -> RunResult:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("用户输入必须是非空字符串")
        return await self._run(user_input)

    async def continue_run(self) -> RunResult:
        return await self._run(None)

    async def _run(self, user_input: str | None) -> RunResult:
        if self._running:
            raise RuntimeError("同一个会话已有任务运行中")
        self._running = True
        self._observed_usage = []
        started = False
        try:
            await self._reload()
            self._require_ready(self._snapshot)
            started = True
            if user_input is None:
                return await self._agent.continue_run()
            return await self._agent.run(user_input)
        except asyncio.CancelledError:
            await self._reload_after_failure()
            raise
        except (SessionError, OSError) as error:
            iterations = self._agent.state.iterations if started else 0
            report = self._agent.state.context_report if started else None
            await self._reload_after_failure()
            result = RunResult(
                status="storage_error", messages=self._snapshot.context_messages,
                iterations=iterations, usage=self._observed_usage, error=str(error), context_report=report,
            )
            await self._handle_event(AgentEvent(type="run_end", reason="storage_error", result=result.model_copy(deep=True)))
            return result
        except Exception:
            await self._reload_after_failure()
            raise
        finally:
            self._running = False
