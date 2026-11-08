"""校验工具消息关联，并为摘要生成切分完整的用户回合。"""

from collections.abc import Sequence

from timocode.llm.schemas import (
    AssistantMessage, DataModel, Message, SystemMessage, ToolCall, ToolMessage,
    UserMessage,
)


def check_tool_association(
    message: Message, pending: dict[str, ToolCall], call_ids: set[str],
) -> None:
    """逐条检查工具关联，并更新调用方持有的校验状态。"""
    if isinstance(message, ToolMessage):
        call = pending.get(message.tool_call_id)
        if call is None or call.name != message.name:
            raise ValueError("工具结果缺少对应调用、重复回传或工具名称不匹配")
        del pending[message.tool_call_id]
        return
    if pending:
        raise ValueError("工具调用尚缺结果，不能插入其他消息")
    if isinstance(message, AssistantMessage):
        for call in message.tool_calls:
            if call.id in call_ids:
                raise ValueError("工具调用 ID 重复")
            call_ids.add(call.id)
            pending[call.id] = call


class HistorySplit(DataModel):
    system_messages: list[SystemMessage]
    to_summarize: list[Message]
    recent_messages: list[Message]

    @property
    def can_compact(self) -> bool:
        return bool(self.to_summarize)

    @property
    def first_kept_index(self) -> int:
        """保留历史在原消息列表中的起始位置，采用从零开始的索引。"""
        return len(self.system_messages) + len(self.to_summarize)


def split_history(
    messages: Sequence[Message], *, keep_recent_turns: int = 2,
) -> HistorySplit:
    """保留系统指令和最近若干用户回合，返回独立的消息副本。

    用户消息开始一个新回合，直到下一个用户消息前的全部内容属于该回合。
    允许保留以用户消息或完整工具结果结束、等待模型继续回答的回合。
    所有工具调用必须已有结果，避免摘要处理尚未核查的工具执行状态。
    """
    if type(keep_recent_turns) is not int or keep_recent_turns < 1:
        raise ValueError("keep_recent_turns 必须为正整数")
    if not messages or not isinstance(messages[0], SystemMessage):
        raise ValueError("历史必须以系统消息开始")

    system_count = 0
    turn_starts: list[int] = []
    pending: dict[str, ToolCall] = {}
    call_ids: set[str] = set()
    for index, message in enumerate(messages):
        check_tool_association(message, pending, call_ids)
        if isinstance(message, SystemMessage):
            if turn_starts:
                raise ValueError("系统消息必须位于历史开头")
            system_count += 1
        elif isinstance(message, UserMessage):
            turn_starts.append(index)
        elif not turn_starts:
            raise ValueError("对话历史必须从用户消息开始")
    if pending:
        raise ValueError("历史存在未完成工具调用，请先核查并补齐结果")

    cut = system_count
    if len(turn_starts) > keep_recent_turns:
        cut = turn_starts[-keep_recent_turns]
    copies = [message.model_copy(deep=True) for message in messages]
    return HistorySplit(
        system_messages=copies[:system_count],
        to_summarize=copies[system_count:cut],
        recent_messages=copies[cut:],
    )
