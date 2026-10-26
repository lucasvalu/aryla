"""Small assertions shared by tests."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage


def tool_messages_match_calls(ai_message: AIMessage, tool_messages: Sequence[BaseMessage]) -> bool:
    expected_ids = [call["id"] for call in ai_message.tool_calls]
    actual_ids = [m.tool_call_id for m in tool_messages if isinstance(m, ToolMessage)]
    return expected_ids == actual_ids
