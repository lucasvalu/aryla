"""Helpers for getting plain text out of LangChain messages."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def message_text(message: BaseMessage) -> str:
    return content_to_text(message.content)


def transcript(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message_text(message)}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message_text(message)}")
    return "\n".join(lines)
