"""Conversation memory: the SQLite checkpointer and the summary of older messages."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aryla import prompts
from aryla.messages import message_text, transcript
from aryla.observability import flow
from aryla.state import ConversationState

STATE_TYPES: list[tuple[str, str]] = [
    ("aryla.state", "RoutingDecision"),
    ("aryla.state", "ToolCallRequest"),
    ("aryla.state", "ConfirmationRecord"),
    ("aryla.state", "SessionContext"),
]

FALLBACK_ANSWER = "Sorry, I couldn't produce an answer this time. Could you try again?"

SUMMARIZER_PROMPT = prompts.load("summarizer")


class Memory:
    def __init__(self, model: BaseChatModel, max_recent: int, keep_last: int) -> None:
        self.model = model
        self.max_recent = max_recent
        self.keep_last = keep_last

    def consolidate(self, state: ConversationState) -> dict:
        text = (state.get("agent_output") or "").strip()
        if not text:
            text = FALLBACK_ANSWER
        return {"messages": [AIMessage(content=text)], "agent_output": None}

    async def summarize(self, state: ConversationState) -> dict:
        messages = state.get("messages", [])
        if len(messages) <= self.max_recent:
            return {}

        old, _recent = split_for_summary(messages, self.keep_last)
        if not old:
            return {}
        flow.info(
            "memory: %d messages in history, summarizing the oldest %d", len(messages), len(old)
        )

        prompt = [
            SystemMessage(content=SUMMARIZER_PROMPT),
            HumanMessage(
                content=(
                    f"Previous summary:\n{state.get('summary') or '(none)'}\n\n"
                    f"Messages to fold in:\n{transcript(old)}"
                )
            ),
        ]
        response = await self.model.ainvoke(prompt)
        summary = message_text(response).strip()
        flow.info("memory: summarized %d older messages into %d chars", len(old), len(summary))

        removals = [RemoveMessage(id=message.id) for message in old if message.id is not None]
        return {"summary": summary, "messages": removals}


def split_for_summary(
    messages: list[BaseMessage], keep_last: int
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split into (old, recent), moving the cut back so `recent` starts at a user message."""
    cut = max(len(messages) - keep_last, 0)
    while cut > 0 and not isinstance(messages[cut], HumanMessage):
        cut -= 1
    return list(messages[:cut]), list(messages[cut:])


@asynccontextmanager
async def open_checkpointer(db_path: Path | str) -> AsyncIterator[AsyncSqliteSaver]:
    """SQLite checkpointer that only rebuilds the Pydantic types listed in STATE_TYPES."""
    serializer = JsonPlusSerializer(allowed_msgpack_modules=STATE_TYPES)
    async with aiosqlite.connect(str(db_path)) as connection:
        checkpointer = AsyncSqliteSaver(connection, serde=serializer)
        await checkpointer.setup()
        yield checkpointer
