"""State schemas for the main conversation graph and for the agent subgraphs."""

from __future__ import annotations

import operator
from datetime import UTC, datetime
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

CLARIFY = "clarify"


def utc_now() -> datetime:
    return datetime.now(UTC)


class RoutingDecision(BaseModel):
    target_agent: str = Field(
        description="Exact agent name from the catalog, or 'clarify' when the request is ambiguous"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the choice, 0 to 1")
    rationale: str = Field(description="One short sentence explaining the choice")


class ToolCallRequest(BaseModel):
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]


class ConfirmationRecord(BaseModel):
    tool_name: str
    args: dict[str, Any]
    approved: bool
    decided_at: datetime = Field(default_factory=utc_now)


class SessionContext(BaseModel):
    workspace_root: str | None = None


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: NotRequired[str]
    routing: NotRequired[RoutingDecision | None]
    agent_output: NotRequired[str | None]
    confirmations: NotRequired[Annotated[list[ConfirmationRecord], operator.add]]
    session: NotRequired[SessionContext]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: NotRequired[str]
    session: NotRequired[SessionContext]
    final_answer: NotRequired[str | None]
    pending_confirmation: NotRequired[list[ToolCallRequest]]
    confirmation_decisions: NotRequired[dict[str, bool]]
    confirmations: NotRequired[Annotated[list[ConfirmationRecord], operator.add]]
