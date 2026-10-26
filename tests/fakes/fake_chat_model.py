"""Scripted chat model for tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import Field

from aryla.state import RoutingDecision


class FakeChatModel(BaseChatModel):
    """Returns `responses` in order, records prompts in `calls`, streams word by word.

    `bind_tools` returns self; `with_structured_output` returns a runnable that pops
    `structured_responses` (dicts in the `include_raw=True` shape).
    """

    responses: list[AIMessage] = Field(default_factory=list)
    structured_responses: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    structured_calls: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _next_response(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeChatModel: no responses left")
        return self.responses.pop(0)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_response(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message = self._next_response(messages)

        if message.tool_calls:
            chunk = AIMessageChunk(
                content=message.content,
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"]),
                        "id": call["id"],
                        "index": index,
                        "type": "tool_call_chunk",
                    }
                    for index, call in enumerate(message.tool_calls)
                ],
            )
            yield ChatGenerationChunk(message=chunk)
            return

        text = message.content if isinstance(message.content, str) else str(message.content)
        words = text.split(" ")
        for index, word in enumerate(words):
            piece = word if index == len(words) - 1 else word + " "
            chunk = AIMessageChunk(content=piece)
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=ChatGenerationChunk(message=chunk))
            yield ChatGenerationChunk(message=chunk)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:  # type: ignore[override]
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:  # type: ignore[override]
        def next_structured(messages: Any) -> dict[str, Any]:
            self.structured_calls.append(list(messages))
            if not self.structured_responses:
                raise AssertionError("FakeChatModel: no structured responses left")
            return self.structured_responses.pop(0)

        return RunnableLambda(next_structured)


def ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def ai_with_tool_calls(*calls: tuple[str, dict[str, Any]], text: str = "") -> AIMessage:
    tool_calls = [
        {"name": name, "args": args, "id": f"call_{index}", "type": "tool_call"}
        for index, (name, args) in enumerate(calls, start=1)
    ]
    return AIMessage(content=text, tool_calls=tool_calls)


def routing_ok(target: str, confidence: float = 0.9, rationale: str = "test") -> dict[str, Any]:
    decision = RoutingDecision(target_agent=target, confidence=confidence, rationale=rationale)
    return {
        "raw": AIMessage(content=decision.model_dump_json()),
        "parsed": decision,
        "parsing_error": None,
    }


def routing_invalid(raw_text: str = "not json") -> dict[str, Any]:
    return {
        "raw": AIMessage(content=raw_text),
        "parsed": None,
        "parsing_error": ValueError("invalid JSON"),
    }


def routing_empty() -> dict[str, Any]:
    return {"raw": AIMessage(content=""), "parsed": None, "parsing_error": None}
