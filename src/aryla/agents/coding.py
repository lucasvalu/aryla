"""Coding agent: a ReAct loop over filesystem MCP tools, with human confirmation for writes."""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from aryla import prompts
from aryla.agents.base import Agent
from aryla.llm import ROLE_CODING
from aryla.messages import content_to_text, message_text
from aryla.observability import flow, short
from aryla.runtime import Runtime
from aryla.state import AgentState, ConfirmationRecord, SessionContext, ToolCallRequest
from aryla.tools.policy import ToolKind

WORKSPACE_HINT = "Workspace root: {path}. Paths in tool calls must be inside it."

SUMMARY_PREFIX = "Summary of the conversation so far (older messages):\n"

REFUSED_BY_USER = "Operation refused by the user. Do not retry; tell the user."

UNKNOWN_TOOL = "Tool '{name}' is not available. Use only the listed tools."

ITERATION_LIMIT_ANSWER = (
    "I couldn't finish within the step limit. "
    "Try a narrower request (for example, name the file or function you mean)."
)

INTERRUPT_KIND_CONFIRM_WRITES = "confirm_write_tools"


class CodingAgent(Agent):
    name = "coding"
    description = (
        "Programming: explains, writes, reviews and debugs source code in any language, "
        "answers questions about programming concepts, libraries, tools and errors, and reads "
        "files from the configured workspace. Read-only."
    )
    examples = (
        "explain what this function does",
        "review this Python snippet for bugs",
        "write a shell script that renames files by date",
        "what does aryla/graph/builder.py do?",
        "why doesn't this regex match?",
        "what is the difference between a list and a tuple in Python?",
    )

    @classmethod
    def available(cls, runtime: Runtime) -> bool:
        return len(runtime.coding_tools) > 0

    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime)
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in runtime.coding_tools}
        self.policy = runtime.coding_policy
        self.max_rounds = runtime.settings.max_tool_iterations
        self.model = runtime.model(ROLE_CODING).bind_tools(runtime.coding_tools)
        self.graph = self.build()

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("call_model", self.call_model)
        graph.add_node("confirm_writes", self.confirm_writes)
        graph.add_node("run_tools", self.run_tools)
        graph.add_node("finish", self.finish)

        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model", self.next_step, ["finish", "confirm_writes", "run_tools"]
        )
        graph.add_edge("confirm_writes", "run_tools")
        graph.add_edge("run_tools", "call_model")
        graph.add_edge("finish", END)
        return graph.compile()

    # --- nodes -------------------------------------------------------------

    async def call_model(self, state: AgentState) -> dict:
        round_number = self._tool_rounds(state) + 1
        flow.info("coding: calling the model (round %d)", round_number)
        started = time.perf_counter()

        response: AIMessage = await self.model.ainvoke(
            [self._system_message(state), *state.get("messages", [])]
        )

        elapsed = time.perf_counter() - started
        if response.tool_calls:
            names = ", ".join(call["name"] for call in response.tool_calls)
            flow.info(
                "coding: model wants %d tool call(s) in %.1fs: %s",
                len(response.tool_calls),
                elapsed,
                names,
            )
        else:
            flow.info("coding: model answered in %.1fs", elapsed)
        return {"messages": [response]}

    def confirm_writes(self, state: AgentState) -> dict:
        """Interrupt once with the pending write calls; resume value is {tool_call_id: approved}.

        Everything before `interrupt()` re-runs on resume, so it must stay side-effect free.
        """
        last = self._last_ai_message(state)
        requests = []
        for call in last.tool_calls:
            if self.policy.is_write(call["name"]):
                requests.append(
                    ToolCallRequest(
                        tool_call_id=call["id"] or "", tool_name=call["name"], args=call["args"]
                    )
                )

        payload = {
            "kind": INTERRUPT_KIND_CONFIRM_WRITES,
            "requests": [request.model_dump() for request in requests],
        }
        flow.info("coding: waiting for human confirmation of %d write call(s)", len(requests))
        decisions: dict[str, bool] = interrupt(payload)

        records = []
        for request in requests:
            approved = bool(decisions.get(request.tool_call_id, False))
            records.append(
                ConfirmationRecord(
                    tool_name=request.tool_name, args=request.args, approved=approved
                )
            )
            flow.info(
                "coding: %s %s by the user", request.tool_name, "approved" if approved else "denied"
            )

        return {
            "pending_confirmation": requests,
            "confirmation_decisions": decisions,
            "confirmations": records,
        }

    async def run_tools(self, state: AgentState) -> dict:
        """Run the last AIMessage's tool calls in order, one ToolMessage per call."""
        last = self._last_ai_message(state)
        decisions = state.get("confirmation_decisions") or {}
        results: list[ToolMessage] = []

        for call in last.tool_calls:
            name = call["name"]
            call_id = call["id"] or ""
            kind = self.policy.classify(name)

            if kind is ToolKind.BLOCKED or name not in self.tools:
                flow.info("coding: model asked for unknown tool %s, rejecting", name)
                results.append(self._error_message(call_id, name, UNKNOWN_TOOL.format(name=name)))
            elif kind is ToolKind.WRITE and not decisions.get(call_id, False):
                flow.info("coding: skipping %s, refused by the user", name)
                results.append(self._error_message(call_id, name, REFUSED_BY_USER))
            else:
                results.append(await self._run_tool(self.tools[name], call))

        return {"messages": results, "pending_confirmation": [], "confirmation_decisions": {}}

    def finish(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls and message_text(last).strip():
            return {"final_answer": message_text(last)}
        return {"final_answer": ITERATION_LIMIT_ANSWER}

    # --- edges -------------------------------------------------------------

    def next_step(self, state: AgentState) -> str:
        last = self._last_ai_message(state)
        if not last.tool_calls:
            return "finish"
        if self._tool_rounds(state) > self.max_rounds:
            flow.info("coding: reached the limit of %d tool rounds, giving up", self.max_rounds)
            return "finish"
        for call in last.tool_calls:
            if self.policy.is_write(call["name"]):
                return "confirm_writes"
        return "run_tools"

    # --- helpers -----------------------------------------------------------

    def _system_message(self, state: AgentState) -> SystemMessage:
        text = prompts.load("coding")
        session: SessionContext = state.get("session") or SessionContext()
        if session.workspace_root:
            text += "\n" + WORKSPACE_HINT.format(path=session.workspace_root)
        if state.get("summary"):
            text += "\n" + SUMMARY_PREFIX + state.get("summary", "")
        return SystemMessage(content=text)

    async def _run_tool(self, tool: BaseTool, call: ToolCall) -> ToolMessage:
        name = call["name"]
        call_id = call["id"] or ""
        flow.info("coding: running %s(%s)", name, short(call["args"]))
        started = time.perf_counter()

        try:
            raw = await tool.ainvoke(call)
        except GraphBubbleUp:
            raise
        except Exception as exc:  # noqa: BLE001 - tool failures are reported back to the model
            flow.info("coding: %s failed with %s", name, type(exc).__name__)
            return self._error_message(call_id, name, f"Error running {name}: {exc}")

        if isinstance(raw, ToolMessage):
            text = content_to_text(raw.content)
            status = raw.status
        else:
            text = content_to_text(raw)
            status = "success"

        elapsed = time.perf_counter() - started
        flow.info("coding: %s returned %s in %.1fs (%d chars)", name, status, elapsed, len(text))
        wrapped = f'<tool_result name="{name}">\n{text}\n</tool_result>'
        return ToolMessage(content=wrapped, tool_call_id=call_id, name=name, status=status)

    @staticmethod
    def _error_message(call_id: str, name: str, text: str) -> ToolMessage:
        return ToolMessage(content=text, tool_call_id=call_id, name=name, status="error")

    @staticmethod
    def _last_ai_message(state: AgentState) -> AIMessage:
        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                return message
        raise ValueError("Expected an AIMessage in the coding agent history")

    @staticmethod
    def _tool_rounds(state: AgentState) -> int:
        rounds = 0
        for message in state["messages"]:
            if isinstance(message, AIMessage) and message.tool_calls:
                rounds += 1
        return rounds
