"""Command-line chat: reads your messages, runs the graph, prints the answers."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt

from aryla.agents.coding import INTERRUPT_KIND_CONFIRM_WRITES
from aryla.config import get_settings
from aryla.graph import open_graph
from aryla.messages import message_text
from aryla.observability import (
    flow,
    flow_handler,
    get_callbacks,
    setup_logging,
    shutdown_tracing,
    trace_metadata,
)

logger = logging.getLogger(__name__)

STREAMING_NODES = {"answer", "call_model"}

HELP_TEXT = """Commands:
  /exit            quit
  /new             start a new conversation (new thread id)
  /state           show summary, last routing decision and confirmations
  /help            this message
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aryla: multi-agent terminal assistant (LangGraph + MCP + Ollama)"
    )
    parser.add_argument("--thread-id", help="resume an existing conversation")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show the flow log (routing, handoffs, tools)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="verbose logs, including message content"
    )
    args = parser.parse_args()

    setup_logging(debug=args.debug, verbose=args.verbose)
    flow_handler.before_emit = screen.end_line

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        sys.exit(130)


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    thread_id = args.thread_id or str(uuid.uuid4())
    show_server_logs = args.verbose or args.debug

    async with open_graph(settings, show_server_logs) as graph:
        try:
            await chat_loop(graph, thread_id)
        finally:
            shutdown_tracing()


async def chat_loop(graph: CompiledStateGraph, thread_id: str) -> None:
    config = thread_config(thread_id)
    print(f"thread_id: {thread_id}  (resume later with --thread-id {thread_id})")
    print("Type a message or /help.\n")

    leftovers = await pending_interrupts(graph, config)
    if leftovers:
        print("There is a pending confirmation from the previous session.")
        answers = {item.id: ask_confirmation(item.value) for item in leftovers}
        await handle_turn(graph, Command(resume=answers), config)

    while True:
        try:
            text = (await asyncio.to_thread(input, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue

        if text in ("/exit", "/quit"):
            return
        if text == "/help":
            print(HELP_TEXT)
            continue
        if text == "/new":
            thread_id = str(uuid.uuid4())
            config = thread_config(thread_id)
            print(f"new conversation. thread_id: {thread_id}\n")
            continue
        if text == "/state":
            await show_state(graph, config)
            print()
            continue

        print("assistant> ", end="", flush=True)
        try:
            await handle_turn(graph, {"messages": [HumanMessage(content=text)]}, config)
        except Exception as exc:  # noqa: BLE001 - keep the chat alive when a turn fails
            logger.exception("turn failed")
            print(f"\n[error] {type(exc).__name__}: {exc}\n")


async def handle_turn(graph: CompiledStateGraph, graph_input: Any, config: RunnableConfig) -> None:
    """Run one user turn to completion, asking for confirmation whenever the graph pauses."""
    started = time.perf_counter()
    flow.info("turn: started")

    pending: Any = graph_input
    while True:
        interrupts, streamed = await stream_turn(graph, pending, config)
        if not interrupts:
            break
        flow.info("turn: paused, %d confirmation(s) needed", len(interrupts))
        answers = {item.id: ask_confirmation(item.value) for item in interrupts}
        flow.info("turn: resuming with the user's decisions")
        pending = Command(resume=answers)

    if not streamed:
        state = await graph.aget_state(config)
        messages = state.values.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            print(message_text(messages[-1]))
    flow.info("turn: finished in %.1fs", time.perf_counter() - started)
    print()


async def stream_turn(
    graph: CompiledStateGraph, graph_input: Any, config: RunnableConfig
) -> tuple[list[Interrupt], bool]:
    """Consume one graph stream, printing tokens as they arrive.

    Returns the interrupts raised (write confirmations) and whether anything was printed.
    """
    interrupts: list[Interrupt] = []
    streamed = False

    async for chunk in graph.astream(
        graph_input,
        config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        version="v2",
    ):
        if chunk["type"] == "messages":
            message, metadata = chunk["data"]
            if not isinstance(message, AIMessageChunk) or message.tool_calls:
                continue
            if metadata.get("langgraph_node") not in STREAMING_NODES:
                continue
            text = message_text(message)
            if text:
                screen.token(text)
                streamed = True

        elif chunk["type"] == "updates" and "__interrupt__" in chunk["data"]:
            # subgraphs=True emits the interrupt for the subgraph and the root; keep the root one.
            if chunk["ns"] == ():
                interrupts.extend(chunk["data"]["__interrupt__"])

    screen.end_line()
    return interrupts, streamed


def ask_confirmation(payload: Any) -> dict[str, bool]:
    if not isinstance(payload, dict) or payload.get("kind") != INTERRUPT_KIND_CONFIRM_WRITES:
        print(f"[unknown interrupt] {payload!r}")
        return {}

    decisions: dict[str, bool] = {}
    print("\n=== Confirmation required: the agent wants to WRITE to the workspace ===")
    for request in payload.get("requests", []):
        print(f"\nTool: {request['tool_name']}")
        print("Arguments:")
        print(json.dumps(request["args"], ensure_ascii=False, indent=2))
        answer = input("Run it? [y/N] ").strip().lower()
        decisions[request["tool_call_id"]] = answer in ("y", "yes")
    print()
    return decisions


async def show_state(graph: CompiledStateGraph, config: RunnableConfig) -> None:
    state = await graph.aget_state(config)
    values = state.values

    print(f"thread_id: {config.get('configurable', {}).get('thread_id')}")
    print(f"messages in history: {len(values.get('messages', []))}")
    print(f"summary: {values.get('summary') or '(empty)'}")

    routing = values.get("routing")
    if routing:
        print(
            f"last decision: {routing.target_agent} "
            f"(confidence {routing.confidence:.2f}) - {routing.rationale}"
        )

    session = values.get("session")
    if session and session.workspace_root:
        print(f"workspace: {session.workspace_root}")

    confirmations = values.get("confirmations") or []
    print(f"confirmations recorded: {len(confirmations)}")
    for record in confirmations[-5:]:
        verdict = "approved" if record.approved else "denied"
        print(f"  - {record.decided_at:%H:%M:%S} {record.tool_name}: {verdict}")

    if state.next:
        print(f"pending nodes: {state.next}")


async def pending_interrupts(graph: CompiledStateGraph, config: RunnableConfig) -> list[Interrupt]:
    state = await graph.aget_state(config)
    found: list[Interrupt] = []
    for task in state.tasks:
        found.extend(task.interrupts)
    return found


def thread_config(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": get_callbacks(),
        "metadata": trace_metadata(thread_id),
    }


class Screen:
    """Tracks whether the cursor is mid-line so log lines never glue to a streamed answer."""

    def __init__(self) -> None:
        self.line_open = False

    def token(self, text: str) -> None:
        print(text, end="", flush=True)
        self.line_open = True

    def end_line(self) -> None:
        if self.line_open:
            print(flush=True)
            self.line_open = False


screen = Screen()


if __name__ == "__main__":
    main()
