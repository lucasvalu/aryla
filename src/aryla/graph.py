"""Root graph: orchestrator, agent subgraphs and memory, plus the factory for `langgraph dev`."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aryla.agents import available_agents
from aryla.agents.orchestrator import Orchestrator
from aryla.config import Settings, get_settings
from aryla.llm import ROLE_ORCHESTRATOR, ROLE_SUMMARIZER
from aryla.memory import Memory, open_checkpointer
from aryla.observability import flow
from aryla.runtime import Runtime, open_runtime
from aryla.state import CLARIFY, ConversationState


def build_graph(
    runtime: Runtime, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    agent_classes = available_agents(runtime)
    if not agent_classes:
        raise RuntimeError("No agent available; check the runtime")
    flow.info("graph: agents available: %s", ", ".join(agent.name for agent in agent_classes))

    orchestrator = Orchestrator(
        runtime.model(ROLE_ORCHESTRATOR), agent_classes, runtime.settings.confidence_threshold
    )
    agents = [agent_class(runtime) for agent_class in agent_classes]
    memory = Memory(
        runtime.model(ROLE_SUMMARIZER),
        runtime.settings.max_recent_messages,
        runtime.settings.keep_last_messages,
    )

    graph = StateGraph(ConversationState)
    graph.add_node("orchestrator", orchestrator.decide)
    for agent in agents:
        graph.add_node(agent.name, agent.run)
    graph.add_node(CLARIFY, orchestrator.clarify)
    graph.add_node("consolidate", memory.consolidate)
    graph.add_node("summarize", memory.summarize)

    destinations = [agent.name for agent in agents] + [CLARIFY]
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges("orchestrator", orchestrator.route, destinations)
    for name in destinations:
        graph.add_edge(name, "consolidate")
    graph.add_edge("consolidate", "summarize")
    graph.add_edge("summarize", END)

    return graph.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_graph(
    settings: Settings, show_server_logs: bool = False
) -> AsyncIterator[CompiledStateGraph]:
    """What the CLI uses: runtime + SQLite checkpointer, closed together on exit."""
    async with AsyncExitStack() as stack:
        runtime = await open_runtime(settings, stack, show_server_logs)
        checkpointer = await stack.enter_async_context(
            open_checkpointer(settings.checkpoint_db_path())
        )
        yield build_graph(runtime, checkpointer)


class DevServerGraph:
    """Builds the graph once for `langgraph dev`, which calls the factory on every run."""

    def __init__(self) -> None:
        self.resources = AsyncExitStack()
        self.graph: CompiledStateGraph | None = None

    async def get(self) -> CompiledStateGraph:
        if self.graph is None:
            runtime = await open_runtime(get_settings(), self.resources)
            self.graph = build_graph(runtime)
        return self.graph

    def close(self) -> None:
        with contextlib.suppress(RuntimeError):
            asyncio.run(self.resources.aclose())


_dev_server_graph = DevServerGraph()
atexit.register(_dev_server_graph.close)


async def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """Factory registered in langgraph.json. The dev server brings its own checkpointer."""
    return await _dev_server_graph.get()
