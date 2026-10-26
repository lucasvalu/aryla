"""Base class for specialist agents: catalog metadata, a subgraph, and the node that runs it."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from aryla.observability import flow
from aryla.runtime import Runtime
from aryla.state import ConversationState, SessionContext


class Agent:
    name: str = ""
    description: str = ""
    examples: tuple[str, ...] = ()

    graph: CompiledStateGraph

    @classmethod
    def available(cls, runtime: Runtime) -> bool:
        return True

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        workspace = runtime.settings.coding_workspace
        self.default_session = SessionContext(workspace_root=str(workspace) if workspace else None)

    def build(self) -> CompiledStateGraph:
        raise NotImplementedError

    async def run(self, state: ConversationState, config: RunnableConfig) -> dict:
        """Main-graph node: runs the agent's subgraph on its own state and returns the answer."""
        payload = {
            "messages": list(state.get("messages", [])),
            "summary": state.get("summary", ""),
            "session": state.get("session") or self.default_session,
        }
        flow.info(
            "handoff: orchestrator -> %s agent (%d messages in context)",
            self.name,
            len(payload["messages"]),
        )
        # GraphInterrupt is an Exception; never wrap this call in a bare `except`.
        output = await self.graph.ainvoke(payload, config)
        answer = output.get("final_answer") or ""
        flow.info("handoff: %s agent -> consolidate (%d chars)", self.name, len(answer))
        return {"agent_output": answer, "confirmations": output.get("confirmations", [])}
