"""General agent: a single model call, no tools."""

from __future__ import annotations

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from aryla import prompts
from aryla.agents.base import Agent
from aryla.llm import ROLE_GENERAL
from aryla.messages import message_text
from aryla.observability import flow
from aryla.runtime import Runtime
from aryla.state import AgentState

SUMMARY_PREFIX = "Summary of the conversation so far (older messages):\n"


class GeneralAgent(Agent):
    name = "general"
    description = (
        "Non-programming topics: general knowledge, business and process questions, writing, "
        "proofreading and translating prose, naming ideas, and questions about the conversation "
        "itself. No access to systems or the internet."
    )
    examples = (
        "what is the capital of Australia?",
        "help me write an email requesting vacation",
        "proofread this paragraph",
        "translate 'deadline' to Portuguese",
        "how do I explain scrum to someone outside IT?",
        "summarize what we discussed so far",
        "what did I ask you before?",
    )

    def __init__(self, runtime: Runtime) -> None:
        super().__init__(runtime)
        self.model = runtime.model(ROLE_GENERAL)
        self.graph = self.build()

    def build(self) -> CompiledStateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("answer", self.answer)
        graph.add_edge(START, "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    async def answer(self, state: AgentState) -> dict:
        system_text = prompts.load("general")
        if state.get("summary"):
            system_text += "\n" + SUMMARY_PREFIX + state.get("summary", "")

        flow.info("general: calling the model")
        response: AIMessage = await self.model.ainvoke(
            [SystemMessage(content=system_text), *state.get("messages", [])]
        )
        flow.info("general: answer ready (%d chars)", len(message_text(response)))
        return {"messages": [response], "final_answer": message_text(response)}
