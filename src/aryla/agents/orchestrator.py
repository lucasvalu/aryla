"""Orchestrator: asks the model which agent should handle the request, then routes to it."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from aryla import prompts
from aryla.agents import catalog
from aryla.agents.base import Agent
from aryla.messages import transcript
from aryla.observability import flow, short
from aryla.state import CLARIFY, ConversationState, RoutingDecision

RECENT_MESSAGES_FOR_ROUTING = 6

CLARIFY_TEMPLATE = (
    "I'm not sure what you need. I can help with:\n{options}\nCould you give me more detail?"
)


class Orchestrator:
    def __init__(self, model: BaseChatModel, agents: list[type[Agent]], threshold: float) -> None:
        self.agents = agents
        self.agent_names = {agent.name for agent in agents}
        self.threshold = threshold
        self.router = model.with_structured_output(
            RoutingDecision, method="json_schema", include_raw=True
        )
        names = ", ".join([agent.name for agent in agents] + [CLARIFY])
        self.system_prompt = prompts.load("orchestrator").format(
            catalog=catalog(agents), names=names
        )
        self.repair_prompt = prompts.load("orchestrator_repair").strip()
        options = "\n".join(f"- {agent.name}: {agent.description}" for agent in agents)
        self.clarify_text = CLARIFY_TEMPLATE.format(options=options)

    # --- nodes -------------------------------------------------------------

    async def decide(self, state: ConversationState) -> dict:
        started = time.perf_counter()
        flow.info("orchestrator: deciding where to route the request")

        suggestion = await self._ask_model(self.prompt(state))
        decision = self.guard(suggestion)
        if suggestion is not None and decision is not suggestion:
            flow.info(
                "orchestrator: overriding %s -> clarify (%s)",
                suggestion.target_agent,
                decision.rationale,
            )
        flow.info(
            "orchestrator -> %s (confidence %.2f, %.1fs): %s",
            decision.target_agent,
            decision.confidence,
            time.perf_counter() - started,
            short(decision.rationale, 120),
        )
        return {"routing": decision, "agent_output": None}

    def clarify(self, state: ConversationState) -> dict:
        flow.info("clarify: asking the user for more detail")
        return {"agent_output": self.clarify_text}

    # --- edge --------------------------------------------------------------

    def route(self, state: ConversationState) -> str:
        routing = state.get("routing")
        if routing is None:
            return CLARIFY
        return routing.target_agent

    # --- steps of decide() -------------------------------------------------

    def prompt(self, state: ConversationState) -> list[Any]:
        messages = state.get("messages", [])
        current = messages[-1] if messages else None
        previous = messages[:-1][-RECENT_MESSAGES_FOR_ROUTING:]

        parts: list[str] = []
        if state.get("summary"):
            parts.append("Summary of the earlier conversation:\n" + state.get("summary", ""))
        if previous:
            parts.append("Recent messages:\n" + transcript(previous))
        current_text = current.content if current is not None else ""
        parts.append(f"Current user message:\n{current_text}")

        return [SystemMessage(content=self.system_prompt), HumanMessage(content="\n\n".join(parts))]

    async def _ask_model(self, prompt: list[Any]) -> RoutingDecision | None:
        """Invoke the structured router; on invalid output, retry once with the error attached."""
        result = await self.router.ainvoke(prompt)
        decision, error = self._parse(result)
        if decision is not None:
            return decision

        flow.info("orchestrator: invalid structured output (%s), retrying once", short(error))
        repair_prompt = list(prompt)
        if isinstance(result, dict) and isinstance(result.get("raw"), AIMessage):
            repair_prompt.append(result["raw"])
        repair_prompt.append(HumanMessage(content=self.repair_prompt.format(error=error)))
        result = await self.router.ainvoke(repair_prompt)
        decision, error = self._parse(result)
        if decision is None:
            flow.info("orchestrator: repair failed (%s), falling back to clarify", short(error))
        return decision

    @staticmethod
    def _parse(result: Any) -> tuple[RoutingDecision | None, str]:
        if isinstance(result, RoutingDecision):
            return result, ""
        parsed = result.get("parsed")
        if isinstance(parsed, RoutingDecision):
            return parsed, ""

        error = result.get("parsing_error")
        raw = result.get("raw")
        raw_text = str(getattr(raw, "content", "")).strip()
        if error is None and raw is not None and not raw_text:
            return None, "empty response"
        if error is None:
            return None, "no parsable content"
        return None, str(error)

    def guard(self, decision: RoutingDecision | None) -> RoutingDecision:
        """Turn the model's suggestion into a decision the graph can safely follow."""
        if decision is None:
            return RoutingDecision(
                target_agent=CLARIFY, confidence=0.0, rationale="router produced no valid decision"
            )
        if decision.target_agent != CLARIFY and decision.target_agent not in self.agent_names:
            return RoutingDecision(
                target_agent=CLARIFY,
                confidence=0.0,
                rationale=f"unknown agent suggested: {decision.target_agent!r}",
            )
        if decision.target_agent != CLARIFY and decision.confidence < self.threshold:
            return RoutingDecision(
                target_agent=CLARIFY,
                confidence=decision.confidence,
                rationale=(
                    f"confidence {decision.confidence:.2f} below threshold {self.threshold:.2f}"
                ),
            )
        return decision
