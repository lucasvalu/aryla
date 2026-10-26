"""Specialist agents and the list the orchestrator can route to."""

from __future__ import annotations

from aryla.agents.base import Agent
from aryla.agents.coding import CodingAgent
from aryla.agents.general import GeneralAgent
from aryla.runtime import Runtime

AGENTS: list[type[Agent]] = [CodingAgent, GeneralAgent]


def available_agents(runtime: Runtime) -> list[type[Agent]]:
    return [agent for agent in AGENTS if agent.available(runtime)]


def get_agent(name: str) -> type[Agent]:
    for agent in AGENTS:
        if agent.name == name:
            return agent
    raise KeyError(f"Agent '{name}' is not registered; known: {[a.name for a in AGENTS]}")


def catalog(agents: list[type[Agent]]) -> str:
    blocks = []
    for agent in agents:
        examples = "\n".join(f'    - "{example}"' for example in agent.examples)
        blocks.append(f"- {agent.name}: {agent.description}\n  Example requests:\n{examples}")
    return "\n".join(blocks)
