import pytest
from langchain_core.tools import tool

from aryla.agents import AGENTS, available_agents, catalog, get_agent


@tool
def read_text_file(path: str) -> str:
    """read"""
    return ""


def test_coding_is_unavailable_without_tools(runtime):
    assert [agent.name for agent in available_agents(runtime)] == ["general"]


def test_coding_becomes_available_with_tools(runtime):
    runtime.coding_tools = [read_text_file]
    assert [agent.name for agent in available_agents(runtime)] == ["coding", "general"]


def test_catalog_mentions_every_agent_and_examples():
    text = catalog(AGENTS)
    for agent in AGENTS:
        assert f"- {agent.name}:" in text
        assert agent.examples[0] in text


def test_get_agent_unknown_raises():
    with pytest.raises(KeyError):
        get_agent("missing")
