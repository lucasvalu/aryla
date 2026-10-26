"""Main graph with fake models and an in-process MCP server, through the production code path."""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from aryla.graph import build_graph
from aryla.llm import ROLE_CODING, ROLE_GENERAL, ROLE_ORCHESTRATOR
from aryla.memory import open_checkpointer
from aryla.tools.mcp import open_tools
from tests.fakes.fake_chat_model import ai, ai_with_tool_calls, routing_ok
from tests.fakes.fake_mcp_server import create_fake_filesystem_server


@pytest.fixture
async def coding_tools(read_only_policy):
    fake = create_fake_filesystem_server()
    async with open_tools(fake.mcp, read_only_policy) as tools:
        yield tools, fake


def thread(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


async def test_policy_filters_mcp_tools(coding_tools):
    tools, _ = coding_tools
    assert sorted(t.name for t in tools) == ["list_directory", "read_text_file", "search_files"]


async def test_coding_read_path_end_to_end(runtime, fake_models, coding_tools, read_only_policy):
    tools, server = coding_tools
    runtime.coding_tools, runtime.coding_policy = tools, read_only_policy
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("coding", 0.9)]
    fake_models[ROLE_CODING].responses = [
        ai_with_tool_calls(("read_text_file", {"path": "/repo/main.py"})),
        ai("main.py defines main(), which prints hello."),
    ]
    graph = build_graph(runtime, InMemorySaver())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="what does main.py do?")]}, thread("t1")
    )

    assert [type(m).__name__ for m in result["messages"]] == ["HumanMessage", "AIMessage"]
    assert result["messages"][-1].content == "main.py defines main(), which prints hello."
    assert result["routing"].target_agent == "coding"
    assert result["confirmations"] == []
    assert "print('hello')" in fake_models[ROLE_CODING].calls[1][-1].content
    assert server.writes == [] and server.moves == []


async def test_general_and_clarify_paths(runtime, fake_models):
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [
        routing_ok("general", 0.9),
        routing_ok("general", 0.2),
    ]
    fake_models[ROLE_GENERAL].responses = [ai("Canberra.")]
    graph = build_graph(runtime, InMemorySaver())

    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="capital of Australia?")]}, thread("t2")
    )
    second = await graph.ainvoke({"messages": [HumanMessage(content="so?")]}, thread("t2"))

    assert first["messages"][-1].content == "Canberra."
    assert second["routing"].target_agent == "clarify"
    assert "Could you give me more detail?" in second["messages"][-1].content
    assert "- coding:" not in second["messages"][-1].content
    assert len(second["messages"]) == 4


async def test_write_confirmation_through_the_main_graph(runtime, fake_models, policy_with_writes):
    server = create_fake_filesystem_server()
    async with open_tools(server.mcp, policy_with_writes) as tools:
        runtime.coding_tools, runtime.coding_policy = tools, policy_with_writes
        fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("coding", 0.95)]
        fake_models[ROLE_CODING].responses = [
            ai_with_tool_calls(("write_file", {"path": "/repo/new.py", "content": "x = 1\n"})),
            ai("Created /repo/new.py."),
        ]
        graph = build_graph(runtime, InMemorySaver())
        config = thread("t3")

        paused = await graph.ainvoke({"messages": [HumanMessage(content="create new.py")]}, config)
        assert "__interrupt__" in paused
        interrupt = paused["__interrupt__"][0]
        assert server.writes == []

        request = interrupt.value["requests"][0]
        done = await graph.ainvoke(
            Command(resume={interrupt.id: {request["tool_call_id"]: True}}), config
        )

        assert done["messages"][-1].content == "Created /repo/new.py."
        assert [c.approved for c in done["confirmations"]] == [True]
        assert server.writes[0]["path"] == "/repo/new.py"


async def test_conversation_persists_across_invocations_with_sqlite(
    runtime, fake_models, tmp_path: Path
):
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("general")] * 2
    fake_models[ROLE_GENERAL].responses = [ai("Nice to meet you, Ana!"), ai("Your name is Ana.")]
    db = tmp_path / "checkpoints.sqlite"

    async with open_checkpointer(db) as checkpointer:
        graph = build_graph(runtime, checkpointer)
        await graph.ainvoke({"messages": [HumanMessage(content="my name is Ana")]}, thread("t4"))

    async with open_checkpointer(db) as checkpointer:
        graph = build_graph(runtime, checkpointer)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="what is my name?")]}, thread("t4")
        )

    assert len(result["messages"]) == 4
    second_prompt = fake_models[ROLE_GENERAL].calls[1]
    assert any("my name is Ana" in m.content for m in second_prompt if isinstance(m, HumanMessage))
    assert isinstance(result["messages"][-1], AIMessage)
