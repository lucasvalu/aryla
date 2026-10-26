"""The CLI's stream_once over the real graph: token output, single interrupt detection, resume."""

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from aryla.cli import stream_turn
from aryla.graph import build_graph
from aryla.llm import ROLE_CODING, ROLE_GENERAL, ROLE_ORCHESTRATOR
from aryla.tools.mcp import open_tools
from tests.fakes.fake_chat_model import ai, ai_with_tool_calls, routing_ok
from tests.fakes.fake_mcp_server import create_fake_filesystem_server


async def test_general_answer_is_streamed(runtime, fake_models, capsys):
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("general")]
    fake_models[ROLE_GENERAL].responses = [ai("Streamed answer.")]
    graph = build_graph(runtime, InMemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "s1"}}

    interrupts, streamed = await stream_turn(
        graph, {"messages": [HumanMessage(content="hi, explain X")]}, config
    )

    assert interrupts == []
    assert streamed is True
    assert "Streamed answer." in capsys.readouterr().out


async def test_interrupt_is_detected_once_and_resumed(runtime, fake_models, policy_with_writes):
    fake = create_fake_filesystem_server()
    async with open_tools(fake.mcp, policy_with_writes) as tools:
        runtime.coding_tools, runtime.coding_policy = tools, policy_with_writes
        fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("coding")]
        fake_models[ROLE_CODING].responses = [
            ai_with_tool_calls(("write_file", {"path": "/repo/s.py", "content": "s"})),
            ai("Created."),
        ]
        graph = build_graph(runtime, InMemorySaver())
        config: RunnableConfig = {"configurable": {"thread_id": "s2"}}

        interrupts, _ = await stream_turn(
            graph, {"messages": [HumanMessage(content="create s.py")]}, config
        )

        assert len(interrupts) == 1
        request = interrupts[0].value["requests"][0]
        assert request["tool_name"] == "write_file"

        answers = {interrupts[0].id: {request["tool_call_id"]: True}}
        interrupts_after, _ = await stream_turn(graph, Command(resume=answers), config)

        assert interrupts_after == []
        state = await graph.aget_state(config)
        assert state.values["messages"][-1].content == "Created."
        assert fake.writes[0]["path"] == "/repo/s.py"
        assert state.next == ()
