from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from aryla.agents.coding import (
    INTERRUPT_KIND_CONFIRM_WRITES,
    ITERATION_LIMIT_ANSWER,
    REFUSED_BY_USER,
    CodingAgent,
)
from aryla.llm import ROLE_CODING
from aryla.messages import message_text
from aryla.state import AgentState, SessionContext
from tests.fakes.fake_chat_model import ai, ai_with_tool_calls
from tests.helpers import tool_messages_match_calls


@tool
def list_directory(path: str) -> str:
    """List a directory."""
    return "[FILE] main.py\n[FILE] utils.py"


@tool
def read_text_file(path: str) -> str:
    """Read a text file."""
    if path.endswith("missing.py"):
        raise ValueError("file not found")
    return "def add(a, b):\n    return a + b\n"


@tool
def write_file(path: str, content: str) -> str:
    """Write a file (write)."""
    return f"Successfully wrote to {path}"


READ_TOOLS = [list_directory, read_text_file]


def user(text: str) -> AgentState:
    return {"messages": [HumanMessage(content=text)], "summary": "", "session": SessionContext()}


async def test_read_tool_runs_without_confirmation(runtime, fake_models, read_only_policy):
    runtime.coding_tools, runtime.coding_policy = READ_TOOLS, read_only_policy
    fake = fake_models[ROLE_CODING]
    fake.responses = [
        ai_with_tool_calls(("read_text_file", {"path": "/repo/utils.py"})),
        ai("`add` returns the sum of its two arguments."),
    ]

    result = await CodingAgent(runtime).graph.ainvoke(user("what does add do in utils.py?"))

    assert result["final_answer"] == "`add` returns the sum of its two arguments."
    assert result.get("confirmations", []) == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert message_text(tool_messages[0]).startswith('<tool_result name="read_text_file">')
    assert "return a + b" in message_text(tool_messages[0])
    assert "Never invent file contents" in fake.calls[0][0].content


async def test_tool_messages_keep_order_and_ids_even_with_errors(
    runtime, fake_models, read_only_policy
):
    runtime.coding_tools, runtime.coding_policy = READ_TOOLS, read_only_policy
    fake_models[ROLE_CODING].responses = [
        ai_with_tool_calls(
            ("read_text_file", {"path": "/repo/missing.py"}),
            ("made_up_tool", {}),
            ("read_text_file", {"path": "/repo/utils.py"}),
        ),
        ai("done"),
    ]

    result = await CodingAgent(runtime).graph.ainvoke(user("details"))

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages_match_calls(result["messages"][1], tool_messages)
    assert tool_messages[0].status == "error" and "Error running" in tool_messages[0].content
    assert tool_messages[1].status == "error" and "not available" in tool_messages[1].content
    assert tool_messages[2].status == "success"


async def test_iteration_limit_stops_the_loop(runtime, fake_models, read_only_policy):
    runtime.coding_tools, runtime.coding_policy = READ_TOOLS, read_only_policy
    fake = fake_models[ROLE_CODING]
    fake.responses = [ai_with_tool_calls(("list_directory", {"path": "/repo"})) for _ in range(5)]

    result = await CodingAgent(runtime).graph.ainvoke(user("loop"))

    assert result["final_answer"] == ITERATION_LIMIT_ANSWER
    assert (
        len(fake.calls) == 4
    )  # max_tool_iterations=3 executed rounds + the call that hit the limit


async def test_workspace_hint_goes_into_system_prompt(runtime, fake_models, read_only_policy):
    runtime.coding_tools, runtime.coding_policy = READ_TOOLS, read_only_policy
    fake = fake_models[ROLE_CODING]
    fake.responses = [ai("ok")]

    await CodingAgent(runtime).graph.ainvoke(
        {
            "messages": [HumanMessage(content="hi")],
            "summary": "",
            "session": SessionContext(workspace_root="/repo"),
        }
    )

    assert "Workspace root: /repo" in fake.calls[0][0].content


async def test_write_tool_interrupts_and_resumes_approved(runtime, fake_models, policy_with_writes):
    runtime.coding_tools = READ_TOOLS + [write_file]
    runtime.coding_policy = policy_with_writes
    fake_models[ROLE_CODING].responses = [
        ai_with_tool_calls(("write_file", {"path": "/repo/new.py", "content": "x = 1\n"})),
        ai("Wrote /repo/new.py."),
    ]
    graph = CodingAgent(runtime).graph
    graph.checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "t1"}}

    first = await graph.ainvoke(user("create new.py"), config)

    assert "__interrupt__" in first
    payload = first["__interrupt__"][0].value
    assert payload["kind"] == INTERRUPT_KIND_CONFIRM_WRITES
    request = payload["requests"][0]
    assert request["tool_name"] == "write_file"
    assert request["args"] == {"path": "/repo/new.py", "content": "x = 1\n"}

    second = await graph.ainvoke(Command(resume={request["tool_call_id"]: True}), config)

    assert second["final_answer"] == "Wrote /repo/new.py."
    assert [c.approved for c in second["confirmations"]] == [True]
    tool_messages = [m for m in second["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "success" and "Successfully wrote" in tool_messages[0].content


async def test_write_tool_denied_sends_refusal_to_model(runtime, fake_models, policy_with_writes):
    runtime.coding_tools = READ_TOOLS + [write_file]
    runtime.coding_policy = policy_with_writes
    fake_models[ROLE_CODING].responses = [
        ai_with_tool_calls(
            ("read_text_file", {"path": "/repo/utils.py"}),
            ("write_file", {"path": "/repo/utils.py", "content": "..."}),
        ),
        ai("Understood, I did not change the file."),
    ]
    graph = CodingAgent(runtime).graph
    graph.checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "t2"}}

    first = await graph.ainvoke(user("read and fix utils.py"), config)
    request = first["__interrupt__"][0].value["requests"][0]
    second = await graph.ainvoke(Command(resume={request["tool_call_id"]: False}), config)

    tool_messages = [m for m in second["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages_match_calls(second["messages"][1], tool_messages)
    assert tool_messages[0].status == "success"
    assert tool_messages[1].status == "error" and tool_messages[1].content == REFUSED_BY_USER
    assert [c.approved for c in second["confirmations"]] == [False]
    assert second["final_answer"] == "Understood, I did not change the file."


async def test_read_only_policy_never_interrupts_even_if_model_asks_for_write(
    runtime, fake_models, read_only_policy
):
    runtime.coding_tools, runtime.coding_policy = READ_TOOLS, read_only_policy
    fake_models[ROLE_CODING].responses = [
        ai_with_tool_calls(("write_file", {"path": "/repo/x.py", "content": ""})),
        ai("I can't write files; here is the code to apply."),
    ]
    graph = CodingAgent(runtime).graph
    graph.checkpointer = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "t3"}}

    result = await graph.ainvoke(user("create x.py"), config)

    assert "__interrupt__" not in result
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[0].status == "error"
    assert result["final_answer"] == "I can't write files; here is the code to apply."
