from langchain_core.messages import HumanMessage, SystemMessage

from aryla.agents.general import GeneralAgent
from aryla.llm import ROLE_GENERAL
from tests.fakes.fake_chat_model import ai


async def test_general_agent_answers_and_includes_summary(runtime, fake_models):
    fake = fake_models[ROLE_GENERAL]
    fake.responses = [ai("A checkpointer persists graph state.")]
    agent = GeneralAgent(runtime)

    result = await agent.graph.ainvoke(
        {
            "messages": [HumanMessage(content="what is a checkpointer?")],
            "summary": "we talked about LangGraph",
        }
    )

    assert result["final_answer"] == "A checkpointer persists graph state."
    prompt = fake.calls[0]
    assert isinstance(prompt[0], SystemMessage)
    assert "we talked about LangGraph" in prompt[0].content
    assert "no access to internal systems" in prompt[0].content
