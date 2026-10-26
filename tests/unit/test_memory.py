from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from aryla.llm import ROLE_SUMMARIZER
from aryla.memory import FALLBACK_ANSWER, Memory, split_for_summary
from tests.fakes.fake_chat_model import FakeChatModel, ai


def test_consolidate_turns_agent_output_into_ai_message():
    memory = Memory(FakeChatModel(), max_recent=6, keep_last=2)
    update = memory.consolidate({"messages": [], "agent_output": "  answer  "})
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == "answer"
    assert update["agent_output"] is None


def test_consolidate_uses_fallback_when_empty():
    memory = Memory(FakeChatModel(), max_recent=6, keep_last=2)
    update = memory.consolidate({"messages": [], "agent_output": ""})
    assert update["messages"][0].content == FALLBACK_ANSWER


def conversation(turns: int) -> list:
    messages = []
    for index in range(turns):
        messages.append(HumanMessage(content=f"question {index}", id=f"h{index}"))
        messages.append(AIMessage(content=f"answer {index}", id=f"a{index}"))
    return messages


def test_split_keeps_recent_starting_at_human_message():
    old, recent = split_for_summary(conversation(4), keep_last=3)
    assert [m.id for m in recent] == ["h2", "a2", "h3", "a3"]
    assert [m.id for m in old] == ["h0", "a0", "h1", "a1"]


async def test_summarize_is_noop_below_limit(fake_models):
    memory = Memory(fake_models[ROLE_SUMMARIZER], max_recent=6, keep_last=2)
    assert await memory.summarize({"messages": conversation(3), "summary": ""}) == {}


async def test_summarize_removes_old_messages_and_updates_summary(fake_models):
    fake = fake_models[ROLE_SUMMARIZER]
    fake.responses = [ai("user asked 0..2")]
    memory = Memory(fake, max_recent=6, keep_last=2)

    update = await memory.summarize({"messages": conversation(4), "summary": "old summary"})

    assert update["summary"] == "user asked 0..2"
    removed = [m for m in update["messages"] if isinstance(m, RemoveMessage)]
    assert [m.id for m in removed] == ["h0", "a0", "h1", "a1", "h2", "a2"]
    assert "old summary" in fake.calls[0][-1].content
