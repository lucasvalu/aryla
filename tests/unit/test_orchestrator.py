from langchain_core.messages import HumanMessage

from aryla.agents.coding import CodingAgent
from aryla.agents.general import GeneralAgent
from aryla.agents.orchestrator import Orchestrator
from aryla.llm import ROLE_ORCHESTRATOR
from aryla.state import CLARIFY, ConversationState, RoutingDecision
from tests.fakes.fake_chat_model import FakeChatModel, routing_empty, routing_invalid, routing_ok

AGENTS = [CodingAgent, GeneralAgent]


def make_orchestrator(fake_models) -> Orchestrator:
    return Orchestrator(fake_models[ROLE_ORCHESTRATOR], AGENTS, threshold=0.6)


def state_with(text: str) -> ConversationState:
    return {"messages": [HumanMessage(content=text)], "summary": ""}


def test_guard_passes_confident_known_agent():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    decision = RoutingDecision(target_agent="coding", confidence=0.9, rationale="ok")
    assert orchestrator.guard(decision) is decision


def test_guard_low_confidence_becomes_clarify():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    decision = RoutingDecision(target_agent="coding", confidence=0.3, rationale="?")
    result = orchestrator.guard(decision)
    assert result.target_agent == CLARIFY
    assert "threshold" in result.rationale


def test_guard_unknown_agent_becomes_clarify():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    decision = RoutingDecision(target_agent="confluence", confidence=0.99, rationale="?")
    assert orchestrator.guard(decision).target_agent == CLARIFY


def test_guard_none_becomes_clarify():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    assert orchestrator.guard(None).target_agent == CLARIFY


def test_clarify_is_not_subject_to_threshold():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    decision = RoutingDecision(target_agent=CLARIFY, confidence=0.1, rationale="vague")
    assert orchestrator.guard(decision) is decision


def test_prompt_contains_catalog_summary_and_current_message():
    orchestrator = Orchestrator(FakeChatModel(), AGENTS, threshold=0.6)
    state: ConversationState = {
        "messages": [HumanMessage(content="hi"), HumanMessage(content="and utils.py?")],
        "summary": "user asked about builder.py",
    }
    system, human = orchestrator.prompt(state)
    assert "- coding:" in system.content and "- general:" in system.content
    assert "clarify" in system.content
    assert "user asked about builder.py" in human.content
    assert "and utils.py?" in human.content


async def test_decide_routes_with_valid_structured_output(fake_models):
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_ok("general", 0.95)]
    orchestrator = make_orchestrator(fake_models)

    update = await orchestrator.decide(state_with("what is recursion?"))

    assert update["routing"].target_agent == "general"
    assert orchestrator.route({"messages": [], "routing": update["routing"]}) == "general"


async def test_decide_repairs_invalid_json_once(fake_models):
    fake = fake_models[ROLE_ORCHESTRATOR]
    fake.structured_responses = [routing_invalid(), routing_ok("coding", 0.9)]
    orchestrator = make_orchestrator(fake_models)

    update = await orchestrator.decide(state_with("review this function"))

    assert update["routing"].target_agent == "coding"
    assert len(fake.structured_calls) == 2
    assert "did not follow the required format" in fake.structured_calls[1][-1].content


async def test_decide_falls_back_to_clarify_after_failed_repair(fake_models):
    fake_models[ROLE_ORCHESTRATOR].structured_responses = [routing_invalid(), routing_invalid()]
    orchestrator = make_orchestrator(fake_models)

    update = await orchestrator.decide(state_with("???"))

    assert update["routing"].target_agent == CLARIFY
    assert update["routing"].confidence == 0.0


async def test_empty_model_response_counts_as_failure(fake_models):
    fake = fake_models[ROLE_ORCHESTRATOR]
    fake.structured_responses = [routing_empty(), routing_ok("general")]
    orchestrator = make_orchestrator(fake_models)

    update = await orchestrator.decide(state_with("explain docker"))

    assert update["routing"].target_agent == "general"
    assert len(fake.structured_calls) == 2
