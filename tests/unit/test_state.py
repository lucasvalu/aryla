import pytest
from pydantic import ValidationError

from aryla.state import CLARIFY, RoutingDecision


def test_clarify_is_a_valid_target():
    assert (
        RoutingDecision(target_agent=CLARIFY, confidence=0.1, rationale="x").target_agent == CLARIFY
    )


@pytest.mark.parametrize("confidence", [-0.1, 1.5])
def test_confidence_must_be_between_0_and_1(confidence):
    with pytest.raises(ValidationError):
        RoutingDecision(target_agent="coding", confidence=confidence, rationale="x")
