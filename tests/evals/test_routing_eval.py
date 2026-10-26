"""Routing accuracy against the real Ollama model (slow; run with `pytest -m eval`)."""

from __future__ import annotations

import pytest
import yaml

from aryla.agents import AGENTS
from aryla.agents.orchestrator import Orchestrator
from aryla.config import PROJECT_ROOT, get_settings
from aryla.llm import ROLE_ORCHESTRATOR, load_model
from tests.evals.routing_report import evaluate_cases

CASES_PATH = PROJECT_ROOT / "tests" / "evals" / "routing_cases.yaml"
TARGET_ACCURACY = 0.90

pytestmark = pytest.mark.eval


def load_cases() -> list[dict]:
    with CASES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["cases"]


async def test_routing_accuracy_meets_target():
    settings = get_settings()
    model = load_model(ROLE_ORCHESTRATOR, settings)
    orchestrator = Orchestrator(model, AGENTS, settings.confidence_threshold)

    report = await evaluate_cases(orchestrator.decide, load_cases())
    print("\n" + report.render())

    assert report.accuracy >= TARGET_ACCURACY, (
        f"accuracy {report.accuracy:.0%} below target {TARGET_ACCURACY:.0%}; "
        f"failures: {[(r.message, r.expected, r.got) for r in report.failures()]}"
    )
