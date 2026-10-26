"""Routing evaluation with a per-case report; `--model` compares another Ollama model."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for tests.evals

from aryla.agents import AGENTS  # noqa: E402
from aryla.agents.orchestrator import Orchestrator  # noqa: E402
from aryla.config import PROJECT_ROOT, get_settings  # noqa: E402
from aryla.llm import ROLE_ORCHESTRATOR, ModelProfile, build_chat_model, load_model  # noqa: E402
from aryla.observability import setup_logging  # noqa: E402
from tests.evals.routing_report import evaluate_cases  # noqa: E402

CASES_PATH = PROJECT_ROOT / "tests" / "evals" / "routing_cases.yaml"


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate orchestrator routing")
    parser.add_argument("--model", help="Ollama model to test instead of the configured one")
    parser.add_argument("--only", help="only run cases whose message contains this text")
    args = parser.parse_args()

    setup_logging()

    settings = get_settings()
    if args.model:
        model = build_chat_model(ModelProfile(provider="ollama", model=args.model), settings)
    else:
        model = load_model(ROLE_ORCHESTRATOR, settings)
    orchestrator = Orchestrator(model, AGENTS, settings.confidence_threshold)

    with CASES_PATH.open(encoding="utf-8") as handle:
        cases = yaml.safe_load(handle)["cases"]
    if args.only:
        cases = [c for c in cases if args.only.lower() in c["message"].lower()]

    print(f"model: {args.model or 'config/models.yaml (orchestrator)'}; cases: {len(cases)}\n")
    report = await evaluate_cases(
        orchestrator.decide,
        cases,
        on_result=lambda r: print(
            f"{'OK' if r.ok else 'XX'} {r.seconds:5.1f}s {r.expected:8s} -> {r.got:8s}  "
            f"{r.message}",
            flush=True,
        ),
    )
    print("\n" + report.render().split("\n\n")[-1])
    return 0 if report.accuracy >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
