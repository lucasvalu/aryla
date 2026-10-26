"""Runs routing cases through the orchestrator node and builds a report."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage


@dataclass
class CaseResult:
    message: str
    expected: str
    got: str
    confidence: float
    seconds: float

    @property
    def ok(self) -> bool:
        return self.expected == self.got


@dataclass
class Report:
    results: list[CaseResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return sum(r.ok for r in self.results) / len(self.results) if self.results else 0.0

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.ok]

    def render(self) -> str:
        lines = [
            f"{'OK' if r.ok else 'XX'} {r.seconds:5.1f}s {r.expected:8s} -> {r.got:8s} "
            f"({r.confidence:.2f})  {r.message}"
            for r in self.results
        ]
        per_expected = Counter(r.expected for r in self.results)
        per_expected_ok = Counter(r.expected for r in self.results if r.ok)
        breakdown = ", ".join(
            f"{name}: {per_expected_ok[name]}/{per_expected[name]}" for name in sorted(per_expected)
        )
        total_time = sum(r.seconds for r in self.results)
        hits = sum(r.ok for r in self.results)
        average = total_time / max(len(self.results), 1)
        lines.append("")
        lines.append(
            f"accuracy: {self.accuracy:.0%} ({hits}/{len(self.results)})  by class: {breakdown}"
        )
        lines.append(f"total time: {total_time:.0f}s, average {average:.1f}s per case")
        return "\n".join(lines)


async def evaluate_cases(
    node: Callable[[Any], Awaitable[dict]],
    cases: list[dict],
    on_result: Callable[[CaseResult], None] | None = None,
) -> Report:
    report = Report()
    for case in cases:
        state = {"messages": [HumanMessage(content=case["message"])], "summary": ""}
        started = time.perf_counter()
        update = await node(state)
        decision = update["routing"]
        result = CaseResult(
            message=case["message"],
            expected=case["expected"],
            got=decision.target_agent,
            confidence=decision.confidence,
            seconds=time.perf_counter() - started,
        )
        report.results.append(result)
        if on_result:
            on_result(result)
    return report
