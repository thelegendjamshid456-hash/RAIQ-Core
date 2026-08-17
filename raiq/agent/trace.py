"""Auditable, non-executing planning traces for the RAIQ Agent foundation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from raiq.agent.types import AgentPlan, TraceEvent


class AgentTrace:
    """Append-only in-memory trace for one RAIQ Agent planning request."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self.created_at = datetime.now(UTC).isoformat()

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(self, event_type: str, detail: str, **metadata: Any) -> TraceEvent:
        event = TraceEvent(
            sequence=len(self._events) + 1,
            event_type=event_type,
            detail=detail,
            metadata=metadata,
        )
        self._events.append(event)
        return event

    def record_plan(self, plan: AgentPlan) -> None:
        """Capture routing and proposed actions without invoking any proposed tool."""

        self.record("task_received", "Accepted a task for planning only.", task=plan.request.task)
        self.record(
            "route_selected",
            f"Selected {plan.route.specialization.value} at confidence {plan.route.confidence:.3f}.",
            specialization=plan.route.specialization.value,
            evidence=list(plan.route.evidence),
            scores=plan.route.scores,
        )
        self.record(
            "reasoning_depth_selected",
            f"Selected {plan.reasoning_depth.value} planning depth.",
            reasoning_depth=plan.reasoning_depth.value,
        )
        self.record(
            "tool_intents_proposed",
            "Recorded tool intents for later policy-controlled authorization; no tool was executed.",
            proposed_tools=[tool.value for tool in plan.proposed_tools],
        )
        self.record(
            "execution_blocked",
            "This agent phase has no tool executor and cannot run files, Python, shell, retrieval, or network actions.",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "events": [asdict(event) for event in self._events],
        }
