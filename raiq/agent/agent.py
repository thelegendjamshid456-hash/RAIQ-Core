"""Public orchestration interface for the non-executing RAIQ Agent foundation."""

from __future__ import annotations

from dataclasses import dataclass

from raiq.agent.planner import create_plan
from raiq.agent.trace import AgentTrace
from raiq.agent.types import AgentPlan, TaskRequest


@dataclass(frozen=True)
class AgentPlanningResult:
    """Inspectable output of one RAIQ Agent planning invocation."""

    plan: AgentPlan
    trace: AgentTrace

    def to_dict(self) -> dict[str, object]:
        return {"plan": self.plan.to_dict(), "trace": self.trace.to_dict()}


class RAIQAgent:
    """Create structured technical plans without authorizing or executing tools."""

    def plan(self, request: TaskRequest) -> AgentPlanningResult:
        agent_plan = create_plan(request)
        trace = AgentTrace()
        trace.record_plan(agent_plan)
        return AgentPlanningResult(plan=agent_plan, trace=trace)
