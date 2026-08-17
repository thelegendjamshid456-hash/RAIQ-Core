"""Typed contracts for the bounded, auditable RAIQ Agent foundation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Specialization(str, Enum):
    """The technical specialization selected by RAIQ Agent routing."""

    CODE = "RAIQ Code"
    NEURAL = "RAIQ Neural"
    CHEM = "RAIQ Chem"
    TECHNICAL = "RAIQ Technical"


class ReasoningDepth(str, Enum):
    """Bounded planning depth selected from task complexity signals."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ToolIntent(str, Enum):
    """Proposed capabilities; this phase never executes them."""

    FILES = "files"
    PYTHON = "python"
    SHELL = "shell"
    SEARCH = "search"
    RETRIEVAL = "retrieval"
    VERIFY = "verify"


@dataclass(frozen=True)
class TaskRequest:
    """A user task supplied to the RAIQ Agent planning layer."""

    task: str
    context: str = ""
    supplied_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.task or not self.task.strip():
            raise ValueError("task must contain non-whitespace text")


@dataclass(frozen=True)
class RouteDecision:
    """Transparent routing output with keyword evidence and confidence."""

    specialization: Specialization
    confidence: float
    evidence: tuple[str, ...]
    scores: dict[str, int]


@dataclass(frozen=True)
class PlanStep:
    """One human-readable, independently auditable action in an agent plan."""

    order: int
    objective: str
    proposed_tools: tuple[ToolIntent, ...] = ()
    verification: str = "Review the result against the task requirements."


@dataclass(frozen=True)
class AgentPlan:
    """A bounded plan that may be inspected before any action is authorized."""

    request: TaskRequest
    route: RouteDecision
    reasoning_depth: ReasoningDepth
    steps: tuple[PlanStep, ...]
    proposed_tools: tuple[ToolIntent, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe plan data for a CLI, REST API, or audit record."""

        payload = asdict(self)
        payload["route"]["specialization"] = self.route.specialization.value
        payload["reasoning_depth"] = self.reasoning_depth.value
        for step in payload["steps"]:
            step["proposed_tools"] = [tool.value for tool in step["proposed_tools"]]
        payload["proposed_tools"] = [tool.value for tool in self.proposed_tools]
        return payload


@dataclass(frozen=True)
class TraceEvent:
    """Immutable trace event representing an Agent planning decision or proposal."""

    sequence: int
    event_type: str
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)
