from __future__ import annotations

from raiq.agent.agent import RAIQAgent
from raiq.agent.planner import create_plan
from raiq.agent.router import route_task, select_reasoning_depth
from raiq.agent.types import ReasoningDepth, Specialization, TaskRequest, ToolIntent


def test_code_task_routes_to_raiq_code() -> None:
    request = TaskRequest(
        task="Debug this FastAPI Python endpoint and add a regression test for the stack trace.",
        supplied_files=("service.py", "test_service.py"),
    )
    route = route_task(request)
    plan = create_plan(request)
    assert route.specialization is Specialization.CODE
    assert plan.reasoning_depth is ReasoningDepth.MODERATE
    assert ToolIntent.SHELL in plan.proposed_tools
    assert ToolIntent.VERIFY in plan.proposed_tools


def test_neural_task_routes_to_raiq_neural() -> None:
    request = TaskRequest(
        task="Why is my PyTorch neural network validation loss unstable after changing the optimizer?",
        supplied_files=("train.py",),
    )
    route = route_task(request)
    plan = create_plan(request)
    assert route.specialization is Specialization.NEURAL
    assert ToolIntent.PYTHON in plan.proposed_tools
    assert "learning objective" in plan.steps[0].objective.lower()


def test_chemical_engineering_task_routes_to_raiq_chem() -> None:
    request = TaskRequest(
        task="Calculate the heat duty for a heat exchanger from the inlet and outlet stream data.",
    )
    route = route_task(request)
    plan = create_plan(request)
    assert route.specialization is Specialization.CHEM
    assert ToolIntent.RETRIEVAL in plan.proposed_tools
    assert ToolIntent.VERIFY in plan.proposed_tools


def test_general_technical_task_uses_fallback_and_complex_plan() -> None:
    request = TaskRequest(
        task="Design an end-to-end production architecture and compare trade-offs for a multi-step technical research system.",
    )
    route = route_task(request)
    assert route.specialization is Specialization.TECHNICAL
    assert select_reasoning_depth(request, route) is ReasoningDepth.COMPLEX
    plan = create_plan(request)
    assert len(plan.steps) == 5
    assert any("trade-offs" in step.objective for step in plan.steps)


def test_agent_trace_records_proposals_but_not_execution() -> None:
    result = RAIQAgent().plan(TaskRequest(task="Implement and test a Python parser."))
    event_types = [event.event_type for event in result.trace.events]
    assert event_types == [
        "task_received",
        "route_selected",
        "reasoning_depth_selected",
        "tool_intents_proposed",
        "execution_blocked",
    ]
    trace_payload = result.trace.to_dict()
    assert trace_payload["events"][-1]["event_type"] == "execution_blocked"
    plan_payload = result.plan.to_dict()
    assert plan_payload["route"]["specialization"] == "RAIQ Code"
    assert "shell" in plan_payload["proposed_tools"]
