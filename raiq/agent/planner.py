"""Structured, non-executing plan construction for RAIQ Agent."""

from __future__ import annotations

from raiq.agent.router import route_task, select_reasoning_depth
from raiq.agent.types import AgentPlan, PlanStep, ReasoningDepth, Specialization, TaskRequest, ToolIntent


def _step(
    order: int,
    objective: str,
    tools: tuple[ToolIntent, ...] = (),
    verification: str = "Review the result against the stated constraints and expected output.",
) -> PlanStep:
    return PlanStep(order=order, objective=objective, proposed_tools=tools, verification=verification)


def _domain_steps(specialization: Specialization, has_files: bool) -> list[PlanStep]:
    file_tools = (ToolIntent.FILES,) if has_files else ()
    if specialization is Specialization.CODE:
        return [
            _step(1, "Identify the requested program behavior, current failure mode, inputs, outputs, and constraints.", file_tools),
            _step(2, "Trace the relevant control flow and isolate the smallest implementation or debugging change.", file_tools),
            _step(3, "Propose an implementation and targeted tests that demonstrate the requested behavior.", (ToolIntent.FILES, ToolIntent.SHELL)),
            _step(4, "Verify compilation or execution results, regression coverage, and error handling.", (ToolIntent.SHELL, ToolIntent.VERIFY)),
        ]
    if specialization is Specialization.NEURAL:
        return [
            _step(1, "Define the learning objective, data split, output metric, and reproducibility constraints.", file_tools),
            _step(2, "Inspect architecture, optimization, normalization, and data-pipeline signals for likely failure points.", (ToolIntent.FILES, ToolIntent.PYTHON)),
            _step(3, "Propose a minimal controlled experiment and record comparable baseline metrics.", (ToolIntent.PYTHON, ToolIntent.VERIFY)),
            _step(4, "Validate the diagnosis with held-out metrics and guard against leakage or unstable training.", (ToolIntent.PYTHON, ToolIntent.VERIFY)),
        ]
    if specialization is Specialization.CHEM:
        return [
            _step(1, "State the system boundary, assumptions, known streams or operating data, units, and requested quantity.", file_tools),
            _step(2, "Select the governing material, energy, transport, reaction, or separation relationships.", (ToolIntent.RETRIEVAL,)),
            _step(3, "Carry out a transparent calculation with unit conversions and intermediate checks.", (ToolIntent.PYTHON,)),
            _step(4, "Verify conservation, dimensions, physical plausibility, and stated assumptions before reporting a result.", (ToolIntent.VERIFY,)),
        ]
    return [
        _step(1, "Clarify the technical objective, supplied evidence, constraints, and success criteria.", file_tools),
        _step(2, "Select an appropriate analytical or implementation method and identify missing information.", (ToolIntent.RETRIEVAL,)),
        _step(3, "Develop a concise, checkable solution path with explicit assumptions.", (ToolIntent.PYTHON,)),
        _step(4, "Verify calculations, sources, and consistency with the requested outcome.", (ToolIntent.VERIFY,)),
    ]


def _proposed_tools(steps: list[PlanStep]) -> tuple[ToolIntent, ...]:
    ordered: list[ToolIntent] = []
    for step in steps:
        for tool in step.proposed_tools:
            if tool not in ordered:
                ordered.append(tool)
    return tuple(ordered)


def create_plan(request: TaskRequest) -> AgentPlan:
    """Classify a task and return a structured, inspectable plan without taking action."""

    route = route_task(request)
    reasoning_depth = select_reasoning_depth(request, route)
    steps = _domain_steps(route.specialization, bool(request.supplied_files))
    if reasoning_depth is ReasoningDepth.COMPLEX:
        steps.insert(
            -1,
            _step(
                len(steps),
                "Compare feasible approaches, trade-offs, dependencies, and failure modes before committing to an implementation path.",
                (ToolIntent.RETRIEVAL, ToolIntent.VERIFY),
                "Record why the selected approach best satisfies the stated constraints.",
            ),
        )
    normalized_steps = tuple(
        PlanStep(
            order=index,
            objective=step.objective,
            proposed_tools=step.proposed_tools,
            verification=step.verification,
        )
        for index, step in enumerate(steps, start=1)
    )
    limitations = (
        "This phase proposes intents only; it does not execute files, code, shell commands, search, or retrieval.",
        "Routing is transparent keyword-based scaffolding until a trained RAIQ Core model can be evaluated as a router.",
        "All high-consequence technical outputs require domain-specific verification before use.",
    )
    return AgentPlan(
        request=request,
        route=route,
        reasoning_depth=reasoning_depth,
        steps=normalized_steps,
        proposed_tools=_proposed_tools(list(normalized_steps)),
        limitations=limitations,
    )
