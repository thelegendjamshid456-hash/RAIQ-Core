"""Transparent specialization routing and complexity estimation for RAIQ Agent."""

from __future__ import annotations

import re
from collections.abc import Iterable

from raiq.agent.types import ReasoningDepth, RouteDecision, Specialization, TaskRequest

DOMAIN_KEYWORDS: dict[Specialization, tuple[str, ...]] = {
    Specialization.CODE: (
        "api", "bash", "bug", "c++", "code", "compile", "debug", "exception",
        "fastapi", "function", "git", "java", "javascript", "program", "python",
        "refactor", "rust", "sql", "stack trace", "test", "typescript",
    ),
    Specialization.NEURAL: (
        "activation", "backprop", "checkpoint", "dataset", "deep learning", "embedding",
        "gradient", "hyperparameter", "loss", "machine learning", "model training", "neural",
        "optimizer", "pytorch", "tensor", "transformer", "validation",
    ),
    Specialization.CHEM: (
        "absorption", "cstr", "distillation", "energy balance", "enthalpy", "evaporation",
        "fluid mechanics", "heat duty", "heat exchanger", "mass balance", "membrane", "pfr",
        "process", "reaction", "reactor", "separation", "thermodynamics", "transfer",
    ),
}

COMPLEXITY_SIGNALS = (
    "architecture", "benchmark", "compare", "design", "end-to-end", "integrate", "multi-step",
    "optimize", "pipeline", "production", "research", "system", "trade-off",
)
MODERATE_SIGNALS = (
    "analyse", "analyze", "calculate", "debug", "derive", "explain", "implement", "investigate",
    "solve", "test", "validate",
)


def _normalized_text(request: TaskRequest) -> str:
    return f"{request.task}\n{request.context}".lower()


def _contains(text: str, phrase: str) -> bool:
    if re.search(r"[^a-z0-9+]", phrase):
        return phrase in text
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _matches(text: str, candidates: Iterable[str]) -> list[str]:
    return [candidate for candidate in candidates if _contains(text, candidate)]


def route_task(request: TaskRequest) -> RouteDecision:
    """Select a specialization using scored, retained keyword evidence."""

    text = _normalized_text(request)
    evidence_by_domain = {
        specialization: _matches(text, keywords)
        for specialization, keywords in DOMAIN_KEYWORDS.items()
    }
    scores = {specialization.value: len(matches) for specialization, matches in evidence_by_domain.items()}
    best_specialization = max(DOMAIN_KEYWORDS, key=lambda item: scores[item.value])
    best_score = scores[best_specialization.value]
    if best_score == 0:
        return RouteDecision(
            specialization=Specialization.TECHNICAL,
            confidence=0.35,
            evidence=("No specialization-specific terms matched; using general technical routing.",),
            scores=scores,
        )

    total_matches = sum(scores.values())
    confidence = min(0.95, 0.5 + 0.1 * best_score + 0.05 * (best_score / max(1, total_matches)))
    return RouteDecision(
        specialization=best_specialization,
        confidence=round(confidence, 3),
        evidence=tuple(evidence_by_domain[best_specialization]),
        scores=scores,
    )


def select_reasoning_depth(request: TaskRequest, route: RouteDecision) -> ReasoningDepth:
    """Use conservative complexity signals to select a bounded plan depth."""

    text = _normalized_text(request)
    complexity_hits = _matches(text, COMPLEXITY_SIGNALS)
    moderate_hits = _matches(text, MODERATE_SIGNALS)
    multi_clause = len(re.findall(r"(?:\band\b|;|\n|\bthen\b)", text)) >= 2
    if len(complexity_hits) >= 2 or (complexity_hits and multi_clause) or len(request.task.split()) >= 45:
        return ReasoningDepth.COMPLEX
    if moderate_hits or route.specialization != Specialization.TECHNICAL or multi_clause:
        return ReasoningDepth.MODERATE
    return ReasoningDepth.SIMPLE
