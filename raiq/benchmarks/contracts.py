"""Validation of benchmark contracts used for RAIQ capability evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_DOMAINS = {"code", "neural", "chem", "reasoning"}


def load_benchmark_contract(path: str | Path) -> dict[str, Any]:
    """Fail closed unless a benchmark declares held-out scoring and contamination controls."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"benchmark_id", "domain", "split", "scoring", "pass_threshold", "contamination_policy"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"benchmark contract is missing fields: {missing}")
    if payload["domain"] not in REQUIRED_DOMAINS:
        raise ValueError("benchmark domain must be code, neural, chem, or reasoning")
    if payload["split"] != "held_out":
        raise ValueError("capability benchmarks must be held_out")
    if not isinstance(payload["pass_threshold"], (int, float)):
        raise ValueError("pass_threshold must be numeric")
    if not payload["contamination_policy"]:
        raise ValueError("contamination_policy must be declared")
    return payload
