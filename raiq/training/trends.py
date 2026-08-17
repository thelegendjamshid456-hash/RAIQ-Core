"""Truthful post-run trend analysis for RAIQ training metrics.

The analyzer reports observed improvements and regressions. It intentionally does not
smooth, reorder, or modify metric values to manufacture a monotonic trajectory.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


DEFAULT_POST_CLIP_TOLERANCE = 1e-5
DEFAULT_REGRESSION_PATIENCE = 3
DEFAULT_REGRESSION_TOLERANCE = 0.01


def _finite_float(value: object, field: str, step: object) -> float:
    """Return a finite metric value or fail with contextual evidence."""

    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"non-finite or missing {field} at step {step}")
    return float(value)


def analyze_training_trends(
    records: Sequence[dict[str, Any]],
    *,
    post_clip_limit: float = 1.0,
    post_clip_tolerance: float = DEFAULT_POST_CLIP_TOLERANCE,
    regression_patience: int = DEFAULT_REGRESSION_PATIENCE,
    regression_tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> dict[str, Any]:
    """Summarize genuine training trends and flag persistent validation regression.

    A net improvement is not confused with strict point-by-point monotonicity. The
    latter is neither required nor manufactured by this analyzer.
    """

    if not records:
        raise ValueError("metrics must contain at least one record")
    if regression_patience <= 0:
        raise ValueError("regression_patience must be positive")
    if post_clip_limit <= 0.0 or post_clip_tolerance < 0.0:
        raise ValueError("post-clip limit and tolerance must be non-negative")
    if regression_tolerance < 0.0:
        raise ValueError("regression_tolerance must be non-negative")

    training_losses: list[tuple[int, float]] = []
    post_clip_violations: list[dict[str, float | int]] = []
    missing_post_clip_steps: list[int] = []
    validation: list[dict[str, float | int]] = []

    for record in records:
        step = record.get("step")
        if not isinstance(step, int) or step <= 0:
            raise ValueError(f"invalid step: {step}")
        train_loss = _finite_float(record.get("train_loss"), "train_loss", step)
        training_losses.append((step, train_loss))

        if "clipped_gradient_norm" not in record:
            missing_post_clip_steps.append(step)
        else:
            clipped = _finite_float(
                record.get("clipped_gradient_norm"), "clipped_gradient_norm", step
            )
            if clipped > post_clip_limit + post_clip_tolerance:
                post_clip_violations.append({"step": step, "clipped_gradient_norm": clipped})

        if "validation_perplexity" in record:
            validation_ppl = _finite_float(
                record.get("validation_perplexity"), "validation_perplexity", step
            )
            validation.append({"step": step, "validation_perplexity": validation_ppl})

    if not validation:
        raise ValueError("metrics must contain at least one validation_perplexity record")

    best_index = min(range(len(validation)), key=lambda index: validation[index]["validation_perplexity"])
    best_validation = validation[best_index]
    trailing_validation = validation[best_index + 1 :]
    persistent_regression = False
    if len(trailing_validation) >= regression_patience:
        recent = trailing_validation[-regression_patience:]
        regression_floor = best_validation["validation_perplexity"] * (1.0 + regression_tolerance)
        persistent_regression = all(
            item["validation_perplexity"] > regression_floor for item in recent
        )

    initial_train_step, initial_train_loss = training_losses[0]
    final_train_step, final_train_loss = training_losses[-1]
    initial_validation = validation[0]
    final_validation = validation[-1]
    return {
        "schema_version": 1,
        "observation_policy": "reports raw records without smoothing, reordering, or monotonic forcing",
        "train_loss": {
            "initial_step": initial_train_step,
            "initial": initial_train_loss,
            "final_step": final_train_step,
            "final": final_train_loss,
            "net_decreased": final_train_loss < initial_train_loss,
        },
        "post_clip_gradient_norm": {
            "available": not missing_post_clip_steps,
            "observed_record_count": len(records) - len(missing_post_clip_steps),
            "missing_steps": missing_post_clip_steps,
            "limit": post_clip_limit,
            "tolerance": post_clip_tolerance,
            "compliant": None if missing_post_clip_steps else not post_clip_violations,
            "violations": post_clip_violations,
        },
        "validation_perplexity": {
            "initial_step": initial_validation["step"],
            "initial": initial_validation["validation_perplexity"],
            "best_step": best_validation["step"],
            "best": best_validation["validation_perplexity"],
            "final_step": final_validation["step"],
            "final": final_validation["validation_perplexity"],
            "best_improved_from_initial": (
                best_validation["validation_perplexity"] < initial_validation["validation_perplexity"]
            ),
            "persistent_regression": persistent_regression,
            "regression_patience": regression_patience,
            "regression_tolerance": regression_tolerance,
        },
    }
