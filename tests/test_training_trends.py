from __future__ import annotations

import pytest

from raiq.training.trends import analyze_training_trends


def record(
    step: int,
    train_loss: float,
    clipped_gradient_norm: float,
    validation_perplexity: float | None = None,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "step": step,
        "train_loss": train_loss,
        "clipped_gradient_norm": clipped_gradient_norm,
    }
    if validation_perplexity is not None:
        result["validation_perplexity"] = validation_perplexity
    return result


def test_trend_analysis_reports_genuine_improvement_without_monotonic_forcing() -> None:
    metrics = [
        record(1, 8.0, 0.999, 200.0),
        record(2, 8.2, 1.0),
        record(3, 7.7, 0.998, 190.0),
        record(4, 7.8, 0.997),
        record(5, 7.5, 0.999, 192.0),
    ]
    report = analyze_training_trends(metrics)

    assert report["train_loss"]["net_decreased"] is True
    assert report["post_clip_gradient_norm"]["compliant"] is True
    assert report["validation_perplexity"]["best_improved_from_initial"] is True
    assert report["validation_perplexity"]["persistent_regression"] is False
    assert report["observation_policy"].startswith("reports raw records")


def test_trend_analysis_labels_legacy_metrics_without_post_clip_telemetry() -> None:
    metrics = [
        {"step": 1, "train_loss": 8.0, "validation_perplexity": 200.0},
        {"step": 2, "train_loss": 7.9, "validation_perplexity": 190.0},
    ]
    report = analyze_training_trends(metrics)

    assert report["post_clip_gradient_norm"]["available"] is False
    assert report["post_clip_gradient_norm"]["compliant"] is None
    assert report["post_clip_gradient_norm"]["missing_steps"] == [1, 2]


def test_trend_analysis_flags_post_clip_limit_violation() -> None:
    metrics = [
        record(1, 8.0, 1.0, 200.0),
        record(2, 7.9, 1.02, 190.0),
    ]
    report = analyze_training_trends(metrics)

    assert report["post_clip_gradient_norm"]["compliant"] is False
    assert report["post_clip_gradient_norm"]["violations"] == [
        {"step": 2, "clipped_gradient_norm": 1.02}
    ]


def test_trend_analysis_flags_sustained_validation_regression() -> None:
    metrics = [
        record(1, 8.0, 1.0, 200.0),
        record(2, 7.8, 1.0, 190.0),
        record(3, 7.7, 1.0, 200.0),
        record(4, 7.6, 1.0, 205.0),
        record(5, 7.5, 1.0, 210.0),
    ]
    report = analyze_training_trends(metrics, regression_tolerance=0.01)

    assert report["validation_perplexity"]["persistent_regression"] is True


def test_trend_analysis_rejects_missing_validation_evidence() -> None:
    with pytest.raises(ValueError, match="validation_perplexity"):
        analyze_training_trends([record(1, 8.0, 1.0)])
