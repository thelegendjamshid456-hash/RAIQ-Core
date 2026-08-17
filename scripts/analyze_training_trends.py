"""Write a truthful RAIQ training-trend report from persisted metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from raiq.training.trends import analyze_training_trends


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RAIQ metric trends without altering observations")
    parser.add_argument("--metrics", required=True, help="Path to a metrics.json artifact")
    parser.add_argument("--output", required=True, help="Path to write the JSON trend report")
    parser.add_argument("--post-clip-limit", type=float, default=1.0)
    parser.add_argument("--post-clip-tolerance", type=float, default=1e-5)
    parser.add_argument("--regression-patience", type=int, default=3)
    parser.add_argument("--regression-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    records = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = analyze_training_trends(
        records,
        post_clip_limit=args.post_clip_limit,
        post_clip_tolerance=args.post_clip_tolerance,
        regression_patience=args.regression_patience,
        regression_tolerance=args.regression_tolerance,
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["post_clip_gradient_norm"]["compliant"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
