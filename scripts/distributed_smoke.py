"""Run under torchrun to verify RAIQ distributed primitives before production use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from raiq.training.distributed import cleanup_distributed, initialize_distributed, reduce_mean, synchronize


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAIQ distributed smoke test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    context = initialize_distributed("cpu")
    try:
        local_value = torch.tensor(float(context.rank + 1), device=context.device)
        mean_value = float(reduce_mean(local_value, context).cpu())
        synchronize(context)
        passed = context.world_size >= 2 and abs(mean_value - (context.world_size + 1) / 2) < 1e-6
        if context.is_primary:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "passed": passed,
                        "backend": "gloo",
                        "world_size": context.world_size,
                        "mean_value": mean_value,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        if not passed:
            raise SystemExit(2)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
