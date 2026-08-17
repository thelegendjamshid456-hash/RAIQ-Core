"""Print validated RAIQ model configuration metadata and exact parameter counts."""

from __future__ import annotations

import argparse
import json

from raiq.core import RAIQModel, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RAIQ Core configuration")
    parser.add_argument("config", help="Path to configuration YAML")
    args = parser.parse_args()
    run_config = load_config(args.config)
    model = RAIQModel(run_config.model)
    print(json.dumps(model.metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
