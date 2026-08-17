"""Command-line interface for inspectable, non-executing RAIQ Agent planning."""

from __future__ import annotations

import argparse
import json

from raiq.agent.agent import RAIQAgent
from raiq.agent.types import TaskRequest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an auditable RAIQ Agent plan without executing tools")
    parser.add_argument("--task", required=True, help="Technical task to classify and plan")
    parser.add_argument("--context", default="", help="Optional technical context")
    parser.add_argument("--file", action="append", default=[], help="Name of a supplied file; may be repeated")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    request = TaskRequest(task=args.task, context=args.context, supplied_files=tuple(args.file))
    result = RAIQAgent().plan(request)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
