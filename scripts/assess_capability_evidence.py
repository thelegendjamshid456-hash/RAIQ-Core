"""Aggregate RAIQ coding and reasoning benchmark evidence without unsupported parity claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_DOMAINS = {"coding", "reasoning"}
REQUIRED_FIELDS = {"domain", "protocol_id", "score", "reference_score", "passed"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [json.loads(Path(item).read_text(encoding="utf-8")) for item in args.results]
    valid = [isinstance(record, dict) and REQUIRED_FIELDS.issubset(record) for record in records]
    domains = {record.get("domain") for record in records if isinstance(record, dict)}
    same_protocol = len({record.get("protocol_id") for record in records if isinstance(record, dict)}) == 1
    parity_passed = bool(all(valid) and REQUIRED_DOMAINS.issubset(domains) and same_protocol and all(
        record["passed"] and record["score"] >= record["reference_score"] for record in records
    ))
    result = {
        "passed": parity_passed,
        "claim": "parity_established" if parity_passed else "parity_not_established",
        "required_domains": sorted(REQUIRED_DOMAINS),
        "same_protocol": same_protocol,
        "results": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not parity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
