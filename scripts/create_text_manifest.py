"""Create a versioned SHA-256 manifest for explicit UTF-8 train/validation text files."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def describe(path: Path, root: Path) -> dict[str, object]:
    payload = path.read_bytes()
    path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--source-reference", required=True)
    args = parser.parse_args()

    train = Path(args.train).resolve()
    validation = Path(args.validation).resolve()
    if train == validation:
        raise SystemExit("train and validation must be different files")
    for path in (train, validation):
        if not path.is_file():
            raise SystemExit(f"missing dataset file: {path}")

    root = Path.cwd().resolve()
    manifest = {
        "corpus_id": args.corpus_id,
        "dataset_version": args.dataset_version,
        "purpose": "Reproducible RAIQ text pretraining input manifest.",
        "production_eligible": False,
        "source_reference": args.source_reference,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits": [
            {"name": "train", **describe(train, root)},
            {"name": "validation", **describe(validation, root)},
        ],
        "preprocessing": {
            "format": "UTF-8 plain text",
            "transformation": "No transformation; source bytes are hashed as provided.",
            "evaluation_data_excluded_from_training": True,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
