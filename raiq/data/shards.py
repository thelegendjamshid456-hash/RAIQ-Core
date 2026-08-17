"""Sharded corpus manifest checks for scalable RAIQ pretraining data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sharded_manifest(path: str | Path) -> dict[str, Any]:
    """Fail closed unless every declared shard exists with its expected hash and byte count."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.parent.parent
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("sharded corpus manifest must list at least one shard")
    names: set[str] = set()
    for shard in shards:
        name = shard.get("name")
        if not isinstance(name, str) or name in names:
            raise ValueError("every shard must have a unique name")
        names.add(name)
        shard_path = root / shard["path"]
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        if shard_path.stat().st_size != int(shard["bytes"]):
            raise ValueError(f"byte-count mismatch for {shard_path}")
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {shard_path}")
    return payload
