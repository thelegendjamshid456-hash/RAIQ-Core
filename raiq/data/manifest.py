"""Versioned corpus-manifest validation for reproducible RAIQ training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_corpus_manifest(path: str | Path) -> dict[str, Any]:
    """Validate every listed corpus split and return the immutable manifest payload."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("corpus_id"):
        raise ValueError("corpus manifest must contain a corpus_id")
    root = manifest_path.parent.parent.parent
    splits = payload.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("corpus manifest must contain at least one split")
    for split in splits:
        declared_path = Path(split["path"])
        split_path = declared_path if declared_path.is_absolute() else root / declared_path
        if not split_path.is_file():
            raise FileNotFoundError(f"manifest split does not exist: {split_path}")
        if split_path.stat().st_size != int(split["bytes"]):
            raise ValueError(f"manifest byte count mismatch: {split_path}")
        if _sha256(split_path) != split["sha256"]:
            raise ValueError(f"manifest SHA-256 mismatch: {split_path}")
    return payload
