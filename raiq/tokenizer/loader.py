"""Load versioned RAIQ tokenizer artifacts without depending on a specific tokenizer class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from raiq.tokenizer.bpe_tokenizer import BytePairTokenizer
from raiq.tokenizer.byte_tokenizer import ByteTokenizer


class RAIQTokenizer(Protocol):
    """Minimal tokenizer behavior required by the current RAIQ data and inference paths."""

    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str: ...

    def to_dict(self) -> dict[str, object]: ...

    def save(self, path: str | Path) -> Path: ...


def load_tokenizer(path: str | Path) -> RAIQTokenizer:
    """Load a supported tokenizer based on its self-describing JSON type field."""

    tokenizer_path = Path(path)
    payload = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    tokenizer_type = payload.get("type")
    if tokenizer_type == "byte":
        return ByteTokenizer.load(tokenizer_path)
    if tokenizer_type == "byte_pair":
        return BytePairTokenizer.load(tokenizer_path)
    raise ValueError(f"unsupported RAIQ tokenizer type in {tokenizer_path}: {tokenizer_type!r}")
