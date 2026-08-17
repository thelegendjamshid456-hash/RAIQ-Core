"""A transparent byte-level tokenizer for RAIQ Core smoke experiments.

This tokenizer is deliberately simple: it maps UTF-8 bytes to fixed IDs and reserves
project-defined special tokens. It is not the final 32K–50K technical BPE tokenizer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SPECIAL_TOKENS = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<system>",
    "<user>",
    "<assistant>",
    "<tool>",
    "<result>",
    "<code>",
    "<execute>",
    "<verify>",
    "<error>",
)


@dataclass(frozen=True)
class ByteTokenizerMetadata:
    name: str = "raiq-byte-v1"
    version: int = 1


class ByteTokenizer:
    """Exact UTF-8 byte tokenizer with fixed RAIQ special-token identifiers."""

    byte_offset = len(SPECIAL_TOKENS)
    vocab_size = byte_offset + 256

    def __init__(self, metadata: ByteTokenizerMetadata | None = None) -> None:
        self.metadata = metadata or ByteTokenizerMetadata()
        self.special_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {index: token for token, index in self.special_to_id.items()}

    @property
    def pad_token_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_to_id["<eos>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode Unicode text losslessly through UTF-8 bytes."""

        ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode IDs, replacing non-byte or special IDs according to the requested policy."""

        byte_values: list[int] = []
        special_text: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if not skip_special_tokens:
                    special_text.append(self.id_to_special[token_id])
                continue
            byte_value = token_id - self.byte_offset
            if not 0 <= byte_value < 256:
                if not skip_special_tokens:
                    special_text.append("<unk>")
                continue
            byte_values.append(byte_value)
        decoded = bytes(byte_values).decode("utf-8", errors="replace")
        return "".join(special_text) + decoded

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "byte",
            "name": self.metadata.name,
            "version": self.metadata.version,
            "special_tokens": list(SPECIAL_TOKENS),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte" or tuple(payload.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("unsupported or incompatible RAIQ byte tokenizer file")
        return cls(ByteTokenizerMetadata(name=payload["name"], version=int(payload["version"])))
