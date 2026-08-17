"""Trainable byte-pair tokenizer for RAIQ Core technical-corpus experiments."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from raiq.tokenizer.byte_tokenizer import SPECIAL_TOKENS


@dataclass(frozen=True)
class BPETokenizerMetadata:
    """Versioned identity for a trained RAIQ BPE tokenizer."""

    name: str = "raiq-technical-bpe"
    version: str = "v1"
    training_corpus_id: str = "unassigned"


class BytePairTokenizer:
    """Transparent byte-level BPE tokenizer with deterministic merge training.

    The base alphabet is all UTF-8 bytes. Training repeatedly merges the most frequent
    adjacent token pair. This implementation favors inspectability and reproducibility
    over optimized large-corpus throughput; production-scale tokenizer training can use
    the same serialized merge format with a faster trainer later.
    """

    byte_offset = len(SPECIAL_TOKENS)
    base_vocab_size = byte_offset + 256

    def __init__(
        self,
        *,
        merges: Sequence[tuple[int, int, int]] = (),
        metadata: BPETokenizerMetadata | None = None,
    ) -> None:
        self.metadata = metadata or BPETokenizerMetadata()
        self.special_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_special = {index: token for token, index in self.special_to_id.items()}
        self.merges = tuple((int(left), int(right), int(merged)) for left, right, merged in merges)
        self._merge_lookup = {(left, right): merged for left, right, merged in self.merges}
        if len(self._merge_lookup) != len(self.merges):
            raise ValueError("each BPE merge pair must be unique")
        self._id_to_bytes: dict[int, bytes] = {
            self.byte_offset + byte: bytes((byte,)) for byte in range(256)
        }
        next_id = self.base_vocab_size
        for left, right, merged in self.merges:
            if merged != next_id:
                raise ValueError("BPE merge IDs must be contiguous and ordered")
            if left not in self._id_to_bytes or right not in self._id_to_bytes:
                raise ValueError("BPE merge references an unknown token")
            self._id_to_bytes[merged] = self._id_to_bytes[left] + self._id_to_bytes[right]
            next_id += 1

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)

    @property
    def pad_token_id(self) -> int:
        return self.special_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.special_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.special_to_id["<eos>"]

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int,
        min_pair_frequency: int = 2,
        metadata: BPETokenizerMetadata | None = None,
    ) -> "BytePairTokenizer":
        """Train deterministic BPE merges from UTF-8 text samples."""

        if vocab_size < cls.base_vocab_size:
            raise ValueError(f"vocab_size must be at least {cls.base_vocab_size}")
        if min_pair_frequency < 1:
            raise ValueError("min_pair_frequency must be positive")
        sequences = [
            [byte + cls.byte_offset for byte in text.encode("utf-8")]
            for text in texts
            if text
        ]
        if not sequences:
            raise ValueError("cannot train a tokenizer from an empty corpus")

        merges: list[tuple[int, int, int]] = []
        next_id = cls.base_vocab_size
        while next_id < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for sequence in sequences:
                pair_counts.update(zip(sequence, sequence[1:]))
            if not pair_counts:
                break
            pair, frequency = min(pair_counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_pair_frequency:
                break
            merges.append((pair[0], pair[1], next_id))
            sequences = [cls._merge_pair(sequence, pair, next_id) for sequence in sequences]
            next_id += 1
        return cls(merges=merges, metadata=metadata)

    @staticmethod
    def _merge_pair(sequence: Sequence[int], pair: tuple[int, int], merged_id: int) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and sequence[index] == pair[0] and sequence[index + 1] == pair[1]:
                merged.append(merged_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1
        return merged

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode UTF-8 text using base byte tokens followed by the learned merge sequence."""

        ids = [byte + self.byte_offset for byte in text.encode("utf-8")]
        for left, right, merged in self.merges:
            ids = self._merge_pair(ids, (left, right), merged)
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a token sequence, retaining special-token text only when requested."""

        chunks: list[bytes] = []
        special_text: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self.id_to_special:
                if not skip_special_tokens:
                    special_text.append(self.id_to_special[token_id])
                continue
            token_bytes = self._id_to_bytes.get(token_id)
            if token_bytes is None:
                if not skip_special_tokens:
                    special_text.append("<unk>")
                continue
            chunks.append(token_bytes)
        return "".join(special_text) + b"".join(chunks).decode("utf-8", errors="replace")

    def compression_stats(self, text: str) -> dict[str, float | int]:
        """Measure token compression for a supplied sample without claiming language quality."""

        byte_count = len(text.encode("utf-8"))
        token_count = len(self.encode(text))
        return {
            "bytes": byte_count,
            "tokens": token_count,
            "tokens_per_byte": token_count / max(1, byte_count),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "byte_pair",
            "name": self.metadata.name,
            "version": self.metadata.version,
            "training_corpus_id": self.metadata.training_corpus_id,
            "special_tokens": list(SPECIAL_TOKENS),
            "merges": [list(merge) for merge in self.merges],
            "vocab_size": self.vocab_size,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "BytePairTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "byte_pair":
            raise ValueError("tokenizer file is not a RAIQ byte-pair tokenizer")
        if tuple(payload.get("special_tokens", ())) != SPECIAL_TOKENS:
            raise ValueError("incompatible RAIQ BPE special-token set")
        tokenizer = cls(
            merges=[tuple(merge) for merge in payload.get("merges", ())],
            metadata=BPETokenizerMetadata(
                name=str(payload["name"]),
                version=str(payload["version"]),
                training_corpus_id=str(payload.get("training_corpus_id", "unassigned")),
            ),
        )
        if tokenizer.vocab_size != int(payload["vocab_size"]):
            raise ValueError("serialized BPE vocabulary size does not match its merges")
        return tokenizer
