"""Small, deterministic text-block dataset utilities for RAIQ Core experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from raiq.tokenizer.loader import RAIQTokenizer


class TextBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    """Tokenize a text file and expose fixed next-token prediction windows."""

    def __init__(self, path: str | Path, tokenizer: RAIQTokenizer, sequence_length: int) -> None:
        self.path = Path(path)
        self.sequence_length = sequence_length
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        text = self.path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(token_ids) <= sequence_length:
            raise ValueError(f"{self.path} is too short for sequence_length={sequence_length}")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.block_count = (len(self.tokens) - 1) // sequence_length
        if self.block_count <= 0:
            raise ValueError("dataset contains no complete token blocks")

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < self.block_count:
            raise IndexError(index)
        start = index * self.sequence_length
        inputs = self.tokens[start : start + self.sequence_length]
        labels = self.tokens[start + 1 : start + self.sequence_length + 1]
        return inputs, labels
