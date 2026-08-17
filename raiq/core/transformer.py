"""Stacked RAIQ decoder runtime."""

from __future__ import annotations

from typing import Sequence

import torch.utils.checkpoint
from torch import Tensor, nn

from raiq.core.attention import KVCache
from raiq.core.block import DecoderBlock
from raiq.core.config import ModelConfig
from raiq.core.embeddings import RMSNorm


class Transformer(nn.Module):
    """Configurable stack of RAIQ causal decoder blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        past_key_values: Sequence[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, tuple[KVCache, ...] | None]:
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one entry per decoder layer")

        present_key_values: list[KVCache] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            can_checkpoint = self.config.gradient_checkpointing and self.training and not use_cache
            if can_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    lambda states: block(states, past_key_value=None, use_cache=False)[0],
                    hidden_states,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    past_key_value=past,
                    use_cache=use_cache,
                )
            if use_cache:
                if present is None:
                    raise RuntimeError("use_cache=True requires each block to return a KV cache")
                present_key_values.append(present)

        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(present_key_values) if use_cache else None
