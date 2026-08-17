"""Residual decoder block for RAIQ Core."""

from __future__ import annotations

from torch import Tensor, nn

from raiq.core.attention import CausalSelfAttention, KVCache
from raiq.core.config import ModelConfig
from raiq.core.embeddings import RMSNorm
from raiq.core.mlp import SwiGLU


class DecoderBlock(nn.Module):
    """Pre-norm causal decoder block with attention and SwiGLU residual branches."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        attention_output, present = self.attention(
            self.attention_norm(hidden_states),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, present
