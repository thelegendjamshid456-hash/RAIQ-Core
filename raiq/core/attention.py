"""Causal self-attention with RoPE and optional KV caching for RAIQ Core."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from raiq.core.config import ModelConfig
from raiq.core.embeddings import RotaryEmbedding

KVCache = tuple[Tensor, Tensor]


class CausalSelfAttention(nn.Module):
    """Decoder self-attention supporting grouped KV heads and incremental decoding."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_group_size = config.kv_group_size
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, KVCache | None]:
        batch_size, query_length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value[0].size(2)

        query = self.q_proj(hidden_states).view(batch_size, query_length, self.n_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch_size, query_length, self.n_kv_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch_size, query_length, self.n_kv_heads, self.head_dim)

        cos, sin = self.rope.cos_sin(
            query_length,
            offset=past_length,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        query = RotaryEmbedding.apply_rotary(query, cos, sin).transpose(1, 2)
        key = RotaryEmbedding.apply_rotary(key, cos, sin).transpose(1, 2)
        value = value.transpose(1, 2)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)

        present = (key, value) if use_cache else None
        if self.kv_group_size > 1:
            key = key.repeat_interleave(self.kv_group_size, dim=1)
            value = value.repeat_interleave(self.kv_group_size, dim=1)

        key_length = key.size(2)
        # Fast built-in causal masking is correct only when query and key lengths match.
        if past_length == 0:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            q_positions = torch.arange(past_length, past_length + query_length, device=query.device)
            k_positions = torch.arange(key_length, device=query.device)
            causal_mask = k_positions[None, :] <= q_positions[:, None]
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=causal_mask[None, None, :, :],
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=1.0 / math.sqrt(self.head_dim),
            )

        attended = attended.transpose(1, 2).contiguous().view(batch_size, query_length, -1)
        return self.out_proj(attended), present
