"""Embedding, normalization, and rotary-position components for RAIQ Core."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by the RAIQ decoder blocks."""

    def __init__(self, dimension: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps).to(dtype=values.dtype)
        return normalized * self.weight.to(dtype=values.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE cache and application utilities for even-sized attention heads."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(
        self,
        sequence_length: int,
        *,
        offset: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(offset, offset + sequence_length, device=device)
        freqs = torch.outer(positions.float(), self.inv_freq.to(device=device))
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)

    @staticmethod
    def apply_rotary(values: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply cached cosines and sines to a [batch, time, heads, dim] tensor."""

        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        even = values[..., 0::2]
        odd = values[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(start_dim=-2)
