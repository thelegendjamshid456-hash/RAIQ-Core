"""SwiGLU feed-forward network for RAIQ decoder blocks."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from raiq.core.config import ModelConfig


class SwiGLU(nn.Module):
    """Bias-free SwiGLU MLP with configurable intermediate width."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
