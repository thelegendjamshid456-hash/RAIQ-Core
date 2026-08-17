"""RAIQ decoder-only language model and model metadata helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from raiq.core.attention import KVCache
from raiq.core.config import ModelConfig
from raiq.core.transformer import Transformer


@dataclass
class CausalLMOutput:
    """Outputs from a RAIQ Core forward pass."""

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[KVCache, ...] | None = None


class RAIQModel(nn.Module):
    """From-scratch decoder-only Transformer used by every RAIQ Core configuration."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2.0 * self.config.n_layers)
        for name, parameter in self.named_parameters():
            if name.endswith("attention.out_proj.weight") or name.endswith("mlp.down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        parameters = self.parameters() if trainable_only else self.parameters()
        return sum(parameter.numel() for parameter in parameters if not trainable_only or parameter.requires_grad)

    def parameter_count_millions(self) -> float:
        return self.parameter_count() / 1_000_000

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.config.name,
            "parameter_count": self.parameter_count(),
            "parameter_count_millions": round(self.parameter_count_millions(), 3),
            "architecture": self.config.to_dict(),
        }

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        past_key_values: Sequence[KVCache | None] | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        if input_ids.size(1) <= 0:
            raise ValueError("input_ids must contain at least one token")
        if input_ids.size(1) > self.config.max_seq_len and past_key_values is None:
            raise ValueError("input length exceeds configured max_seq_len")

        hidden_states = self.token_embeddings(input_ids)
        hidden_states, present = self.transformer(
            hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return CausalLMOutput(logits=logits, loss=loss, past_key_values=present)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressively generate tokens using the model's KV cache."""

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        generated = input_ids
        cache: tuple[KVCache, ...] | None = None
        was_training = self.training
        self.eval()
        try:
            for step in range(max_new_tokens):
                model_input = generated if cache is None else generated[:, -1:]
                output = self(model_input, past_key_values=cache, use_cache=True)
                cache = output.past_key_values
                next_logits = output.logits[:, -1, :] / temperature
                if top_k is not None:
                    k = min(top_k, next_logits.size(-1))
                    threshold = torch.topk(next_logits, k=k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                if generated.size(1) >= self.config.max_seq_len:
                    break
        finally:
            self.train(was_training)
        return generated
