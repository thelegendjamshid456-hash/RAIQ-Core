"""Configuration objects for scalable RAIQ decoder-only Transformer variants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by all RAIQ Core model sizes."""

    name: str
    vocab_size: int
    max_seq_len: int
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.max_seq_len <= 0:
            raise ValueError("vocab_size and max_seq_len must be positive")
        if self.n_layers <= 0 or self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("n_layers, d_model, and d_ff must be positive")
        if self.n_heads <= 0 or self.n_kv_heads <= 0:
            raise ValueError("n_heads and n_kv_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducibility and optimization settings for a RAIQ training run."""

    seed: int = 1337
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 4
    grad_accumulation_steps: int = 1
    max_steps: int = 100
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 10
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    eval_interval: int = 20
    save_interval: int = 20
    log_interval: int = 5

    def __post_init__(self) -> None:
        positive_ints = (
            "batch_size",
            "grad_accumulation_steps",
            "max_steps",
            "eval_interval",
            "save_interval",
            "log_interval",
        )
        for name in positive_ints:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0.0 or self.min_learning_rate < 0.0:
            raise ValueError("learning rates must be non-negative, with learning_rate positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    """Versioned identifiers and local paths for a training dataset."""

    dataset_name: str = "unassigned"
    dataset_version: str | int = "unassigned"
    tokenizer_name: str = "unassigned"
    tokenizer_version: str | int = "unassigned"
    train_path: str | None = None
    validation_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration persisted with every RAIQ Core experiment."""

    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
        }


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a RAIQ configuration YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "model" not in raw:
        raise ValueError(f"{config_path} must define a top-level 'model' mapping")
    return RunConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw.get("training", {})),
        data=DataConfig(**raw.get("data", {})),
    )
