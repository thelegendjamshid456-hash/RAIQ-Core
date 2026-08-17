"""Portable checkpoint helpers for reproducible RAIQ Core training runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_config: dict[str, Any],
    metadata: dict[str, Any],
) -> Path:
    """Atomically save all information necessary to continue a training run."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "run_config": run_config,
        "metadata": metadata,
        "torch_version": torch.__version__,
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a RAIQ Core checkpoint and return the recorded run information."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported RAIQ checkpoint format version")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
