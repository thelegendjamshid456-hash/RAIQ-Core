"""Portable checkpoint helpers for reproducible RAIQ Core training runs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    """Capture Python and PyTorch random states needed for deterministic resume."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a previously captured random state when resuming an experiment."""

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


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
        "rng_state": capture_rng_state(),
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
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a RAIQ Core checkpoint and return the recorded run information."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") not in {1, CHECKPOINT_FORMAT_VERSION}:
        raise ValueError("unsupported RAIQ checkpoint format version")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if restore_rng:
        if "rng_state" not in payload:
            raise ValueError("checkpoint does not contain the RNG state required for deterministic resume")
        restore_rng_state(payload["rng_state"])
    return payload
