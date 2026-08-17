"""Deterministic training utilities for RAIQ Core experiments."""

from __future__ import annotations

import math
import random

import torch


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch generators for repeatable experiments."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicitly requested device, failing rather than silently falling back."""

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA-capable PyTorch device is available")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve a configured floating point type with conservative CPU checks."""

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU training is not supported by this baseline")
    return dtype


def warmup_cosine_lr(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Return a warmup-cosine learning rate for a zero-indexed optimizer step."""

    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)
