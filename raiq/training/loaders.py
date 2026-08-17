"""Rank-aware data loading for RAIQ distributed pretraining."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from raiq.training.distributed import DistributedContext


def build_training_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    context: DistributedContext,
    seed: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create a deterministic DataLoader and optional rank-partitioning sampler."""

    sampler = None
    if context.enabled:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(seed + context.rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        drop_last=False,
    )
    return loader, sampler
