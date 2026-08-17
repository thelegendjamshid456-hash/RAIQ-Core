from __future__ import annotations

import torch
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from raiq.training.distributed import DistributedContext
from raiq.training.loaders import build_training_loader


def test_single_process_loader_uses_standard_shuffle() -> None:
    dataset = TensorDataset(torch.arange(12))
    context = DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    loader, sampler = build_training_loader(dataset, batch_size=3, context=context, seed=7)
    assert sampler is None
    assert len(list(loader)) == 4


def test_distributed_loader_uses_rank_partitioning_sampler() -> None:
    dataset = TensorDataset(torch.arange(12))
    context = DistributedContext(rank=1, local_rank=1, world_size=2, device=torch.device("cpu"))
    loader, sampler = build_training_loader(dataset, batch_size=2, context=context, seed=7)
    assert isinstance(sampler, DistributedSampler)
    assert sampler.rank == 1
    assert sampler.num_replicas == 2
    assert len(list(loader)) == 3
