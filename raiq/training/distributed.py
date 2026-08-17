"""Distributed-training primitives for future RAIQ production pretraining.

The module is inert for single-process runs and fails explicitly rather than silently
pretending to distribute work when a requested backend or device is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    """Immutable description of the current training process topology."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize_distributed(requested_device: str = "cuda") -> DistributedContext:
    """Initialize a process group from standard torchrun environment variables.

    A world size of one returns a normal single-process context. A multi-process CUDA
    run uses NCCL; a multi-process CPU run uses Gloo for local smoke validation only.
    """

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank or world size environment")

    device_type = torch.device(requested_device).type
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if world_size == 1:
        return DistributedContext(rank=0, local_rank=0, world_size=1, device=torch.device(requested_device))
    if not dist.is_available():
        raise RuntimeError("PyTorch distributed support is unavailable")

    backend = "nccl" if device_type == "cuda" else "gloo"
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        initialized_here = True
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    """Average a scalar or tensor across ranks, returning a detached aggregate."""

    aggregate = value.detach().clone()
    if context.enabled:
        dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        aggregate /= context.world_size
    return aggregate


def synchronize(context: DistributedContext) -> None:
    """Synchronize all ranks when a distributed process group is active."""

    if context.enabled:
        dist.barrier()


def cleanup_distributed(context: DistributedContext) -> None:
    """Destroy only the process group created by this RAIQ process."""

    if context.enabled and context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()
