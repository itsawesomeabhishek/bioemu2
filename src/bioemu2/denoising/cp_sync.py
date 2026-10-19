"""Sampling operations that synchronize only during context-parallel sampling."""

import torch
import torch.distributed as dist

from bioemu2.denoising.augmentation import compute_random_augmentation


def is_distributed_sampling() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _random_augmentation_broadcast(
    multiplicity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random rotation + translation on rank 0, broadcast to all ranks."""
    if is_rank0():
        R, tr = compute_random_augmentation(multiplicity, device=device, dtype=dtype)
    else:
        R = torch.empty(multiplicity, 3, 3, device=device, dtype=dtype)
        tr = torch.empty(multiplicity, 1, 3, device=device, dtype=dtype)
    broadcast_from_rank0(R)
    broadcast_from_rank0(tr)
    return R, tr


def randn_broadcast(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate standard normal noise on rank 0, broadcast to all ranks.

    CUDA RNG is not bit-identical across devices even with the same seed, so
    EDM sampling and CP parity tests both build noise on rank 0 and broadcast.
    """
    noise = (
        torch.randn(shape, device=device, dtype=dtype)
        if is_rank0()
        else torch.empty(shape, device=device, dtype=dtype)
    )
    return broadcast_from_rank0(noise)


def broadcast_from_rank0(tensor: torch.Tensor) -> torch.Tensor:
    """Broadcast ``tensor`` from rank 0 when distributed sampling is active.

    Same fixture used by tests (rank-0 generates, peers receive) and by the
    EDM sampler (rank-0 generates the noise schedule, peers receive).
    """
    if is_distributed_sampling():
        dist.broadcast(tensor, src=0)
    return tensor


def random_augmentation(
    n: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_distributed_sampling():
        return _random_augmentation_broadcast(n, device, dtype)
    return compute_random_augmentation(n, device=device, dtype=dtype)


def randn(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if is_distributed_sampling():
        return randn_broadcast(shape, device, dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def barrier() -> None:
    if is_distributed_sampling():
        dist.barrier()
