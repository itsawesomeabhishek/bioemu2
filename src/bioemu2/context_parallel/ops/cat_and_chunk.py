"""Shardwise concat along replicated DTensor dimensions.

No serial counterpart — distributed-only DTensor helper.
Subset of Fold-CP cat_and_chunk.py (only the non-sharded-dim concat path).
"""

import torch
from torch.distributed.tensor import DTensor

from bioemu2.context_parallel.utils import contiguous_stride


def shardwise_cat(tensors: list[DTensor], dim: int) -> DTensor:
    """Concatenate DTensors along a non-sharded tensor dimension."""
    if not tensors:
        raise ValueError("shardwise_cat requires at least one DTensor")
    first = tensors[0]
    dim = dim if dim >= 0 else dim + first.ndim
    out_local = torch.cat([tensor.to_local() for tensor in tensors], dim=dim)
    out_shape = list(first.shape)
    out_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    return DTensor.from_local(
        out_local,
        device_mesh=first.device_mesh,
        placements=first.placements,
        shape=torch.Size(out_shape),
        stride=contiguous_stride(out_shape),
    )


__all__ = ["shardwise_cat"]
