# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""DTensor construction helpers with no model-package dependencies."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from bioemu2.context_parallel.utils.mesh import pair_placements

# Canonical placements shared by CP tests and helpers.
PL_SEQ = (Shard(0), Shard(1), Replicate())
PL_PAIR = (Shard(0), Shard(1), Shard(2))
PL_MSA = (Shard(0), Shard(1), Shard(2))


def contiguous_stride(shape: list[int] | torch.Size) -> tuple[int, ...]:
    """C-contiguous stride tuple for constructing global-shape DTensors via
    :func:`DTensor.from_local`."""
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return tuple(stride)


def pair_dtensor_from_local(
    local: torch.Tensor,
    mesh: DeviceMesh,
    global_shape: torch.Size,
) -> DTensor:
    """Wrap a local [B, Ni, Nj, *] slab as a pair-placements DTensor.

    Uses contiguous stride for ``global_shape``, matching ``DistributedOuterProductMean.forward``'s
    convention so downstream CP modules (cuEq TriMul, etc.) accept the result without resharding.
    """
    return DTensor.from_local(
        local.contiguous(),
        device_mesh=mesh,
        placements=PL_PAIR,
        shape=global_shape,
        stride=contiguous_stride(global_shape),
    )


def apply_pointwise(module: nn.Module, dt: DTensor) -> DTensor:
    """Apply a feature-dim-only module (LayerNorm, Linear) to a DTensor, bypassing
    ``LinearParamsReplicated``/``LayerNormParamsReplicated`` wrapping.

    Safe *only* because these modules act on the last (feature) dim, which maps to Replicate
    in the CP placement — local-shard application is bit-equivalent to full-tensor application.
    """
    local = module(dt.to_local())
    out_feat = local.shape[-1]
    global_shape = torch.Size([*dt.shape[:-1], out_feat])
    return DTensor.from_local(
        local.contiguous(),
        device_mesh=dt.device_mesh,
        placements=dt.placements,
        shape=global_shape,
        stride=contiguous_stride(global_shape),
    )


def shard_along_token(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    *,
    src_data_rank: int | None = 0,
) -> list[DTensor]:
    # (B, N_token, ...) -> DTensor; token axis (dim 1) sharded on cp_axis_0.
    # src_data_rank=None skips the broadcast — pass it when callers already
    # have the same data on every rank (model forward path).
    placements = (Shard(0), Shard(1)) + tuple(Replicate() for _ in range(mesh.ndim - 2))
    return [
        distribute_tensor(v.contiguous(), mesh, placements, src_data_rank=src_data_rank)
        for v in tensors
    ]


def gather_along_token(tensors: list[DTensor]) -> list[torch.Tensor]:
    # DTensor(B, N_token_pad, ...) -> plain; all-gather across cp_axis_0.
    return [v.full_tensor() for v in tensors]


def shard_pair_along_token(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    *,
    src_data_rank: int | None = 0,
) -> list[DTensor]:
    # (B, N_token_pad, N_token_pad, ...) -> DTensor; sharded on cp_axis_0 and cp_axis_1.
    placements = pair_placements(mesh)
    return [
        distribute_tensor(v.contiguous(), mesh, placements, src_data_rank=src_data_rank)
        for v in tensors
    ]


def gather_pair_along_token(tensors: list[DTensor]) -> list[torch.Tensor]:
    # All-gather across cp_axis_0 and cp_axis_1.
    return [v.full_tensor() for v in tensors]


def shard_along_atom(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    *,
    src_data_rank: int | None = 0,
) -> list[DTensor]:
    # (B, N_atom_pad, ...) -> DTensor; atom axis (dim 1) sharded on cp_axis_0.
    placements = (Shard(0), Shard(1)) + tuple(Replicate() for _ in range(mesh.ndim - 2))
    return [
        distribute_tensor(v.contiguous(), mesh, placements, src_data_rank=src_data_rank)
        for v in tensors
    ]


def gather_along_atom(tensors: list[DTensor]) -> list[torch.Tensor]:
    # All-gather across cp_axis_0.
    return [v.full_tensor() for v in tensors]


def shard_along_window(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    *,
    src_data_rank: int | None = 0,
) -> list[DTensor]:
    # (B, K_pad, W, ...)
    placements = (Shard(0), Shard(1)) + tuple(Replicate() for _ in range(mesh.ndim - 2))
    return [
        distribute_tensor(v.contiguous(), mesh, placements, src_data_rank=src_data_rank)
        for v in tensors
    ]


def gather_along_window(tensors: list[DTensor]) -> list[torch.Tensor]:
    # All-gather across cp_axis_0.
    return [v.full_tensor() for v in tensors]


def msa_placements(shape: torch.Size, mesh: DeviceMesh) -> tuple:
    leading = tuple(Replicate() if shape[i] == 1 else PL_MSA[i] for i in range(3))
    return leading + (Replicate(),) * (mesh.ndim - 3)


def shard_along_msa(
    tensors: list[torch.Tensor],
    mesh: DeviceMesh,
    *,
    src_data_rank: int | None = 0,
) -> list[DTensor]:
    # (B, S_msa_pad, N, ...) -> DTensor; Shard(0)=B, Shard(1)=S, Shard(2)=N,
    # Replicate() on trailing feature dims. Size-1 leading axes fall back to
    # Replicate() (Shard(k) on size-1 can't split, e.g. (B, 1, N, c_emb)).
    return [
        distribute_tensor(
            v.contiguous(), mesh, msa_placements(v.shape, mesh), src_data_rank=src_data_rank
        )
        for v in tensors
    ]
