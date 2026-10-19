# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CP mesh geometry: dimension calculations, placements, shard ranges."""

import math

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


def pad_amount(n: int, m: int) -> int:
    """Number of zeros to append so ``n`` becomes a multiple of ``m``."""
    return (m - (n % m)) % m


def shard_multiple(device_mesh: DeviceMesh) -> int:
    """The number each sharded axis must be divisible by (LCM of the two CP axis sizes).

    Returns 1 on a degenerate ``(1, 1, 1)`` mesh so the serial path incurs
    zero padding.
    """
    cp0 = device_mesh.size(1) if device_mesh.ndim > 1 else 1
    cp1 = device_mesh.size(2) if device_mesh.ndim > 2 else 1
    return (cp0 * cp1) // math.gcd(cp0, cp1)


def pair_placements(device_mesh: DeviceMesh) -> tuple:
    """``(Shard(0), Shard(1), Shard(2), Replicate()...)`` — placements for ``[B, N, N, ...]`` tensors.

    The two N axes shard on cp_axis_0 and cp_axis_1 respectively; the trailing
    feature dim replicates. Symmetric peer of :func:`pair_placements` and
    the layout produced by :func:`pair_dtensor_from_local`.
    """
    return (
        Shard(0),
        Shard(1),
        Shard(2),
    ) + tuple(Replicate() for _ in range(device_mesh.ndim - 3))


# Index-range helpers for tiled pair construction


def _shard_range(n: int, num_shards: int, shard_idx: int) -> tuple[int, int]:
    """Contiguous ``[lo, hi)`` slice of ``[0, n)`` that DTensor ``Shard`` assigns to one shard.

    ``Shard`` splits like ``torch.chunk``: every shard holds ``ceil(n / num_shards)`` rows
    except at the tail, so the short shard and any empty shards are always the last ones.
    For ``n=5, num_shards=4`` the sizes are ``(2, 2, 1, 0)``. Local slabs built from these
    ranges line up with tensors placed by ``distribute_tensor``.
    """
    chunk = -(-n // num_shards)
    lo = min(shard_idx * chunk, n)
    return lo, min(lo + chunk, n)


def rank_pair_ranges(
    mesh: DeviceMesh, batch_size: int, n: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """``((b_lo, b_hi), (i_lo, i_hi), (j_lo, j_hi))`` — this rank's slab of a ``(B, N, N, ...)``
    pair tensor under :func:`pair_placements`.
    """
    coords = mesh.get_coordinate()
    if coords is None:
        raise RuntimeError("DeviceMesh.get_coordinate() returned None")
    return (
        _shard_range(batch_size, mesh.size(0), int(coords[0])),
        _shard_range(n, mesh.size(1), int(coords[1])),
        _shard_range(n, mesh.size(2), int(coords[2])),
    )


def rank_msa_ranges(
    mesh: DeviceMesh, batch_size: int, n_msa: int, n_token: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Like :func:`rank_pair_ranges` for a ``(B, S, N, ...)`` MSA tensor under ``msa_placements``."""
    coords = mesh.get_coordinate()
    if coords is None:
        raise RuntimeError("DeviceMesh.get_coordinate() returned None")
    b, s, t = (
        (0, n) if n == 1 else _shard_range(n, mesh.size(axis), int(coords[axis]))
        for axis, n in enumerate((batch_size, n_msa, n_token))
    )
    return b, s, t


def get_row_column_for_tile_z_ops(
    full_rol: torch.Tensor,  # Plain replicated (B, N) or (B, N, C).
    full_col: torch.Tensor,  # Plain replicated (B, N) or (B, N, C).
    mesh: DeviceMesh,  # Mesh axes: (DP, CP0, CP1).
) -> tuple[torch.Tensor, torch.Tensor]:  # Plain (B/DP, N/CP0, ...) and (B/DP, N/CP1, ...).
    # Placements map mesh axes to tensor axes; both global shapes stay (B, N, ...).
    row_dt = distribute_tensor(
        full_rol,
        mesh,
        (Shard(0), Shard(1), Replicate()),
        src_data_rank=None,
    )
    col_dt = distribute_tensor(
        full_col,
        mesh,
        (Shard(0), Replicate(), Shard(1)),
        src_data_rank=None,
    )
    return row_dt.to_local(), col_dt.to_local()
