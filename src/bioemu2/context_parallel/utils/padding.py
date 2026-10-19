"""
NOTE: This module must not import from bioemu2 model packages (e.g. bioemu2.biomoa)
to avoid corrupting test imports.
The CP test runner spawns subprocesses that import this before model packages are wired up.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.context_parallel.utils.mesh import pad_amount, shard_multiple


def pad_dim(
    tensor: torch.Tensor,
    dim: int,
    pad_len: int,
    value: float = 0,
) -> torch.Tensor:
    """Pad the end of one tensor dimension."""
    if pad_len == 0:
        return tensor
    return F.pad(tensor, (0, 0) * (tensor.ndim - dim - 1) + (0, pad_len), value=value)


def pad_amounts(n_token: int, n_atom: int, n_msa: int, W: int, mesh: DeviceMesh) -> dict[str, int]:
    """{n_token, n_atom, n_msa} -> {n_token_pad, n_atom_pad, n_window_pad, n_msa_pad}; mirrors make_cp_pad_sizes."""
    shard_n = shard_multiple(mesh)
    # mesh.size(1) is cp_axis_0; msa pad mirrors make_cp_pad_sizes.depth_n which reads mesh.size(1).
    cp0 = int(mesh.size(1)) if mesh.ndim > 1 else 1
    n_atom_pad = pad_amount(n_atom, W * shard_n)
    return {
        "n_token_pad": pad_amount(n_token, shard_n),
        "n_atom_pad": n_atom_pad,
        # K = (n_atom + n_atom_pad) // W; must be a multiple of cp_axis_0 for the sliding-window K shard.
        "n_window_pad": pad_amount((n_atom + n_atom_pad) // W, cp0) if n_atom else 0,
        "n_msa_pad": pad_amount(n_msa, cp0),
    }


def pad_along_token(tensors: list[torch.Tensor], n_pad: int) -> list[torch.Tensor]:
    """(B, N_token, ...) -> (B, N_token_pad, ...); token axis is dim 1."""
    return [pad_dim(tensor, dim=1, pad_len=n_pad) for tensor in tensors]


def unpad_along_token(tensors: list[torch.Tensor], n_real: int) -> list[torch.Tensor]:
    """(B, N_token_pad, ...) -> (B, N_token, ...); trims token axis (dim 1) back to real length."""
    return [v.narrow(1, 0, n_real) for v in tensors]


def pad_pair_along_token(tensors: list[torch.Tensor], n_pad: int) -> list[torch.Tensor]:
    """(B, N_token, N_token, ...) -> (B, N_token_pad, N_token_pad, ...); pair-rep, both token axes padded."""
    return [
        pad_dim(pad_dim(tensor, dim=1, pad_len=n_pad), dim=2, pad_len=n_pad) for tensor in tensors
    ]


def unpad_pair_along_token(tensors: list[torch.Tensor], n_real: int) -> list[torch.Tensor]:
    """(B, N_token_pad, N_token_pad, ...) -> (B, N_token, N_token, ...); trims both token axes back to real length."""
    return [v.narrow(1, 0, n_real).narrow(2, 0, n_real) for v in tensors]


def pad_along_atom(tensors: list[torch.Tensor], n_pad: int) -> list[torch.Tensor]:
    """(B, N_atom, ...) -> (B, N_atom_pad, ...); padded to W * shard_multiple."""
    return [pad_dim(tensor, dim=1, pad_len=n_pad) for tensor in tensors]


def unpad_along_atom(tensors: list[torch.Tensor], n_real: int) -> list[torch.Tensor]:
    """(B, N_atom_pad, ...) -> (B, N_atom, ...); trims atom axis (dim 1) back to real length."""
    return [v.narrow(1, 0, n_real) for v in tensors]


def pad_along_window(tensors: list[torch.Tensor], n_pad: int) -> list[torch.Tensor]:
    """(B, K, W, ...) -> (B, K_pad, W, ...); K must be a multiple of cp_axis_0 for sliding-window attention."""
    return [pad_dim(tensor, dim=1, pad_len=n_pad) for tensor in tensors]


def unpad_along_window(tensors: list[torch.Tensor], n_real: int) -> list[torch.Tensor]:
    """(B, K_pad, W, ...) -> (B, K, W, ...); trims window axis (dim 1) back to real K."""
    return [v.narrow(1, 0, n_real) for v in tensors]


def unpad_along_msa(tensors: list[torch.Tensor], n_real: int) -> list[torch.Tensor]:
    """(B, S_msa_pad, ...) -> (B, S_msa, ...); trims msa depth axis (dim 1) back to real length."""
    return [v.narrow(1, 0, n_real) for v in tensors]


class CPPadSizes(NamedTuple):
    """Per-axis CP padding metadata.

    This metadata belongs to the caller, not the model's static context.
    """

    n_token_orig: int
    n_atom_pre_pad: int
    n_msa_orig: int
    n_token_pad: int  # amount of padding added (NOT padded length)
    n_atom_pad: int
    n_msa_pad: int
    n_atom_final: int  # = n_atom_pre_pad + n_atom_pad; convenient padded length
    n_token_final: int  # = n_token_orig + n_token_pad; convenient padded length


def make_cp_pad_sizes(
    *,
    n_token: int,
    n_atom: int,
    n_msa: int,
    atoms_per_window: int,
    mesh: DeviceMesh,
) -> CPPadSizes:
    """Compute all final CP axis lengths before tensors are padded."""
    shard_n = shard_multiple(mesh)
    depth_n = int(mesh.size(1))
    n_token_pad = pad_amount(n_token, shard_n)
    n_atom_pad = pad_amount(n_atom, atoms_per_window * shard_n)
    n_msa_pad = pad_amount(n_msa, depth_n) if n_msa else 0
    return CPPadSizes(
        n_token_orig=n_token,
        n_atom_pre_pad=n_atom,
        n_msa_orig=n_msa,
        n_token_pad=n_token_pad,
        n_atom_pad=n_atom_pad,
        n_msa_pad=n_msa_pad,
        n_atom_final=n_atom + n_atom_pad,
        n_token_final=n_token + n_token_pad,
    )


__all__ = [
    "CPPadSizes",
    "make_cp_pad_sizes",
    "pad_along_atom",
    "pad_along_token",
    "pad_along_window",
    "pad_amounts",
    "pad_dim",
    "pad_pair_along_token",
    "unpad_along_atom",
    "unpad_along_msa",
    "unpad_along_token",
    "unpad_along_window",
    "unpad_pair_along_token",
]
