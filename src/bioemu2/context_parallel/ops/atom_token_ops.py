"""CP counterparts of ``bioemu2.biomoa.ops.atom_token_ops``.

Public function names mirror ``bioemu2.biomoa.ops.atom_token_ops`` so the two
can be read side by side. Signatures are DTensor-flavoured (they take an
extra device mesh and mesh-derived shape kwargs), and the underlying
distributed primitives live alongside in ``scatter.py`` / ``pairmixer.py``
-- this module is the biomoa-parity surface.
"""

import torch
from torch.distributed.tensor import DTensor

from bioemu2.context_parallel.ops.distributed_primitives.flatten_and_unflatten import (
    shardwise_flatten_sharded,
    shardwise_unflatten_sharded,
)
from bioemu2.context_parallel.ops.distributed_primitives.gather import distributed_gather
from bioemu2.context_parallel.ops.distributed_primitives.outer_gather import (
    distributed_outer_gather,
)
from bioemu2.context_parallel.ops.distributed_primitives.utils import (
    convert_single_repr_to_window_batched_key,
)
from bioemu2.context_parallel.ops.scatter import distributed_scatter_reduce


def broadcast_token_pair_to_atom_window(
    z_dt: DTensor,  # Shard(0), Shard(1), Shard(2); (B, N_token, N_token, D)
    atom_to_token_ids_dt: DTensor,  # Shard(0), Shard(1), Replicate(); long (B, N_atom)
    atom_valid_mask_dt: DTensor,  # Shard(0), Shard(1), Replicate(); bool (B, N_atom), True = valid
    *,
    W: int,
    H: int,
    K: int,
) -> DTensor:  # Shard(0), Shard(1), Replicate(); (B, K, W, H, D)
    """CP z→p windowed atom-pair gather; mirrors ``bioemu2.biomoa.ops.atom_token_ops.broadcast_token_pair_to_atom_window``."""
    atom_to_token_q_dt = shardwise_unflatten_sharded(atom_to_token_ids_dt, axis=1, sizes=(K, W))
    atom_to_token_k_dt = convert_single_repr_to_window_batched_key(atom_to_token_ids_dt, W=W, H=H)
    valid_q_dt = shardwise_unflatten_sharded(atom_valid_mask_dt, axis=1, sizes=(K, W))
    valid_k_dt = convert_single_repr_to_window_batched_key(atom_valid_mask_dt, W=W, H=H)

    return distributed_outer_gather(
        z_dt,
        atom_to_token_q_dt,
        atom_to_token_k_dt,
        axis=1,
        are_ids_contiguous=True,
        idx_n_mask=valid_q_dt,
        idx_m_mask=valid_k_dt,
    )


def gather_center_tokens_from_atomic(
    q_windowed: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, K, W, D)
    token_center_atom_idx: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, N_token, 1)
    token_mask: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, N_token) bool, True = real
) -> DTensor:  # Shard(0), Shard(1), Replicate(); (B*M, N_token, D)
    """CP token-center gather; mirrors ``bioemu2.biomoa.ops.atom_token_ops.gather_center_tokens_from_atomic``.

    Uses ``shardwise_flatten_sharded`` + ``distributed_gather`` instead of the biomoa dense
    one-hot bmm; ``token_center_atom_idx`` retains a trailing singleton dim so it can be
    passed straight to ``distributed_gather`` (biomoa flattens the ``(K, W)`` axes implicitly).
    """
    q_flat = shardwise_flatten_sharded(q_windowed, start_dim=1, end_dim=2)
    gathered = distributed_gather(q_flat, token_center_atom_idx, axis=1, are_ids_contiguous=True)
    gathered = gathered.squeeze(2)  # drop the singleton gather-index dim
    gathered = gathered * token_mask.unsqueeze(-1).to(gathered.dtype)
    return gathered


def scatter_mean_atoms_to_tokens(
    per_atom: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, N, D)
    atom_to_token_ids: DTensor,  # Shard(0), Shard(1), Replicate(); long (B*M, N)
    is_padded_atom: DTensor,  # Shard(0), Shard(1), Replicate(); bool (B*M, N), True = pad
    num_tokens_per_rank: int,  # T // cp_size; the per-rank output slot count
) -> DTensor:  # Shard(0), Shard(1), Replicate(); (B*M, num_tokens_per_rank, D)
    """CP atom→token mean; mirrors ``bioemu2.biomoa.ops.atom_token_ops.scatter_mean_atoms_to_tokens``.

    Thin wrapper over :func:`distributed_scatter_reduce` with ``reduce='mean'`` and the biomoa
    biased denominator ``count + 1e-6`` (via ``eps_for_mean=1e-6``), so CP=1 matches the serial
    biomoa reference bit-for-bit. The pad mask polarity is flipped (biomoa's ``is_padded_atom``
    is True on pads; the underlying primitive wants True on valid) before dispatch.
    """
    idx_mask = ~is_padded_atom  # underlying primitive expects True = valid
    return distributed_scatter_reduce(
        output_size_per_rank=num_tokens_per_rank,
        axis=1,
        idx=atom_to_token_ids,
        src=per_atom,
        reduce="mean",
        idx_mask=idx_mask,
        are_ids_contiguous=True,
        eps_for_mean=1e-6,
    )


def broadcast_tokens_to_atoms(*args, **kwargs) -> torch.Tensor:
    """Raise because distributed token-to-atom broadcast is unsupported."""
    raise NotImplementedError(
        "distributed broadcast_tokens_to_atoms is not implemented; no CP caller exercises this op yet"
    )
