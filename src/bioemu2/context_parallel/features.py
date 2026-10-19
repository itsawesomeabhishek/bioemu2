"""Serial feature conversion and caller-owned CP padding."""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.conditioning.contact import contact_conditioning_info
from bioemu2.context_parallel.utils.dtensor import (
    contiguous_stride,
    msa_placements,
    pair_dtensor_from_local,
)
from bioemu2.context_parallel.utils.mesh import rank_msa_ranges, rank_pair_ranges
from bioemu2.context_parallel.utils.padding import (
    CPPadSizes,
    pad_along_atom,
    pad_along_token,
)
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.all_atom_structure import ATOM_LEVEL_PROPERTIES, TOKEN_LEVEL_PROPERTIES
from bioemu2.structure.batch import REQUIRES_MSA_PADDING


def pad_feats_for_cp(batch: BatchAllAtomStructure, sizes: CPPadSizes) -> dict[str, torch.Tensor]:
    """Convert an aligned batch and pad whole feature groups to the supplied sizes."""
    feats = convert_densebatch_to_dictionary(batch, build_pair_features=False)
    feats["token_idx_at_atom"] = feats["token_idx_at_atom"].clamp_min(0)
    token_keys = [
        key
        for key in (
            *TOKEN_LEVEL_PROPERTIES,
            "res_type",
            "residue_index",
            "pocket_feature",
            "method_feature",
            "token_pad_mask",
        )
        if key in feats and key not in REQUIRES_MSA_PADDING
    ]
    atom_keys = [
        key for key in (*ATOM_LEVEL_PROPERTIES, "atom_pad_mask", "ref_space_uid") if key in feats
    ]
    n_atom_pad = sizes.n_atom_final - feats["template_coords"].shape[1]
    token_feats = pad_along_token([feats[key] for key in token_keys], sizes.n_token_pad)
    atom_feats = pad_along_atom([feats[key] for key in atom_keys], n_atom_pad)
    feats.update(zip(token_keys, token_feats, strict=True))
    feats.update(zip(atom_keys, atom_feats, strict=True))
    return feats


def build_shard_pair_feats(
    batch: BatchAllAtomStructure,
    sizes: CPPadSizes,
    mesh: DeviceMesh,
    *,
    device: torch.device,
) -> dict[str, DTensor]:
    """Build this rank's slab of the pair input features without any full ``N x N`` tensor.

    ``batch`` may stay on the CPU: only bond indices and this rank's contact slab are moved
    to ``device``, so dense contact tensors never reach the accelerator.
    """

    batch_size = int(batch.batch_size)  # B

    n_real = sizes.n_token_orig  # N
    n_final = sizes.n_token_final  # Nf
    (b_lo, b_hi), (i_lo, i_hi), (j_lo, j_hi) = rank_pair_ranges(mesh, batch_size, n_final)
    local_batch, n_rows, n_cols = b_hi - b_lo, i_hi - i_lo, j_hi - j_lo  # Bl, Ni, Nj
    real_rows = min(max(n_real - i_lo, 0), n_rows)  # Nr
    real_cols = min(max(n_real - j_lo, 0), n_cols)  # Nc

    token_idx_at_atom_dp = batch.token_idx_at_atom[b_lo:b_hi]  # (Bl, A)
    bond_index_dp = batch.bond_index[b_lo:b_hi]  # (Bl, 2, E)
    token_bond_index_dp = token_idx_at_atom_dp[
        torch.arange(local_batch, device=bond_index_dp.device)[:, None, None], bond_index_dp
    ].to(device)  # (Bl, 2, E) in [0, N)

    # Map bond endpoints to slab coordinates; endpoints outside the slab go to a dump row/column.
    row_index = token_bond_index_dp - i_lo  # (Bl, 2, E)
    row_index = row_index.masked_fill((row_index < 0) | (row_index >= n_rows), n_rows)
    col_index = token_bond_index_dp - j_lo  # (Bl, 2, E)
    col_index = col_index.masked_fill((col_index < 0) | (col_index >= n_cols), n_cols)

    batch_idx = torch.arange(local_batch, device=device)[:, None]  # (Bl, 1)
    token_bonds_local = torch.zeros(
        local_batch, n_rows + 1, n_cols + 1, 1, dtype=batch.coords.dtype, device=device
    )  # (Bl, Ni + 1, Nj + 1, 1)
    # Write both (i, j) and (j, i): some templates only list each bond once.
    token_bonds_local[batch_idx, row_index[:, 0, :], col_index[:, 1, :]] = 1
    token_bonds_local[batch_idx, row_index[:, 1, :], col_index[:, 0, :]] = 1
    token_bonds_local = token_bonds_local[
        :, :n_rows, :n_cols
    ].contiguous()  # (Bl, Ni, Nj, 1), values in {0, 1}

    contact_conditioning = torch.zeros(
        local_batch, n_rows, n_cols, len(contact_conditioning_info), dtype=torch.long, device=device
    )  # (Bl, Ni, Nj, C) int64
    contact_threshold = torch.zeros(
        local_batch, n_rows, n_cols, dtype=torch.float32, device=device
    )  # (Bl, Ni, Nj) float32
    if "contact_conditioning" in batch:
        real = (slice(b_lo, b_hi), slice(i_lo, i_lo + real_rows), slice(j_lo, j_lo + real_cols))
        contact_conditioning[:, :real_rows, :real_cols] = batch.contact_conditioning[real].to(
            device
        )  # (Bl, Nr, Nc, C) int64
        contact_threshold[:, :real_rows, :real_cols] = batch.contact_threshold[real].to(
            device
        )  # (Bl, Nr, Nc) float32
    else:
        # Same values as make_unspecified_contact_conditioning/_threshold (threshold stays zero).
        contact_conditioning[
            :, :real_rows, :real_cols, contact_conditioning_info["UNSPECIFIED"]
        ] = 1

    shape = (batch_size, n_final, n_final)  # (B, Nf, Nf)
    return {
        "token_bonds": pair_dtensor_from_local(
            token_bonds_local, mesh, torch.Size([*shape, 1])
        ),  # local (Bl, Ni, Nj, 1), global (B, Nf, Nf, 1)
        "contact_conditioning": pair_dtensor_from_local(
            contact_conditioning, mesh, torch.Size([*shape, len(contact_conditioning_info)])
        ),  # local (Bl, Ni, Nj, C), global (B, Nf, Nf, C)
        "contact_threshold": pair_dtensor_from_local(
            contact_threshold, mesh, torch.Size(shape)
        ),  # local (Bl, Ni, Nj), global (B, Nf, Nf)
    }


def build_shard_msa_feats(
    batch: BatchAllAtomStructure,
    sizes: CPPadSizes,
    mesh: DeviceMesh,
    *,
    device: torch.device,
) -> dict[str, DTensor]:
    """This rank's padded ``(B, S, N, ...)`` MSA slabs; see :func:`build_shard_pair_feats`."""
    shape = (int(batch.batch_size), sizes.n_msa_orig + sizes.n_msa_pad, sizes.n_token_final)
    (b_lo, b_hi), (s_lo, s_hi), (t_lo, t_hi) = rank_msa_ranges(mesh, *shape)
    feats = {}
    for key in REQUIRES_MSA_PADDING:
        if key not in batch:
            continue
        real = batch[key][b_lo:b_hi, t_lo:t_hi, s_lo:s_hi].transpose(1, 2)
        local = torch.zeros(
            b_hi - b_lo, s_hi - s_lo, t_hi - t_lo, *real.shape[3:], dtype=real.dtype, device=device
        )
        local[:, : real.shape[1], : real.shape[2]] = real.to(device)
        global_shape = torch.Size([*shape, *real.shape[3:]])
        feats[key] = DTensor.from_local(
            local,
            device_mesh=mesh,
            placements=msa_placements(global_shape, mesh),
            shape=global_shape,
            stride=contiguous_stride(global_shape),
        )
    return feats
