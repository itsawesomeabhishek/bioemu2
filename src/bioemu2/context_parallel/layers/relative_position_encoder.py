"""Canonical CP wrapper for BioMoa's fused relative position encoder."""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.modules.relative_position_encoder import FusedRelativePositionEncoder
from bioemu2.context_parallel.utils import (
    get_row_column_for_tile_z_ops,
    pair_dtensor_from_local,
)


class DistributedRelativePositionEncoder(nn.Module):
    def __init__(
        self,
        serial: FusedRelativePositionEncoder,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.r_max = serial.r_max
        self.s_max = serial.s_max
        self.embed_rel_pos = serial.embed_rel_pos
        self.embed_rel_token = serial.embed_rel_token
        self.embed_same_entity = serial.embed_same_entity
        self.embed_rel_chain = serial.embed_rel_chain

    def forward(
        self,
        asym_id: torch.Tensor,  # plain; (B, N)
        entity_id: torch.Tensor,  # plain; (B, N)
        residue_index: torch.Tensor,  # plain; (B, N)
        sym_id: torch.Tensor,  # plain; (B, N)
    ) -> DTensor:  # Shard(0), Shard(1), Shard(2); (B, N, N, c_z)
        row_asym, col_asym = get_row_column_for_tile_z_ops(asym_id, asym_id, self.device_mesh)
        same_chain = row_asym[:, :, None] == col_asym[:, None, :]

        row_res, col_res = get_row_column_for_tile_z_ops(
            residue_index, residue_index, self.device_mesh
        )
        same_residue = row_res[:, :, None] == col_res[:, None, :]
        d_residue = (row_res[:, :, None] - col_res[:, None, :] + self.r_max).clamp(
            0, 2 * self.r_max
        )
        d_residue = torch.where(
            same_chain, d_residue, d_residue.new_full((), 2 * self.r_max + 1)
        ).long()

        batch, n_tokens = asym_id.shape
        token_index = torch.arange(n_tokens, device=asym_id.device).expand(batch, -1)
        row_token, col_token = get_row_column_for_tile_z_ops(
            token_index, token_index, self.device_mesh
        )
        d_token = (row_token[:, :, None] - col_token[:, None, :] + self.r_max).clamp(
            0, 2 * self.r_max
        )
        d_token = torch.where(
            same_chain & same_residue,
            d_token,
            d_token.new_full((), 2 * self.r_max + 1),
        ).long()

        row_entity, col_entity = get_row_column_for_tile_z_ops(
            entity_id, entity_id, self.device_mesh
        )
        same_entity = row_entity[:, :, None] == col_entity[:, None, :]
        row_sym, col_sym = get_row_column_for_tile_z_ops(sym_id, sym_id, self.device_mesh)
        d_chain = (row_sym[:, :, None] - col_sym[:, None, :] + self.s_max).clamp(0, 2 * self.s_max)
        d_chain = torch.where(same_entity, d_chain, d_chain.new_full((), 2 * self.s_max + 1)).long()

        local = (
            self.embed_rel_pos(d_residue)
            + self.embed_rel_token(d_token)
            + self.embed_same_entity(
                same_entity.unsqueeze(-1).to(self.embed_same_entity.weight.dtype)
            )
            + self.embed_rel_chain(d_chain)
        )
        return pair_dtensor_from_local(
            local,
            self.device_mesh,
            torch.Size([batch, n_tokens, n_tokens, local.shape[-1]]),
        )


__all__ = ["DistributedRelativePositionEncoder"]
