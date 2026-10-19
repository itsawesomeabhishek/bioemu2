"""Canonical CP mirror of bioemu2.biomoa.modules.conditioning."""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.modules.conditioning import MethodConditioning
from bioemu2.context_parallel.utils import (
    get_row_column_for_tile_z_ops,
    pair_dtensor_from_local,
)


class DistributedMethodConditioning(nn.Module):
    """Distributed BioMoa method conditioning that builds only this rank's pair slab."""

    def __init__(self, serial: MethodConditioning, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.embedding = serial.embedding
        self.embed_into_pairs = serial.embed_into_pairs
        self.pair_embedding = serial.pair_embedding

    def forward(
        self,
        method_feature: torch.Tensor,  # plain replicated; (B, N) long
        token_pad_mask: torch.Tensor,  # plain replicated; (B, N)
    ) -> tuple[
        torch.Tensor,  # plain replicated; (B, N, token_s)
        DTensor | None,  # Shard(0), Shard(1), Shard(2); (B, N, N, token_z)
    ]:
        single_method = self.embedding(method_feature)
        single_method.masked_fill_(~token_pad_mask.unsqueeze(-1), 0.0)
        if not self.embed_into_pairs:
            return single_method, None

        assert self.pair_embedding is not None
        per_token = self.pair_embedding(method_feature)  # (B, N, token_z)
        row, col = get_row_column_for_tile_z_ops(per_token, per_token, self.device_mesh)
        row_mask, col_mask = get_row_column_for_tile_z_ops(
            token_pad_mask, token_pad_mask, self.device_mesh
        )
        pair_local = row[:, :, None] + col[:, None, :]  # (B / DP, N_i, N_j, token_z)
        pair_local.masked_fill_(~(row_mask[:, :, None] & col_mask[:, None, :]).unsqueeze(-1), 0.0)
        batch, n_tokens, token_z = per_token.shape
        pair_method = pair_dtensor_from_local(
            pair_local,
            self.device_mesh,
            torch.Size([batch, n_tokens, n_tokens, token_z]),
        )
        return single_method, pair_method


__all__ = ["DistributedMethodConditioning"]
