"""Fused relative position encoder using embedding lookups.

Embeds relative position and chain indices directly with ``nn.Embedding``,
avoiding fp32 one-hot intermediates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class FusedRelativePositionEncoder(nn.Module):
    """Memory-efficient RelativePositionEncoder using embedding lookups.

    Weights are initialised so each embedding entry is drawn from
    ``N(0, 1/n_variants)`` where ``n_variants = 4`` (the number of summed
    embeddings).  This gives O(1) output variance per dimension.
    """

    def __init__(self, token_z: int, r_max: int = 32, s_max: int = 2) -> None:
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        n_rel = 2 * r_max + 2
        n_chain = 2 * s_max + 2

        self.embed_rel_pos = nn.Embedding(n_rel, token_z)
        self.embed_rel_token = nn.Embedding(n_rel, token_z)
        self.embed_same_entity = nn.Linear(1, token_z, bias=False)
        self.embed_rel_chain = nn.Embedding(n_chain, token_z)

        # Init: N(0, 1/4) per entry → sum of 4 lookups has O(1) variance.
        n_variants = 4
        std = (1.0 / n_variants) ** 0.5
        for param in [
            self.embed_rel_pos.weight,
            self.embed_rel_token.weight,
            self.embed_same_entity.weight,
            self.embed_rel_chain.weight,
        ]:
            nn.init.normal_(param, mean=0.0, std=std)

    def forward(
        self,
        asym_id: Tensor,  # (B, T)
        entity_id: Tensor,  # (B, T)
        residue_index: Tensor,  # (B, T)
        sym_id: Tensor,  # (B, T)
    ) -> Tensor:
        """Compute pairwise relative position encoding.

        Args:
            asym_id: Chain / asymmetric-unit id per token, ``(B, T)``.
            entity_id: Entity id per token, ``(B, T)``.
            residue_index: Residue index per token, ``(B, T)``.
            sym_id: Symmetry id per token, ``(B, T)``.

        Returns:
            Pairwise relative position encoding, ``(B, T, T, token_z)``.
        """
        B, T = asym_id.shape
        # token_index is a sequential 0..T-1 index — generated on the fly.
        token_index = torch.arange(T, device=asym_id.device).expand(B, -1)

        b_same_chain = asym_id[:, :, None] == asym_id[:, None, :]
        b_same_residue = residue_index[:, :, None] == residue_index[:, None, :]
        b_same_entity = entity_id[:, :, None] == entity_id[:, None, :]

        d_residue = (residue_index[:, :, None] - residue_index[:, None, :] + self.r_max).clamp(
            0, 2 * self.r_max
        )
        d_residue = torch.where(
            b_same_chain, d_residue, d_residue.new_full((), 2 * self.r_max + 1)
        ).long()

        d_token = (token_index[:, :, None] - token_index[:, None, :] + self.r_max).clamp(
            0, 2 * self.r_max
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            d_token.new_full((), 2 * self.r_max + 1),
        ).long()

        d_chain = (sym_id[:, :, None] - sym_id[:, None, :] + self.s_max).clamp(0, 2 * self.s_max)
        d_chain = torch.where(
            ~b_same_entity,
            d_chain.new_full((), 2 * self.s_max + 1),
            d_chain,
        ).long()

        return (
            self.embed_rel_pos(d_residue)
            + self.embed_rel_token(d_token)
            + self.embed_same_entity(
                b_same_entity.unsqueeze(-1).to(self.embed_same_entity.weight.dtype)
            )
            + self.embed_rel_chain(d_chain)
        )
