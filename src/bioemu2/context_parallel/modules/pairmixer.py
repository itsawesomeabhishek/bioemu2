# Evaluation-only CP mirror of bioemu2.biomoa.modules.pairmixer.
# Pair shards are updated in place; callers must not retain their previous values.

"""Context-parallel PairMixer trunk, MSA, and sequence-embedder modules."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from bioemu2.biomoa.modules.pairmixer import (
    PairMixerLayer,
    PairMixerModule,
    PairMixerMSALayer,
    PairMixerMSAModule,
    PairMixerSequenceEmbedder,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.layers.relative_position_encoder import (
    DistributedRelativePositionEncoder,
)
from bioemu2.context_parallel.modules.conditioning import DistributedMethodConditioning
from bioemu2.context_parallel.ops.cat_and_chunk import shardwise_cat
from bioemu2.context_parallel.ops.distributed_primitives.elementwise_op import (
    ElementwiseOp,
    elementwise_op,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.outer_product_mean import DistributedOuterProductMean
from bioemu2.context_parallel.ops.pair_averaging import Ring2DCommPairAveraging
from bioemu2.context_parallel.ops.pair_weighted_averaging import (
    DistributedPairWeightedAveraging,
)
from bioemu2.context_parallel.ops.transition import DTensorAwareTransition
from bioemu2.context_parallel.ops.triangle_multiplication import (
    BioMoaTriangleMultiplication,
    TriangleMultiplicationAddRes,
    _Direction,
)
from bioemu2.context_parallel.utils import (
    apply_pointwise,
    get_row_column_for_tile_z_ops,
    pair_dtensor_from_local,
)
from bioemu2.context_parallel.utils.dtensor import shard_along_msa
from bioemu2.context_parallel.utils.mesh import pair_placements


def _distributed_trimul(
    direction: _Direction,
    layer: BioMoaTriangleMultiplication,
    dist_manager: DistributedManager,
) -> TriangleMultiplicationAddRes:
    comm = Ring2DComm(
        dist_manager.group["cp"],
        dist_manager.subgroups["cp"][0],
        dist_manager.layout_subgroups["cp"],
    )
    return TriangleMultiplicationAddRes(
        direction=direction,
        layer=layer,
        device_mesh=dist_manager.device_mesh_subgroups,
        comm=comm,
    )


@dataclass
class _MSAPrepared:
    m: DTensor
    msa_mask: DTensor
    msa_row_count: torch.Tensor  # Valid MSA rows per local system; OPM denominator, (B / DP,).
    token_mask: DTensor


class DistributedPairMixerLayer(nn.Module):
    """Distributed BioMoa pair update block."""

    def __init__(self, serial: PairMixerLayer, dist_manager: DistributedManager) -> None:
        super().__init__()
        mesh = dist_manager.device_mesh_subgroups
        self.tri_mul_out = _distributed_trimul(
            _Direction.Outgoing, serial.tri_mul_out, dist_manager
        )
        self.tri_mul_in = _distributed_trimul(_Direction.Incoming, serial.tri_mul_in, dist_manager)
        self.transition_z = DTensorAwareTransition(serial.transition_z, mesh)

    def forward(
        self,
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        pair_mask: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T)
    ) -> DTensor:  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        self.tri_mul_out(z, pair_mask)
        self.tri_mul_in(z, pair_mask)
        return elementwise_op(z, self.transition_z(z), ElementwiseOp.SUM, in_place=True)


class DistributedPairMixerModule(nn.Module):
    """Distributed stack of BioMoa pair update blocks."""

    def __init__(self, serial: PairMixerModule, dist_manager: DistributedManager) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DistributedPairMixerLayer(layer, dist_manager) for layer in serial.layers
        )

    def forward(
        self,
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        pair_mask: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T)
    ) -> DTensor:  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        for layer in self.layers:
            z = layer(z, pair_mask)
        return z


class DistributedPairMixerMSALayer(nn.Module):
    """Distributed BioMoa MSA-to-pair update block."""

    def __init__(self, serial: PairMixerMSALayer, dist_manager: DistributedManager) -> None:
        super().__init__()
        mesh = dist_manager.device_mesh_subgroups
        self.msa_transition = serial.msa_transition
        self.pair_weighted_averaging = DistributedPairWeightedAveraging(
            serial.pair_weighted_averaging,
            mesh,
            Ring2DCommPairAveraging(
                dist_manager.group["cp"],
                dist_manager.subgroups["cp"][0],
                dist_manager.layout_subgroups["cp"],
            ),
        )
        self.outer_product_mean = DistributedOuterProductMean(
            serial.outer_product_mean,
            mesh,
            Ring2DComm(
                dist_manager.group["cp"],
                dist_manager.subgroups["cp"][0],
                dist_manager.layout_subgroups["cp"],
            ),
        )
        self.tri_mul_out = _distributed_trimul(
            _Direction.Outgoing, serial.tri_mul_out, dist_manager
        )
        self.tri_mul_in = _distributed_trimul(_Direction.Incoming, serial.tri_mul_in, dist_manager)
        self.z_transition = DTensorAwareTransition(serial.z_transition, mesh)

    def forward(
        self,
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        m: DTensor,  # Shard(0), Shard(1), Shard(2); (B, S, T, c_m)
        token_mask: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T)
        msa_mask: DTensor,  # Shard(0), Shard(1), Shard(2); (B, S, T)
        msa_row_count: torch.Tensor,  # Valid MSA rows per local system; (B / DP,).
    ) -> tuple[DTensor, DTensor]:
        m_local = m.to_local()
        m_local = m_local + self.pair_weighted_averaging(m, z, token_mask).to_local()
        m_local = m_local + self.msa_transition(m_local)
        m = DTensor.from_local(
            m_local,
            device_mesh=m.device_mesh,
            placements=m.placements,
            shape=m.shape,
            stride=m.stride(),
        )

        elementwise_op(
            z, self.outer_product_mean(m, msa_mask, msa_row_count), ElementwiseOp.SUM, in_place=True
        )
        self.tri_mul_out(z, token_mask)
        self.tri_mul_in(z, token_mask)
        elementwise_op(z, self.z_transition(z), ElementwiseOp.SUM, in_place=True)
        return z, m


class DistributedPairMixerMSAModule(nn.Module):
    """Distributed BioMoa MSA module."""

    def __init__(self, serial: PairMixerMSAModule, dist_manager: DistributedManager) -> None:
        super().__init__()
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.use_paired_feature = serial.use_paired_feature
        self.s_proj = LinearParamsReplicated(serial.s_proj, self.device_mesh)
        self.msa_proj = LinearParamsReplicated(serial.msa_proj, self.device_mesh)
        self.layers = nn.ModuleList(
            DistributedPairMixerMSALayer(layer, dist_manager) for layer in serial.layers
        )

    def prepare(
        self,
        *,
        emb: torch.Tensor,  # plain replicated; (B, T, c_emb)
        msa: DTensor,  # PL_MSA; (B, S, T, c_msa)
        has_deletion: DTensor,  # PL_MSA; (B, S, T)
        deletion_value: DTensor,  # PL_MSA; (B, S, T)
        msa_mask: DTensor,  # PL_MSA; (B, S, T)
        token_pad_mask: torch.Tensor,  # plain replicated; (B, T)
        msa_paired: DTensor | None,  # PL_MSA; (B, S, T)
        z_dtype: torch.dtype,
    ) -> _MSAPrepared:
        msa_features = [msa, has_deletion.unsqueeze(-1), deletion_value.unsqueeze(-1)]
        if self.use_paired_feature:
            assert msa_paired is not None
            msa_features.append(msa_paired.unsqueeze(-1))
        m = self.msa_proj(shardwise_cat(msa_features, dim=-1))

        (emb_dt,) = shard_along_msa(
            [emb.unsqueeze(1)],
            self.device_mesh,
            src_data_rank=None,
        )
        m_local = m.to_local() + self.s_proj(emb_dt).to_local()
        m_local.masked_fill_(~msa_mask.to_local().bool().unsqueeze(-1), 0.0)
        if m_local.dtype != z_dtype:
            m_local = m_local.to(z_dtype)
        m = DTensor.from_local(
            m_local,
            device_mesh=m.device_mesh,
            placements=pair_placements(self.device_mesh),
            shape=m.shape,
            stride=m.stride(),
        )

        row, col = get_row_column_for_tile_z_ops(
            token_pad_mask.bool(),
            token_pad_mask.bool(),
            self.device_mesh,
        )
        token_mask_local = row.float()[:, :, None] * col.float()[:, None, :]
        token_mask = pair_dtensor_from_local(
            token_mask_local,
            self.device_mesh,
            torch.Size([token_pad_mask.shape[0], token_pad_mask.shape[1], token_pad_mask.shape[1]]),
        )
        msa_row_count = distribute_tensor(
            msa_mask.float().amax(dim=-1).full_tensor().sum(dim=1).clamp(min=1),
            self.device_mesh,
            (Shard(0), Replicate(), Replicate()),
            src_data_rank=None,
        )
        return _MSAPrepared(
            m=m,
            msa_mask=msa_mask,
            msa_row_count=msa_row_count.to_local(),
            token_mask=token_mask,
        )

    def forward(
        self,
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        prepared: _MSAPrepared,
    ) -> DTensor:  # Shard(0), Shard(1), Shard(2); (B, T, T, c_z)
        m = prepared.m
        for layer in self.layers:
            z, m = layer(
                z,
                m,
                prepared.token_mask,
                prepared.msa_mask,
                prepared.msa_row_count,
            )
        return z


class PairMixerSequenceEmbedderDistributed(nn.Module):
    """Evaluation-only BioMoa sequence embedder with in-place pair updates."""

    def __init__(self, serial: PairMixerSequenceEmbedder, dist_manager: DistributedManager) -> None:
        super().__init__()
        self.device_mesh = dist_manager.device_mesh_subgroups
        self.s_init = serial.s_init
        self.z_init_1 = serial.z_init_1
        self.z_init_2 = serial.z_init_2
        self.token_bonds = serial.token_bonds
        self.s_recycle = serial.s_recycle
        self.z_recycle = serial.z_recycle
        self.s_norm = serial.s_norm
        self.z_norm = serial.z_norm
        self.rel_pos = DistributedRelativePositionEncoder(serial.rel_pos, self.device_mesh)
        self.msa_module = DistributedPairMixerMSAModule(serial.msa_module, dist_manager)
        self.pairmixer_module = DistributedPairMixerModule(serial.pairmixer_module, dist_manager)
        self.use_method_conditioning = serial.use_method_conditioning
        self.use_contact_conditioning = serial.use_contact_conditioning
        if self.use_method_conditioning:
            self.method_conditioning = DistributedMethodConditioning(
                serial.method_conditioning, self.device_mesh
            )
        if self.use_contact_conditioning:
            self.contact_conditioning = serial.contact_conditioning

    @torch.no_grad()
    def forward(
        self,
        s_inputs: torch.Tensor,  # plain replicated; (B, T, c_input)
        *,
        asym_id: torch.Tensor,  # plain replicated; (B, T)
        entity_id: torch.Tensor,  # plain replicated; (B, T)
        residue_index: torch.Tensor,  # plain replicated; (B, T)
        sym_id: torch.Tensor,  # plain replicated; (B, T)
        token_pad_mask: torch.Tensor,  # plain replicated; (B, T)
        token_bonds: DTensor,  # pair-sharded; global (B, T, T, 1)
        msa: DTensor,  # PL_MSA; (B, S, T, c_msa)
        has_deletion: DTensor,  # PL_MSA; (B, S, T)
        deletion_value: DTensor,  # PL_MSA; (B, S, T)
        msa_mask: DTensor,  # PL_MSA; (B, S, T)
        msa_paired: DTensor | None = None,  # PL_MSA; (B, S, T)
        method_feature: torch.Tensor | None = None,  # plain replicated; (B, T)
        contact_conditioning: DTensor | None = None,  # pair-sharded; global (B, T, T, C)
        contact_threshold: DTensor | None = None,  # pair-sharded; global (B, T, T)
        num_steps_recycle: int = 0,
    ) -> tuple[torch.Tensor, DTensor]:
        s_init = self.s_init(s_inputs)
        pair_method = None
        if self.use_method_conditioning:
            assert method_feature is not None
            single_method, pair_method = self.method_conditioning(
                method_feature=method_feature,
                token_pad_mask=token_pad_mask,
            )
            s_init = s_init + single_method

        z1 = self.z_init_1(s_inputs)  # Plain replicated (B, N, c_z).
        z2 = self.z_init_2(s_inputs)  # Plain replicated (B, N, c_z).
        batch_size, n_tokens, c_z = z1.shape
        # Row: (B/DP, N/CP0, c_z); column: (B/DP, N/CP1, c_z).
        row, col = get_row_column_for_tile_z_ops(z1, z2, self.device_mesh)
        z = pair_dtensor_from_local(
            row[:, :, None, :] + col[:, None, :, :],  # (B/DP, N/CP0, N/CP1, c_z).
            self.device_mesh,
            torch.Size([batch_size, n_tokens, n_tokens, c_z]),
        )
        z = z + self.rel_pos(
            asym_id=asym_id,
            entity_id=entity_id,
            residue_index=residue_index,
            sym_id=sym_id,
        )

        if pair_method is not None:
            z = z + pair_method
            del pair_method
        token_bonds_local = self.token_bonds(token_bonds.to_local())
        z = z + pair_dtensor_from_local(
            token_bonds_local,
            self.device_mesh,
            torch.Size([batch_size, n_tokens, n_tokens, c_z]),
        )
        del token_bonds_local
        if self.use_contact_conditioning:
            assert contact_conditioning is not None and contact_threshold is not None
            row_mask, col_mask = get_row_column_for_tile_z_ops(
                token_pad_mask.bool(),
                token_pad_mask.bool(),
                self.device_mesh,
            )
            contact_local = self._contact_conditioning(
                contact_conditioning.to_local(),
                contact_threshold.to_local(),
                row_mask,
                col_mask,
            )
            z = z + pair_dtensor_from_local(
                contact_local,
                self.device_mesh,
                torch.Size([batch_size, n_tokens, n_tokens, c_z]),
            )
            del contact_local

        row_mask, col_mask = get_row_column_for_tile_z_ops(
            token_pad_mask.bool(),
            token_pad_mask.bool(),
            self.device_mesh,
        )
        pair_mask_local = row_mask[:, :, None] & col_mask[:, None, :]
        pair_mask = pair_dtensor_from_local(
            pair_mask_local,
            self.device_mesh,
            torch.Size([batch_size, n_tokens, n_tokens]),
        )
        msa_prepared = self.msa_module.prepare(
            emb=s_inputs,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_mask=msa_mask,
            token_pad_mask=token_pad_mask,
            msa_paired=msa_paired,
            z_dtype=z.dtype,
        )
        z_init = z
        s = torch.zeros_like(s_init)
        z = z * 0
        for _ in range(num_steps_recycle + 1):
            s = s_init + self.s_recycle(self.s_norm(s))
            z = z_init + apply_pointwise(self.z_recycle, apply_pointwise(self.z_norm, z))
            z = self.msa_module(z, msa_prepared)
            z = self.pairmixer_module(z, pair_mask)
        return s, z

    def _contact_conditioning(
        self,
        contact_conditioning: torch.Tensor,
        contact_threshold: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
    ) -> torch.Tensor:
        features = contact_conditioning[..., 2:]
        normalized_threshold = (contact_threshold - self.contact_conditioning.cutoff_min) / (
            self.contact_conditioning.cutoff_max - self.contact_conditioning.cutoff_min
        )
        threshold_fourier = self.contact_conditioning.fourier_embedding(
            normalized_threshold.flatten()
        ).reshape(normalized_threshold.shape + (-1,))
        features = self.contact_conditioning.encoder(
            torch.cat([features, normalized_threshold.unsqueeze(-1), threshold_fourier], dim=-1)
        )
        features = (
            features * (1 - contact_conditioning[..., :2].sum(dim=-1, keepdim=True))
            + self.contact_conditioning.encoding_unspecified * contact_conditioning[..., :1]
            + self.contact_conditioning.encoding_unselected * contact_conditioning[..., 1:2]
        )
        pair_mask = row_mask[:, :, None] & col_mask[:, None, :]
        return features.masked_fill(~pair_mask.unsqueeze(-1), 0.0)
