"""Canonical distributed BioMoaC dual-stream transformer layers."""

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.modules.diffusion import DualStreamCoupling, DualStreamDiffusionLayer
from bioemu2.biomoa.modules.transformer import (
    ConditionedSlidingWindowTransformerLayer,
    ConditionedTransformerLayer,
    ada_ln,
    ada_scale,
)
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.atom_token_ops import (
    gather_center_tokens_from_atomic,
)
from bioemu2.context_parallel.ops.attention import (
    AttentionPairBias,
    AttentionPairBiasSlidingWindow,
)
from bioemu2.context_parallel.ops.cat_and_chunk import shardwise_cat
from bioemu2.context_parallel.ops.distributed_primitives.gather import distributed_gather
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import LayerNormParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.transition import DTensorAwareTransition


class DistributedConditionedTransformerLayer(nn.Module):
    """DTensor counterpart of BioMoa's full-token conditioned block."""

    def __init__(
        self,
        serial: ConditionedTransformerLayer,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        device_mesh = dist_manager.device_mesh_subgroups
        self.lns_scale = serial.lns_scale
        self.cond = nn.Sequential(
            LayerNormParamsReplicated(serial.cond[0], device_mesh),
            nn.SiLU(),
            LinearParamsReplicated(serial.cond[2], device_mesh),
        )
        self.attention = AttentionPairBias(serial.attention.pair_bias_attn, dist_manager)
        self.transition = DTensorAwareTransition(serial.transition, device_mesh)

    def forward(
        self,
        x: DTensor,
        pairs: DTensor,
        c: DTensor,
        mask: DTensor,
        multiplicity: int,
    ) -> DTensor:
        attn_alpha, attn_beta, attn_gamma, mlp_alpha, mlp_beta, mlp_gamma = self.cond(c).chunk(
            6, dim=-1
        )
        h = self.attention(
            s=self.lns_scale * ada_ln(x, attn_alpha, attn_beta),
            z=pairs,
            mask=mask,
            multiplicity=multiplicity,
        )
        x = x + ada_scale(h, attn_gamma)
        h = self.transition(self.lns_scale * ada_ln(x, mlp_alpha, mlp_beta))
        return x + ada_scale(h, mlp_gamma)


class DistributedConditionedSlidingWindowTransformerLayer(nn.Module):
    """DTensor counterpart of BioMoa's windowed atom conditioned block."""

    def __init__(
        self,
        serial: ConditionedSlidingWindowTransformerLayer,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        device_mesh = dist_manager.device_mesh_subgroups
        layer = serial.transformer_layer
        self.lns_scale = layer.lns_scale
        self.cond = nn.Sequential(
            LayerNormParamsReplicated(layer.cond[0], device_mesh),
            nn.SiLU(),
            LinearParamsReplicated(layer.cond[2], device_mesh),
        )
        self.attention = AttentionPairBiasSlidingWindow(
            layer.attention.attention_layer.pair_bias_attn,
            device_mesh,
        )
        self.transition = DTensorAwareTransition(layer.transition, device_mesh)

    def forward(
        self,
        x: DTensor,
        pairs: DTensor,
        c: DTensor,
        mask: DTensor,
        multiplicity: int,
    ) -> DTensor:
        del multiplicity
        attn_alpha, attn_beta, attn_gamma, mlp_alpha, mlp_beta, mlp_gamma = self.cond(c).chunk(
            6, dim=-1
        )
        h = self.attention(
            s=self.lns_scale * ada_ln(x, attn_alpha, attn_beta),
            z=pairs,
            mask=mask,
        )
        x = x + ada_scale(h, attn_gamma)
        h = self.transition(self.lns_scale * ada_ln(x, mlp_alpha, mlp_beta))
        return x + ada_scale(h, mlp_gamma)


class DistributedDualStreamCoupling(nn.Module):
    """DTensor counterpart of BioMoa's bidirectional stream coupling."""

    def __init__(self, serial: DualStreamCoupling, dist_manager: DistributedManager) -> None:
        super().__init__()
        device_mesh = dist_manager.device_mesh_subgroups
        self.lns_scale = serial.lns_scale
        self.q_coupling_cond = nn.Sequential(
            LayerNormParamsReplicated(serial.q_coupling_cond[0], device_mesh),
            nn.SiLU(),
            LinearParamsReplicated(serial.q_coupling_cond[2], device_mesh),
        )
        self.s_coupling_cond = nn.Sequential(
            LayerNormParamsReplicated(serial.s_coupling_cond[0], device_mesh),
            nn.SiLU(),
            LinearParamsReplicated(serial.s_coupling_cond[2], device_mesh),
        )
        self.s_to_q_mixer = DTensorAwareTransition(serial.s_to_q_mixer, device_mesh)
        self.q_to_s_mixer = DTensorAwareTransition(serial.q_to_s_mixer, device_mesh)

    def forward(
        self,
        s: DTensor,
        q: DTensor,
        c: DTensor,
        c_to_q: DTensor,
        atom_to_token_ids: DTensor,
        is_padded_atom: DTensor,
        token_center_atom_index: DTensor,
        token_mask: DTensor,
    ) -> tuple[DTensor, DTensor]:
        atom_s = q.shape[-1]
        token_s = s.shape[-1]
        alpha3_q, beta3_q, gamma3_q = torch.split(
            self.q_coupling_cond(c_to_q),
            [token_s + atom_s, token_s + atom_s, atom_s],
            dim=-1,
        )
        alpha3_s, beta3_s, gamma3_s = torch.split(
            self.s_coupling_cond(c),
            [token_s + atom_s, token_s + atom_s, token_s],
            dim=-1,
        )

        s_to_q = distributed_gather(s, atom_to_token_ids, axis=1, are_ids_contiguous=True)
        s_to_q = s_to_q * (~is_padded_atom).unsqueeze(-1).to(s_to_q.dtype)
        s_to_q = shardwise_cat([s_to_q, q], dim=-1)
        s_to_q = ada_scale(
            self.s_to_q_mixer(self.lns_scale * ada_ln(s_to_q, alpha3_q, beta3_q)),
            gamma3_q,
        )

        q_to_s = gather_center_tokens_from_atomic(
            q,
            token_center_atom_index,
            token_mask,
        )
        q_to_s = shardwise_cat([q_to_s, s], dim=-1)
        q_to_s = ada_scale(
            self.q_to_s_mixer(self.lns_scale * ada_ln(q_to_s, alpha3_s, beta3_s)),
            gamma3_s,
        )
        return s + q_to_s, q + s_to_q


class DistributedDualStreamDiffusionLayer(nn.Module):
    """One BioMoaC atom block, token block, then bidirectional coupling."""

    def __init__(
        self,
        serial: DualStreamDiffusionLayer,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        self.atom_block = DistributedConditionedSlidingWindowTransformerLayer(
            serial.atom_block, dist_manager
        )
        self.token_block = DistributedConditionedTransformerLayer(serial.token_block, dist_manager)
        self.coupling = DistributedDualStreamCoupling(serial.coupling, dist_manager)

    def forward(
        self,
        s: DTensor,
        z: DTensor,
        q: DTensor,
        p: DTensor,
        c: DTensor,
        atom_mask: DTensor,
        token_attention_mask: DTensor,
        token_mask: DTensor,
        atom_to_token_ids: DTensor,
        is_padded_atom: DTensor,
        token_center_atom_index: DTensor,
        multiplicity: int,
    ) -> tuple[DTensor, DTensor]:
        c_to_q = distributed_gather(c, atom_to_token_ids, axis=1, are_ids_contiguous=True)
        c_to_q = c_to_q * (~is_padded_atom).unsqueeze(-1).to(c_to_q.dtype)

        q = self.atom_block(q, p, c_to_q, atom_mask, multiplicity)
        s = self.token_block(s, z, c, token_attention_mask, multiplicity)
        return self.coupling(
            s,
            q,
            c,
            c_to_q,
            atom_to_token_ids,
            is_padded_atom,
            token_center_atom_index,
            token_mask,
        )
