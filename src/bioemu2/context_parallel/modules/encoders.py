"""Canonical CP atom encoder for BioMoaC structure prediction."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.functional import softplus

from bioemu2.biomoa.modules.encoders import AtomAttentionEncoder
from bioemu2.biomoa.ops.atom_token_ops import (
    window_atom_single_to_pair_keys,
    windowed_atom_pair_geometry,
)
from bioemu2.context_parallel.ops.atom_token_ops import (
    broadcast_token_pair_to_atom_window,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.tiled_params_replicated import (
    LayerNormLinearParamsReplicatedTiled,
)
from bioemu2.context_parallel.utils.dtensor import shard_along_atom, shard_along_window


@dataclass
class AtomAttentionEncoderStaticContext:
    q: torch.Tensor
    c: torch.Tensor
    p: DTensor


class DistributedAtomAttentionEncoder(nn.Module):
    def __init__(
        self,
        serial: AtomAttentionEncoder,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        if serial.use_atom_encoder or serial.use_atom_to_token_aggregation:
            raise NotImplementedError(
                "BioMoaC structure CP requires atom encoder and aggregation disabled"
            )

        self.device_mesh = device_mesh
        self.atoms_per_window_queries = serial.atoms_per_window_queries
        self.atoms_per_window_keys = serial.atoms_per_window_keys
        self.proj_atom_continuous = serial.proj_atom_continuous
        self.embed_element = serial.embed_element
        self.embed_atom_name_chars = serial.embed_atom_name_chars
        self.embed_atompair_ref_pos = serial.embed_atompair_ref_pos
        self.embed_atompair_ref_dist = serial.embed_atompair_ref_dist
        self.embed_atompair_mask = serial.embed_atompair_mask
        self.s_to_c_trans = serial.s_to_c_trans
        self.r_to_q_trans = serial.r_to_q_trans
        self.use_dynamics_conditioning = serial.use_dynamics_conditioning
        self.downscale_condition_contrib = serial.downscale_condition_contrib
        if self.use_dynamics_conditioning:
            self.cond_to_q_trans = serial.cond_to_q_trans
            if self.downscale_condition_contrib:
                self.dt_gate_embed = serial.dt_gate_embed
                self.dt_gate_log_w = serial.dt_gate_log_w
        self.z_to_p_trans = LayerNormLinearParamsReplicatedTiled(serial.z_to_p_trans, device_mesh)
        self.c_to_p_trans_k = serial.c_to_p_trans_k
        self.c_to_p_trans_q = serial.c_to_p_trans_q

        p_mlp: list[nn.Module] = []
        for layer in serial.p_mlp:
            p_mlp.append(
                LinearParamsReplicated(layer, device_mesh)
                if isinstance(layer, nn.Linear)
                else layer
            )
        self.p_mlp = nn.Sequential(*p_mlp)

    def compute_static_context(
        self,
        *,
        atom_ref_pos: torch.Tensor,  # plain; (B, N, 3)
        ref_charge: torch.Tensor,  # plain; (B, N)
        ref_element: torch.Tensor,  # plain; (B, N)
        ref_atom_name_chars: torch.Tensor,  # plain; (B, N, 4)
        atom_ref_space_uid: torch.Tensor,  # plain; (B, N)
        atom_pad_mask: torch.Tensor,  # plain; (B, N)
        atom_to_token_ids: torch.Tensor,  # plain; (B, N)
        atom_to_token_ids_dt: DTensor,  # Shard(0), Shard(1), Replicate(); (B, N)
        atom_valid_mask_dt: DTensor,  # Shard(0), Shard(1), Replicate(); (B, N)
        s_trunk: torch.Tensor,  # plain; (B, T, token_s)
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, T, T, token_z)
    ) -> AtomAttentionEncoderStaticContext:
        batch, n_atoms = atom_ref_pos.shape[:2]
        is_padded_atom = ~atom_pad_mask.bool()

        continuous = torch.cat([atom_ref_pos, ref_charge.unsqueeze(-1)], dim=-1)
        c = self.proj_atom_continuous(continuous)
        c = c + self.embed_element(ref_element)
        for index, embedding in enumerate(self.embed_atom_name_chars):
            c = c + embedding(ref_atom_name_chars[:, :, index])
        c.masked_fill_(is_padded_atom.unsqueeze(-1), 0)
        q = c

        window_queries = self.atoms_per_window_queries
        window_keys = self.atoms_per_window_keys
        n_windows = n_atoms // window_queries
        atom_pair_disp, atom_pair_inv_dist, same_ref_space = windowed_atom_pair_geometry(
            atom_ref_pos=atom_ref_pos,
            atom_mask=atom_pad_mask,
            atom_uid=atom_ref_space_uid,
            atoms_per_window_queries=window_queries,
            atoms_per_window_keys=window_keys,
            out_dtype=c.dtype,
        )
        p = self.embed_atompair_ref_pos(atom_pair_disp)
        p = p + self.embed_atompair_ref_dist(atom_pair_inv_dist)
        p = p + self.embed_atompair_mask(same_ref_space)
        p.masked_fill_(~same_ref_space.bool(), 0)

        s_to_c = self.s_to_c_trans(s_trunk)
        s_to_c = s_to_c[torch.arange(batch, device=s_to_c.device).unsqueeze(1), atom_to_token_ids]
        s_to_c.masked_fill_(is_padded_atom.unsqueeze(-1), 0)
        c = c + s_to_c

        z_to_p = broadcast_token_pair_to_atom_window(
            self.z_to_p_trans(z),
            atom_to_token_ids_dt=atom_to_token_ids_dt,
            atom_valid_mask_dt=atom_valid_mask_dt,
            W=window_queries,
            H=window_keys,
            K=n_windows,
        )
        p = p + self.c_to_p_trans_q(c.view(batch, n_windows, window_queries, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(window_atom_single_to_pair_keys(c, window_queries, window_keys))
        (p_dt,) = shard_along_window([p], self.device_mesh, src_data_rank=None)
        p_dt = p_dt + z_to_p
        p_dt = p_dt + self.p_mlp(p_dt)
        return AtomAttentionEncoderStaticContext(q=q, c=c, p=p_dt)

    def apply_dynamic(
        self,
        static: AtomAttentionEncoderStaticContext,
        *,
        r: torch.Tensor,  # plain; (B*M, N, 3)
        multiplicity: int,
        coords_cond: torch.Tensor | None = None,
        is_dynamics_conditioned: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> tuple[DTensor, DTensor, DTensor]:
        q = static.q.repeat_interleave(multiplicity, 0) + self.r_to_q_trans(r)
        if self.use_dynamics_conditioning:
            assert coords_cond is not None and is_dynamics_conditioned is not None
            cond_contrib = self.cond_to_q_trans(coords_cond)
            if self.downscale_condition_contrib:
                assert dt is not None
                decay = self.dt_gate_embed(dt.expand(-1, multiplicity).reshape(-1, 1))
                gate = (decay * softplus(self.dt_gate_log_w)).sum(-1, keepdim=True)
                cond_contrib = cond_contrib * gate.unsqueeze(-1)
            cond_mask = ~is_dynamics_conditioned.expand(-1, multiplicity).reshape(-1, 1, 1)
            cond_contrib.masked_fill_(cond_mask, 0.0)
            q = q + cond_contrib
        c = static.c.repeat_interleave(multiplicity, 0)
        q_dt, c_dt = shard_along_atom([q, c], self.device_mesh, src_data_rank=None)
        return q_dt, c_dt, static.p


__all__ = [
    "AtomAttentionEncoderStaticContext",
    "DistributedAtomAttentionEncoder",
]
