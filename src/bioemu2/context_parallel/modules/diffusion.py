"""Canonical distributed BioMoaC structure diffusion module."""

from dataclasses import dataclass

from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.modules.diffusion import (
    DualStreamDiffusionModule,
    PairwiseConditioning,
)
from bioemu2.biomoa.ops.atom_token_ops import (
    gather_center_tokens_from_atomic as gather_center_tokens_from_atomic_tensor,
)
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.layers.transformer_layers import (
    DistributedDualStreamDiffusionLayer,
)
from bioemu2.context_parallel.modules.encoders import (
    AtomAttentionEncoderStaticContext,
    DistributedAtomAttentionEncoder,
)
from bioemu2.context_parallel.ops.distributed_primitives.flatten_and_unflatten import (
    shardwise_flatten_sharded,
    shardwise_unflatten_sharded,
)
from bioemu2.context_parallel.ops.readout_superposition import (
    DTensorReadoutSuperposition,
)
from bioemu2.context_parallel.ops.tiled_params_replicated import (
    LayerNormLinearParamsReplicatedTiled,
)
from bioemu2.context_parallel.ops.transition import DTensorAwareTransition
from bioemu2.context_parallel.utils.dtensor import (
    shard_along_atom,
    shard_along_token,
    shard_along_window,
)


@dataclass
class CPStaticContext:
    """Coordinate-independent inputs prepared for one denoising multiplicity.

    Attributes:
        multiplicity: Number of coordinate samples per system used to prepare the
            expanded atom and token shards.
        z: Conditioned token pairs, global shape ``(B, N_token, N_token, a_z)``.
        atom_encoder_context: Static atom-encoder outputs.
        atom_mask_window: Valid-atom mask, global shape ``(B, N_window, W)``.
        atom_ids_window: Parent-token indices, global shape ``(B * multiplicity, N_window, W)``.
        is_padded_atom_window: Padding mask, global shape ``(B * multiplicity, N_window, W)``.
        token_attention_mask: Valid-token mask, global shape ``(B, N_token)``.
        token_mask: Expanded valid-token mask, global shape ``(B * multiplicity, N_token)``.
        center_indices: Center-atom indices, global shape ``(B * multiplicity, N_token, 1)``.
        atom_mask: Plain mask for coordinate operations, shape ``(B, N_atom)``;
            separate from the distributed token masks.
        atom_pad_mask: Valid-atom mask, shape ``(B, N_atom)``.
        token_pad_mask: Valid-token mask, shape ``(B, N_token)``.
        token_center_atom_index: Center-atom indices, shape ``(B, N_token)``.
        coords_cond: Optional coordinates, shape ``(B, N_atom, 3)`` or
            ``(B * multiplicity, N_atom, 3)``.
        is_dynamics_conditioned: Optional mask, shape ``(B, 1)`` or ``(B, multiplicity)``.
        dt: Optional time lag, shape ``(B, 1)`` or ``(B, multiplicity)``.
    """

    multiplicity: int
    z: DTensor
    atom_encoder_context: AtomAttentionEncoderStaticContext
    atom_mask_window: DTensor
    atom_ids_window: DTensor
    is_padded_atom_window: DTensor
    token_attention_mask: DTensor  # Shard(0), Shard(1), Replicate(); (B, N_token).
    token_mask: DTensor  # Shard(0), Shard(1), Replicate(); (B * multiplicity, N_token).
    center_indices: DTensor
    atom_mask: Tensor
    atom_pad_mask: Tensor
    token_pad_mask: Tensor
    token_center_atom_index: Tensor
    coords_cond: Tensor | None
    is_dynamics_conditioned: Tensor | None
    dt: Tensor | None


class DistributedPairwiseConditioning(nn.Module):
    def __init__(self, serial: PairwiseConditioning, dist_manager: DistributedManager) -> None:
        super().__init__()
        device_mesh = dist_manager.device_mesh_subgroups
        self.dim_pairwise_init_proj = LayerNormLinearParamsReplicatedTiled(
            serial.dim_pairwise_init_proj, device_mesh
        )
        self.transitions = nn.ModuleList(
            DTensorAwareTransition(transition, device_mesh) for transition in serial.transitions
        )

    def forward(
        self,
        z_trunk: DTensor,
        relative_position_encoding: DTensor,
    ) -> DTensor:
        z = self.dim_pairwise_init_proj(z_trunk, relative_position_encoding)
        for transition in self.transitions:
            z = z + transition(z)
        return z


class DistributedDualStreamDiffusionModule(nn.Module):
    """CP implementation of BioMoaC's configured structure score module."""

    def __init__(
        self,
        serial: DualStreamDiffusionModule,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        if (
            serial.atom_feature_embedder.use_atom_encoder
            or serial.atom_feature_embedder.use_atom_to_token_aggregation
        ):
            raise NotImplementedError(
                "BioMoaC CP supports only the no-atom-encoder, no-aggregation path"
            )

        self.device_mesh = dist_manager.device_mesh_subgroups
        self.atoms_per_window_queries = serial.atoms_per_window_queries
        self.use_dynamics_conditioning = serial.use_dynamics_conditioning
        self.single_conditioner = serial.single_conditioner
        self.s_trunk_to_a = serial.s_trunk_to_a
        self.r_token_to_a = serial.r_token_to_a
        self.pairwise_conditioner = DistributedPairwiseConditioning(
            serial.pairwise_conditioner,
            dist_manager,
        )
        self.atom_feature_embedder = DistributedAtomAttentionEncoder(
            serial.atom_feature_embedder,
            self.device_mesh,
        )
        self.layers = nn.ModuleList(
            DistributedDualStreamDiffusionLayer(layer, dist_manager) for layer in serial.layers
        )
        self.readout = DTensorReadoutSuperposition(serial.readout, self.device_mesh)

    def compute_static_context(
        self,
        s_trunk: Tensor,
        z_trunk: DTensor,
        relative_position_encoding: DTensor,
        *,
        template_coords: Tensor,
        charge: Tensor,
        element: Tensor,
        name_encoding: Tensor,
        ref_space_uid: Tensor,
        atom_pad_mask: Tensor,
        atom_mask: Tensor,
        token_idx_at_atom: Tensor,
        token_pad_mask: Tensor,
        token_center_atom_index: Tensor,
        multiplicity: int,
        coords_cond: Tensor | None = None,
        is_dynamics_conditioned: Tensor | None = None,
        dt: Tensor | None = None,
    ) -> CPStaticContext:
        z = self.pairwise_conditioner(z_trunk, relative_position_encoding)
        atom_pad_mask = atom_pad_mask.bool()
        (atom_ids_dt, atom_mask_dt) = shard_along_atom(
            [token_idx_at_atom, atom_pad_mask],
            self.device_mesh,
            src_data_rank=None,
        )
        encoder = self.atom_feature_embedder.compute_static_context(
            atom_ref_pos=template_coords,
            ref_charge=charge,
            ref_element=element,
            ref_atom_name_chars=name_encoding,
            atom_ref_space_uid=ref_space_uid,
            atom_pad_mask=atom_pad_mask,
            atom_to_token_ids=token_idx_at_atom,
            atom_to_token_ids_dt=atom_ids_dt,
            atom_valid_mask_dt=atom_mask_dt,
            s_trunk=s_trunk,
            z=z_trunk,
        )

        batch, n_atoms = atom_pad_mask.shape
        window = self.atoms_per_window_queries
        n_windows = n_atoms // window
        (atom_mask_window_dt,) = shard_along_window(
            [atom_pad_mask.view(batch, n_windows, window)],
            self.device_mesh,
            src_data_rank=None,
        )
        atom_ids_mult = token_idx_at_atom.repeat_interleave(multiplicity, 0)
        is_padded_atom_mult = (~atom_pad_mask).repeat_interleave(multiplicity, 0)
        atom_ids_mult_dt, is_padded_atom_mult_dt = shard_along_atom(
            [atom_ids_mult, is_padded_atom_mult],
            self.device_mesh,
            src_data_rank=None,
        )
        atom_ids_mult_dt = shardwise_unflatten_sharded(
            atom_ids_mult_dt, axis=1, sizes=(n_windows, window)
        )
        is_padded_atom_mult_dt = shardwise_unflatten_sharded(
            is_padded_atom_mult_dt, axis=1, sizes=(n_windows, window)
        )

        token_pad_mask = token_pad_mask.bool()
        (token_attention_mask_dt,) = shard_along_token(
            [token_pad_mask],
            self.device_mesh,
            src_data_rank=None,
        )
        token_mask_mult = token_pad_mask.repeat_interleave(multiplicity, 0)
        center_indices_mult = token_center_atom_index.repeat_interleave(multiplicity, 0).unsqueeze(
            -1
        )
        token_mask_dt, center_indices_dt = shard_along_token(
            [token_mask_mult, center_indices_mult],
            self.device_mesh,
            src_data_rank=None,
        )
        return CPStaticContext(
            multiplicity=multiplicity,
            z=z,
            atom_encoder_context=encoder,
            atom_mask_window=atom_mask_window_dt,
            atom_ids_window=atom_ids_mult_dt,
            is_padded_atom_window=is_padded_atom_mult_dt,
            token_attention_mask=token_attention_mask_dt,
            token_mask=token_mask_dt,
            center_indices=center_indices_dt,
            atom_mask=atom_mask,
            atom_pad_mask=atom_pad_mask,
            token_pad_mask=token_pad_mask,
            token_center_atom_index=token_center_atom_index,
            coords_cond=coords_cond,
            is_dynamics_conditioned=is_dynamics_conditioned,
            dt=dt,
        )

    def denoise(
        self,
        context: CPStaticContext,
        s_inputs: Tensor,
        s_trunk: Tensor,
        r_noisy: Tensor,
        times: Tensor,
        multiplicity: int,
    ) -> dict[str, DTensor]:
        assert r_noisy.shape[0] == context.atom_pad_mask.shape[0] * multiplicity, (
            f"{r_noisy.shape[0]} != {context.atom_pad_mask.shape[0]} * {multiplicity}"
        )
        atom_mask = context.atom_mask.repeat_interleave(multiplicity, 0)
        r_noisy = r_noisy.masked_fill(~atom_mask[:, :, None], 0.0)

        dt = None
        is_dynamics_conditioned = None
        if self.use_dynamics_conditioning:
            assert context.dt is not None and context.is_dynamics_conditioned is not None
            dt = context.dt.expand(-1, multiplicity).reshape(-1, 1)
            is_dynamics_conditioned = context.is_dynamics_conditioned.expand(
                -1, multiplicity
            ).reshape(-1, 1)
        c, _ = self.single_conditioner(
            times=times,
            s_trunk=s_trunk.repeat_interleave(multiplicity, 0),
            s_inputs=s_inputs.repeat_interleave(multiplicity, 0),
            dt=dt,
            is_dynamics_conditioned=is_dynamics_conditioned,
        )
        (c_dt,) = shard_along_token([c], self.device_mesh, src_data_rank=None)
        coords_cond = context.coords_cond
        if coords_cond is not None and coords_cond.shape[0] == context.atom_pad_mask.shape[0]:
            coords_cond = coords_cond.repeat_interleave(multiplicity, 0)
        q_dt, _, p_dt = self.atom_feature_embedder.apply_dynamic(
            context.atom_encoder_context,
            r=r_noisy,
            multiplicity=multiplicity,
            coords_cond=coords_cond,
            is_dynamics_conditioned=context.is_dynamics_conditioned,
            dt=context.dt,
        )
        window = self.atoms_per_window_queries
        n_windows = context.atom_pad_mask.shape[1] // window
        q_dt = shardwise_unflatten_sharded(q_dt, axis=1, sizes=(n_windows, window))
        r_noisy_token = gather_center_tokens_from_atomic_tensor(
            r_noisy,
            context.token_center_atom_index,
            context.token_pad_mask,
            multiplicity,
        )
        a = self.s_trunk_to_a(s_trunk).repeat_interleave(multiplicity, 0)
        a = a + self.r_token_to_a(r_noisy_token)
        (a_dt,) = shard_along_token([a], self.device_mesh, src_data_rank=None)

        for layer in self.layers:
            a_dt, q_dt = layer(
                a_dt,
                context.z,
                q_dt,
                p_dt,
                c_dt,
                context.atom_mask_window,
                context.token_attention_mask,
                context.token_mask,
                context.atom_ids_window,
                context.is_padded_atom_window,
                context.center_indices,
                multiplicity,
            )

        q_dt = shardwise_flatten_sharded(q_dt, start_dim=1, end_dim=2)
        r_update = self.readout(
            a_dt,
            q_dt,
            shardwise_flatten_sharded(
                context.atom_ids_window,
                start_dim=1,
                end_dim=2,
            ),
        )
        return {"r_update": r_update, "token_a": a_dt}

    def forward(
        self,
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: DTensor,
        r_noisy: Tensor,
        times: Tensor,
        relative_position_encoding: DTensor,
        feats: dict[str, Tensor],
        multiplicity: int,
    ) -> dict[str, DTensor]:
        context = self.compute_static_context(
            s_trunk,
            z_trunk,
            relative_position_encoding,
            template_coords=feats["template_coords"],
            charge=feats["charge"],
            element=feats["element"],
            name_encoding=feats["name_encoding"],
            ref_space_uid=feats["ref_space_uid"],
            atom_pad_mask=feats["atom_pad_mask"],
            atom_mask=feats["atom_mask"],
            token_idx_at_atom=feats["token_idx_at_atom"],
            token_pad_mask=feats["token_pad_mask"],
            token_center_atom_index=feats["token_center_atom_index"],
            multiplicity=multiplicity,
            coords_cond=feats.get("coords_cond"),
            is_dynamics_conditioned=feats.get("is_dynamics_conditioned"),
            dt=feats.get("dt"),
        )
        return self.denoise(
            context,
            s_inputs,
            s_trunk,
            r_noisy,
            times,
            multiplicity,
        )
