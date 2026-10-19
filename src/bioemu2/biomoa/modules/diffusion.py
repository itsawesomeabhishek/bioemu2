"""Dual-stream atom/token diffusion module and its conditioning layers."""

import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from bioemu2.biomoa.modules.encoders import (
    AtomAttentionEncoder,
    AtomAttentionEncoderStaticContext,
    FourierEmbedding,
)
from bioemu2.biomoa.modules.time import DampedHarmonicTimeEmbedding
from bioemu2.biomoa.modules.transformer import (
    ConditionedSlidingWindowTransformerLayer,
    ConditionedTransformerLayer,
    ada_ln,
    ada_scale,
)
from bioemu2.biomoa.ops.atom_token_ops import (
    broadcast_tokens_to_atoms,
    gather_center_tokens_from_atomic,
)
from bioemu2.biomoa.ops.transition import Transition
from bioemu2.biomoa.utils import LinearNoBias


@dataclass
class StaticContext:
    """Coordinate-independent inputs returned by ``compute_static_context``.

    Attributes:
        z: Conditioned token-pair representations, shape
            ``(B, N_token, N_token, a_z)``.
        atom_encoder_context: Static atom-encoder outputs consumed by the dynamic atom pass.
        atom_mask: Shape ``(B, N)``; ``True`` means the atom participates in
            coordinate operations. When sampling from a ``System``, every
            topology atom participates.
        atom_pad_mask: Shape ``(B, N)``; ``True`` means the atom is not padding.
        token_pad_mask: Shape ``(B, N_token)``; ``True`` means the token is not padding.
        atom_to_token_ids: Parent-token index for each atom, shape ``(B, N)``.
        is_padded_atom: Shape ``(B, N)``; ``True`` means the atom is padding.
        token_center_atom_index: Center-atom index for each token, shape
            ``(B, N_token)``.
        coords_cond: Optional conditioning coordinates, shape ``(B, N, 3)`` or
            ``(B * multiplicity, N, 3)`` after multiplicity expansion.
        is_dynamics_conditioned: Optional shape ``(B, 1)`` or ``(B, multiplicity)``;
            ``True`` means the corresponding sample uses ``coords_cond``.
        dt: Time lag between ``coords_cond`` and the target, shape ``(B, 1)`` or
            ``(B, multiplicity)``.
    """

    z: Tensor
    atom_encoder_context: AtomAttentionEncoderStaticContext
    atom_mask: Tensor
    atom_pad_mask: Tensor
    token_pad_mask: Tensor
    atom_to_token_ids: Tensor
    is_padded_atom: Tensor
    token_center_atom_index: Tensor
    coords_cond: Tensor | None
    is_dynamics_conditioned: Tensor | None
    dt: Tensor | None


class DualStreamDiffusionModule(nn.Module):
    """Diffusion module."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        s_input_dim: int,
        a_s: int | None = None,
        a_z: int | None = None,
        num_layers: int = 16,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        sigma_data: float = 16.0,
        dim_fourier: int = 256,
        atom_attention_heads: int = 4,
        token_attention_heads: int = 16,
        conditioning_transition_layers: int = 2,
        attn_pair_bias_tile_size: int | None = None,
        coupling_transition_expansion_factor: int = 2,
        transition_chunk_size: int | None = None,
        use_lns: bool = True,
        use_dynamics_conditioning: bool = False,
        downscale_condition_contrib: bool = False,
        dt_embed_num_timescales: int = 32,
    ) -> None:
        """Initialize the diffusion module.

        Args:
            token_s: The single representation dimension.
            token_z: The pair representation dimension.
            atom_s: The atom single representation dimension.
            atom_z: The atom pair representation dimension.
            s_input_dim: Dimension of the ``InputEmbedder`` output
                (``token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes``).
            a_s: Internal single representation dimension for the diffusion layers.
                Defaults to ``token_s`` when ``None``.
            a_z: Internal pair representation dimension for the diffusion layers.
                Defaults to ``token_z`` when ``None``.
            num_layers: The number of layers.
            atoms_per_window_queries: The number of atoms per window for queries.
            atoms_per_window_keys: The number of atoms per window for keys.
            sigma_data: The standard deviation of the data distribution.
            dim_fourier: The dimension of the fourier embedding.
            atom_attention_heads: The number of attention heads for atom attention.
            token_attention_heads: The number of attention heads for token attention.
            conditioning_transition_layers: The number of transition layers for conditioning.
            attn_pair_bias_tile_size: Optional tile size for tiled pair-bias variants.
            coupling_transition_expansion_factor: Hidden-dimension expansion factor for the
                cross-stream coupling mixer MLPs.
            use_lns: Whether to use LayerNorm Scaling (arXiv:2502.05795).
            use_dynamics_conditioning: Enable MD-sequential (ITO) conditioning. The lag ``dt``
                is injected into the token single representation and ``coords_cond`` into the
                atom queries, both gated per-sample by ``is_dynamics_conditioned``.
            downscale_condition_contrib: Gate the per-atom ``coords_cond`` contribution by a
                learned, monotonically decreasing function of the lag ``dt``.
            dt_embed_num_timescales: Number of log-spaced timescales in the ``dt`` embeddings.
        """
        super().__init__()

        a_s = a_s if a_s is not None else token_s
        a_z = a_z if a_z is not None else token_z

        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys
        self.sigma_data = sigma_data
        self.use_dynamics_conditioning = use_dynamics_conditioning

        self.s_trunk_to_a = nn.Sequential(nn.LayerNorm(token_s), LinearNoBias(token_s, a_s))

        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            token_s=token_s,
            s_input_dim=s_input_dim,
            a_s=a_s,
            dim_fourier=dim_fourier,
            num_transitions=conditioning_transition_layers,
            use_dt_conditioning=use_dynamics_conditioning,
            dt_embed_num_timescales=dt_embed_num_timescales,
        )

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            a_z=a_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
            transition_chunk_size=transition_chunk_size,
        )

        self.atom_feature_embedder = AtomAttentionEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            structure_prediction=True,
            use_atom_encoder=False,
            use_atom_to_token_aggregation=False,
            use_dynamics_conditioning=use_dynamics_conditioning,
            downscale_condition_contrib=downscale_condition_contrib,
            dt_embed_num_timescales=dt_embed_num_timescales,
        )

        self.layers = nn.ModuleList(
            [
                DualStreamDiffusionLayer(
                    a_s=a_s,
                    a_z=a_z,
                    atom_s=atom_s,
                    atom_z=atom_z,
                    atoms_per_window_queries=atoms_per_window_queries,
                    atoms_per_window_keys=atoms_per_window_keys,
                    atom_attention_heads=atom_attention_heads,
                    token_attention_heads=token_attention_heads,
                    attn_pair_bias_tile_size=attn_pair_bias_tile_size,
                    layer_index=i,
                    use_lns=use_lns,
                    coupling_transition_expansion_factor=coupling_transition_expansion_factor,
                )
                for i in range(num_layers)
            ]
        )

        self.r_token_to_a = LinearNoBias(3, a_s, bias=False)

        self.readout = ReadoutSuperposition(atom_s=atom_s, token_s=a_s)

    def compute_static_context(
        self,
        s_trunk: Tensor,
        z_trunk: Tensor,
        relative_position_encoding: Tensor,
        feats: dict[str, Tensor],
    ) -> StaticContext:
        # Combine trunk pairs with pair representations build from the input features.
        z = self.pairwise_conditioner(
            z_trunk=z_trunk, token_rel_pos_feats=relative_position_encoding
        )  # (B, N_token, N_token, a_z)

        # Embed atom features and construct atom pair conditioning for the atom attention layers.
        # Time conditioning signal will enter the atom representations via AdaLN and AdaScale based on the single representations `s`.
        # `AtomAttentionEncoder` is configured with use_atom_encoder=False and
        # use_atom_to_token_aggregation=False, so `q` carries the raw atom features
        # (no per-window self-attention applied).
        atom_encoder_context: AtomAttentionEncoderStaticContext = (
            self.atom_feature_embedder.compute_static_context(
                atom_ref_pos=feats["template_coords"],
                ref_charge=feats["charge"],
                ref_element=feats["element"],
                ref_atom_name_chars=feats["name_encoding"],
                atom_ref_space_uid=feats["ref_space_uid"],
                atom_pad_mask=feats["atom_pad_mask"],
                atom_to_token_ids=feats["token_idx_at_atom"],
                token_pad_mask=feats["token_pad_mask"],
                s_trunk=s_trunk,
                z=z_trunk,
            )
        )

        atom_pad_mask = feats["atom_pad_mask"]
        token_pad_mask = feats["token_pad_mask"]
        atom_to_token_ids = feats["token_idx_at_atom"]
        is_padded_atom = ~atom_pad_mask  # (B, N)
        return StaticContext(
            z=z,
            atom_encoder_context=atom_encoder_context,
            atom_mask=feats["atom_mask"],
            atom_pad_mask=atom_pad_mask,
            token_pad_mask=token_pad_mask,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            token_center_atom_index=feats["token_center_atom_index"],
            coords_cond=feats.get("coords_cond"),
            is_dynamics_conditioned=feats.get("is_dynamics_conditioned"),
            dt=feats.get("dt"),
        )

    def denoise(
        self,
        context: StaticContext,
        s_inputs: Tensor,
        s_trunk: Tensor,
        r_noisy: Tensor,
        times: Tensor,
        multiplicity: int,
    ) -> dict[str, Tensor]:
        assert r_noisy.shape[0] == context.atom_pad_mask.shape[0] * multiplicity, (
            f"{r_noisy.shape[0]} != {context.atom_pad_mask.shape[0]} * {multiplicity}"
        )
        atom_mask = context.atom_mask.repeat_interleave(multiplicity, 0)
        r_noisy = r_noisy.masked_fill(~atom_mask[:, :, None], 0.0)

        # Combine trunk singles with representations build from the input features
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
        )  # (B * multiplicity, N_token, a_s)

        coords_cond = context.coords_cond
        is_dynamics_conditioned = context.is_dynamics_conditioned
        if coords_cond is not None and coords_cond.shape[0] == context.atom_pad_mask.shape[0]:
            coords_cond = coords_cond.repeat_interleave(multiplicity, 0)

        _, q, _, p = self.atom_feature_embedder.apply_dynamic(
            context.atom_encoder_context,
            r=r_noisy,
            multiplicity=multiplicity,
            coords_cond=coords_cond,
            is_dynamics_conditioned=is_dynamics_conditioned,
            dt=context.dt,
        )  # q: (B * multiplicity, N, atom_s), p: (B, NW, W, H, atom_z)

        a = self.s_trunk_to_a(s_trunk)  # (B, N_token, a_s)
        a = a.repeat_interleave(multiplicity, 0)  # (B * multiplicity, N_token, a_s)

        # Embed the noisy token positions.
        r_noisy_token = gather_center_tokens_from_atomic(
            r_noisy,
            context.token_center_atom_index,
            context.token_pad_mask,
            multiplicity,
        )  # (B * multiplicity, N_token, 3)
        a = a + self.r_token_to_a(r_noisy_token)  # (B * multiplicity, N_token, a_s)

        for layer in self.layers:
            a, q = layer(
                a,
                context.z,
                q,
                p,
                c,
                context.atom_pad_mask,
                context.token_pad_mask,
                context.atom_to_token_ids,
                context.is_padded_atom,
                context.token_center_atom_index,
                multiplicity,
            )

        r_update = self.readout(
            a=a,
            q=q,
            atom_to_token_ids=context.atom_to_token_ids,
            is_padded_atom=context.is_padded_atom,
            multiplicity=multiplicity,
        )  # (B * multiplicity, N, 3)

        return {"r_update": r_update, "token_a": a}

    def forward(
        self,
        s_inputs: Tensor,
        s_trunk: Tensor,
        z_trunk: Tensor,
        r_noisy: Tensor,
        times: Tensor,
        relative_position_encoding: Tensor,
        feats: dict[str, Tensor],
        multiplicity: int = 1,
    ) -> dict[str, Tensor]:
        context = self.compute_static_context(
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            relative_position_encoding=relative_position_encoding,
            feats=feats,
        )
        return self.denoise(
            context=context,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            r_noisy=r_noisy,
            times=times,
            multiplicity=multiplicity,
        )


class SingleConditioning(nn.Module):
    """Single conditioning layer which outputs tokens of dimension ``a_s`` (defaults to ``token_s``).

    Concatenates s_trunk (from the sequence embedder) and s_inputs (from the
    InputEmbedder, which already contains one-hot / continuous per-token
    features), then projects to a_s via LayerNorm + Linear.
    """

    def __init__(
        self,
        sigma_data: float,
        token_s: int = 384,
        s_input_dim: int = 455,
        a_s: int | None = None,
        dim_fourier: int = 256,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        transition_chunk_size: int | None = None,
        eps: float = 1e-20,
        use_dt_conditioning: bool = False,
        dt_embed_num_timescales: int = 32,
    ) -> None:
        """Initialize the single conditioning layer.

        Args:
            sigma_data: The data sigma.
            token_s: The input single representation dimension.
            s_input_dim: Dimension of the ``InputEmbedder`` output
                (``token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes``).
            a_s: The output single representation dimension. Defaults to ``token_s``
                when ``None``.
            dim_fourier: The fourier embeddings dimension.
            num_transitions: The number of transition layers.
            transition_expansion_factor: The transition expansion factor.
            transition_chunk_size: Hidden-dim chunk size for the transition
                layers. ``None`` disables chunking.
            eps: The epsilon value.
            use_dt_conditioning: If True, add an embedded time lag ``dt`` to the token-level
                single representation, gated per-sample by ``is_dynamics_conditioned``.
            dt_embed_num_timescales: Number of log-spaced timescales in the damped-harmonic
                ``dt`` embedding. Only used when ``use_dt_conditioning`` is True.
        """
        super().__init__()
        self.eps = eps
        self.sigma_data = sigma_data

        a_s = a_s if a_s is not None else token_s

        # input_dim = s_trunk(token_s) + s_inputs(s_input_dim)
        input_dim = token_s + s_input_dim
        self.norm_single = nn.LayerNorm(input_dim)
        self.single_embed = nn.Linear(input_dim, a_s)
        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = nn.LayerNorm(dim_fourier)
        self.fourier_to_single = LinearNoBias(dim_fourier, a_s)

        self.use_dt_conditioning = use_dt_conditioning
        if self.use_dt_conditioning:
            self.dt_embed = DampedHarmonicTimeEmbedding(num_timescales=dt_embed_num_timescales)
            self.dt_embed_to_single = LinearNoBias(self.dt_embed.output_dim, a_s)

        self.transitions = nn.ModuleList(
            [
                Transition(
                    dim=a_s,
                    hidden=transition_expansion_factor * a_s,
                    chunk_size=transition_chunk_size,
                )
                for _ in range(num_transitions)
            ]
        )

    def forward(
        self,
        *,
        times: Tensor,
        s_trunk: Tensor,
        s_inputs: Tensor,
        dt: Tensor | None = None,
        is_dynamics_conditioned: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # s_inputs already contains one-hot residue type, MSA profile, deletion
        # mean, and pocket one-hot from the InputEmbedder — just concatenate.
        s = torch.cat((s_trunk, s_inputs), dim=-1)  # (B, N_token, token_s + s_input_dim)
        s = self.single_embed(self.norm_single(s))  # (B, N_token, a_s)
        fourier_embed = self.fourier_embed(times)  # (B, dim_fourier)
        normed_fourier = self.norm_fourier(fourier_embed)  # (B, dim_fourier)
        fourier_to_single = self.fourier_to_single(normed_fourier)  # (B, a_s)

        s = rearrange(fourier_to_single, "b d -> b 1 d") + s  # (B, N_token, a_s)

        if self.use_dt_conditioning:
            assert dt is not None and is_dynamics_conditioned is not None
            dt_embed = self.dt_embed(dt)
            dt_to_single = self.dt_embed_to_single(dt_embed) * is_dynamics_conditioned.to(
                dt_embed.dtype
            )
            s = rearrange(dt_to_single, "b d -> b 1 d") + s

        for transition in self.transitions:
            s = transition(s) + s  # (B, N_token, a_s)

        return s, normed_fourier


class PairwiseConditioning(nn.Module):
    """Pairwise conditioning layer with configurable output dimension ``a_z``. Defaults to ``token_z`` when ``a_z`` is None."""

    def __init__(
        self,
        token_z: int,
        dim_token_rel_pos_feats: int,
        a_z: int | None = None,
        num_transitions: int = 2,
        transition_expansion_factor: int = 2,
        transition_chunk_size: int | None = None,
    ) -> None:
        """Initialize the pairwise conditioning layer.

        Args:
            token_z: The input pair representation dimension.
            dim_token_rel_pos_feats: The token relative position features dimension.
            a_z: The output pair representation dimension. Defaults to ``token_z``
                when ``None``.
            num_transitions: The number of transition layers.
            transition_expansion_factor: The transition expansion factor.
            transition_chunk_size: Hidden-dim chunk size for the transition
                layers. ``None`` disables chunking.
        """
        super().__init__()

        a_z = a_z if a_z is not None else token_z

        self.dim_pairwise_init_proj = nn.Sequential(
            nn.LayerNorm(token_z + dim_token_rel_pos_feats),
            LinearNoBias(token_z + dim_token_rel_pos_feats, a_z),
        )

        self.transitions = nn.ModuleList(
            [
                Transition(
                    dim=a_z,
                    hidden=transition_expansion_factor * a_z,
                    chunk_size=transition_chunk_size,
                )
                for _ in range(num_transitions)
            ]
        )

    def forward(
        self,
        z_trunk: torch.Tensor,
        token_rel_pos_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_trunk and token_rel_pos_feats are concatenated along the feature dimension, then independently projected for each token pair using the same MLP.
        There is no mixing of information across different token pairs in this module.

        Args:
            z_trunk: Pairwise embeddings from the sequence trunk, shape (B, N_token, N_token, token_z).
            token_rel_pos_feats: Relative position features, shape (B, N_token, N_token, dim_token_rel_pos_feats). These describe
                how the tokens relate in sequence, chain, entity, and symmetry space.

        Returns:
            Conditioned pairwise embeddings, shape (B, N_token, N_token, a_z).

        """
        z = torch.cat(
            (z_trunk, token_rel_pos_feats), dim=-1
        )  # (B, N_token, N_token, token_z + rel_pos)
        z = self.dim_pairwise_init_proj(z)  # (B, N_token, N_token, a_z)

        for transition in self.transitions:
            z = transition(z) + z  # (B, N_token, N_token, a_z)

        return z


class ReadoutSuperposition(nn.Module):
    """Readout layer which produces atomic updates by superposing token representations based on the atom-to-token mapping."""

    def __init__(self, atom_s: int, token_s: int, hidden_dim: int | None = 384) -> None:
        """Initialize the readout layer.

        Args:
            atom_s: The atom single representation dimension.
            token_s: The token single representation dimension.
            hidden_dim: Hidden projection dimension; defaults to ``atom_s`` when ``None``.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = atom_s
        self.s_to_hidden = nn.Linear(token_s, hidden_dim, bias=False)
        self.q_to_hidden = nn.Linear(atom_s, hidden_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim, bias=False), nn.Linear(hidden_dim, 3, bias=False)
        )

    def forward(
        self,
        a: Tensor,  # (B * M, T, a_s)
        q: Tensor,  # (B * M, N, atom_s)
        atom_to_token_ids: Tensor,  # (B, N)
        is_padded_atom: Tensor,  # (B, N) bool
        multiplicity: int = 1,
    ) -> Tensor:
        """Compute per-atom updates by superposing token and atom activations.

        Args:
            a: Token single representation, ``(B * M, T, a_s)``.
            q: Atom single representation, ``(B * M, N, atom_s)``.
            atom_to_token_ids: Maps each atom to its parent token, ``(B, N)``.
            is_padded_atom: True for padded (invalid) atoms, ``(B, N)``.
            multiplicity: Number of noise replicates per batch element.

        Returns:
            Per-atom coordinate updates, ``(B * M, N, 3)``.
        """

        # No local float32 guard needed: broadcast_tokens_to_atoms is a gather.
        a_to_q = self.s_to_hidden(a)  # (B*M, N_token, hidden)
        a_to_q = broadcast_tokens_to_atoms(a_to_q, atom_to_token_ids, is_padded_atom, multiplicity)

        q = self.q_to_hidden(q) + a_to_q  # (B * multiplicity, N, hidden_dim)
        out = self.to_out(q)  # (B * multiplicity, N, 3)

        return out  # (B * multiplicity, N, 3)


class DualStreamCoupling(nn.Module):
    """Cross-stream coupling transitions joining the token and atom residual streams.

    After the token and atom blocks each update their own stream, this module mixes
    information across streams: it folds token activations into the atom stream
    (``s_to_q``) and atom activations into the token stream (``q_to_s``), each gated by
    its own DiT conditioning (adaLN-Zero, ``x + gamma * mixer(lns_scale * ada_ln(x))``).

    These transitions became independent of the per-block attention/MLP conditioning once
    the latter moved inside :class:`ConditionedTransformerLayer`, so they own their own
    conditioning producers (``q_coupling_cond``/``s_coupling_cond``) and live in this
    separate module rather than sharing a fused projection with the blocks.
    """

    def __init__(
        self,
        a_s: int,
        atom_s: int,
        lns_scale: float = 1.0,
        transition_expansion_factor: int = 2,
    ) -> None:
        """Initialize the dual-stream coupling.

        Args:
            a_s: Token single representation dimension.
            atom_s: Atom single representation dimension.
            lns_scale: LayerNorm-Scaling factor multiplying each ``ada_ln`` output
                (arXiv:2502.05795); ``1.0`` disables it.
            transition_expansion_factor: Hidden-dimension expansion factor for the
                mixer MLPs (``hidden = factor * (a_s + atom_s)``).
        """
        super().__init__()
        self.lns_scale = lns_scale

        # MLP mixing token activations into atom stream.
        self.s_to_q_mixer = Transition(
            dim=a_s + atom_s,
            hidden=transition_expansion_factor * (a_s + atom_s),
            out_dim=atom_s,
            use_norm=False,
        )

        # MLP mixing atom activations into token stream.
        self.q_to_s_mixer = Transition(
            dim=a_s + atom_s,
            hidden=transition_expansion_factor * (a_s + atom_s),
            out_dim=a_s,
            use_norm=False,
        )

        # Coupling conditioning for the token->atom transition: alpha3/beta3/gamma3.
        self.q_coupling_cond = nn.Sequential(
            nn.LayerNorm(a_s), nn.SiLU(), nn.Linear(a_s, 3 * atom_s + 2 * a_s)
        )

        # Coupling conditioning for the atom->token transition: alpha3/beta3/gamma3.
        self.s_coupling_cond = nn.Sequential(
            nn.LayerNorm(a_s), nn.SiLU(), nn.Linear(a_s, 3 * a_s + 2 * atom_s)
        )

    def forward(
        self,
        s: torch.Tensor,  # (B * M, T, a_s)
        q: torch.Tensor,  # (B * M, N, atom_s)
        c: torch.Tensor,  # (B * M, T, a_s) token conditioning
        c_to_q: torch.Tensor,  # (B * M, N, a_s) atom-broadcast conditioning
        atom_to_token_ids: torch.Tensor,  # (B, N)
        is_padded_atom: torch.Tensor,  # (B, N) bool
        token_center_atom_index: torch.Tensor,  # (B, T)
        token_pad_mask: torch.Tensor,  # (B, T)
        multiplicity: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Apply both cross-stream coupling transitions and return updated ``(s, q)``.

        Args:
            s: Updated token single representation, ``(B * multiplicity, T, a_s)``.
            q: Updated atom single representation, ``(B * multiplicity, N, atom_s)``.
            c: Token conditioning activations, ``(B * multiplicity, T, a_s)``.
            c_to_q: Atom-broadcast conditioning activations, ``(B * multiplicity, N, a_s)``.
            atom_to_token_ids: Maps each atom to its parent token, ``(B, N)``.
            is_padded_atom: True for padded (invalid) atoms, ``(B, N)``.
            token_center_atom_index: Center atom index per token, ``(B, T)``.
            token_pad_mask: Token validity mask, ``(B, T)``.
            multiplicity: Number of noise replicates per batch element.

        Returns:
            Updated ``(s, q)`` activations.
        """
        atom_s = q.shape[-1]
        a_s = s.shape[-1]

        alpha3_q, beta3_q, gamma3_q = torch.split(
            self.q_coupling_cond(c_to_q),
            split_size_or_sections=[a_s + atom_s, a_s + atom_s, atom_s],
            dim=-1,
        )  # 2 x (B * multiplicity, N, a_s + atom_s), (B * multiplicity, N, atom_s)

        alpha3_s, beta3_s, gamma3_s = torch.split(
            self.s_coupling_cond(c),
            split_size_or_sections=[a_s + atom_s, a_s + atom_s, a_s],
            dim=-1,
        )

        # broadcast_tokens_to_atoms is a precision-agnostic gather; no float32 guard needed.
        s_to_q = broadcast_tokens_to_atoms(
            s, atom_to_token_ids, is_padded_atom, multiplicity
        )  # (B * multiplicity, N, a_s)

        s_to_q = torch.cat([s_to_q, q], dim=-1)  # (B * multiplicity, N, a_s + atom_s)
        s_to_q = ada_scale(
            self.s_to_q_mixer(self.lns_scale * ada_ln(s_to_q, alpha=alpha3_q, beta=beta3_q)),
            gamma3_q,
        )  # (B * multiplicity, N, atom_s)

        q_to_s = gather_center_tokens_from_atomic(
            q,
            token_center_atom_index,
            token_pad_mask,
            multiplicity,
        )  # (B * multiplicity, N_token, a_s)

        q_to_s = torch.cat([q_to_s, s], dim=-1)  # (B * multiplicity, N_token, atom_s + a_s)
        q_to_s = ada_scale(
            self.q_to_s_mixer(self.lns_scale * ada_ln(q_to_s, alpha=alpha3_s, beta=beta3_s)),
            gamma3_s,
        )  # (B * multiplicity, N_token, a_s)

        q = q + s_to_q  # (B * multiplicity, N, atom_s)
        s = s + q_to_s  # (B * multiplicity, N_token, a_s)

        return s, q


class DualStreamDiffusionLayer(nn.Module):
    """Dual-stream diffusion layer with coupled token and atom residual streams.

    Maintains two parallel residual streams (token-level and atom-level), each
    with attention and MLP sub-layers, joined by cross-stream coupling
    transitions at the end of each block.
    """

    def __init__(
        self,
        a_s: int,
        a_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_attention_heads: int = 4,
        token_attention_heads: int = 16,
        attn_pair_bias_tile_size: int | None = None,
        layer_index: int = 0,
        use_lns: bool = True,
        coupling_transition_expansion_factor: int = 2,
    ) -> None:
        super().__init__()

        # LayerNorm Scaling (arXiv:2502.05795): scale ada_ln outputs by 1/sqrt(layer_index)
        self.lns_scale = 1.0 / math.sqrt(layer_index + 1) if use_lns else 1.0

        # Cross-stream coupling transitions (token<->atom); own their conditioning.
        self.coupling = DualStreamCoupling(
            a_s=a_s,
            atom_s=atom_s,
            lns_scale=self.lns_scale,
            transition_expansion_factor=coupling_transition_expansion_factor,
        )

        # Atom residual stream: windowed attention + MLP; conditioning produced internally from c_to_q.
        self.atom_block = ConditionedSlidingWindowTransformerLayer(
            dim=atom_s,
            dim_pairwise=atom_z,
            dim_cond=a_s,
            heads=atom_attention_heads,
            transition_hidden=4 * atom_s,
            window_queries=atoms_per_window_queries,
            window_keys=atoms_per_window_keys,
            attn_pair_bias_tile_size=attn_pair_bias_tile_size,
            lns_scale=self.lns_scale,
        )

        # Token residual stream: full attention + MLP; conditioning produced internally from c.
        self.token_block = ConditionedTransformerLayer(
            dim=a_s,
            dim_pairwise=a_z,
            dim_cond=a_s,
            heads=token_attention_heads,
            transition_hidden=4 * a_s,
            is_sliding_window=False,
            attn_pair_bias_tile_size=attn_pair_bias_tile_size,
            lns_scale=self.lns_scale,
        )

    def forward(
        self,
        s: torch.Tensor,  # (B * M, T, a_s)
        z: torch.Tensor,  # (B, T, T, a_z)
        q: torch.Tensor,  # (B * M, N, atom_s)
        p: torch.Tensor,  # (B, NW, W, H, atom_z)
        c: torch.Tensor,  # (B * M, T, a_s)
        atom_mask: torch.Tensor,  # (B, N)
        token_pad_mask: torch.Tensor,  # (B, T)
        atom_to_token_ids: torch.Tensor,  # (B, N)
        is_padded_atom: torch.Tensor,  # (B, N) bool
        token_center_atom_index: torch.Tensor,  # (B, T)
        multiplicity: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """Run one dual-stream block.

        Args:
            s: Token single representation, ``(B * M, T, a_s)``.
            z: Token pair representation, ``(B, T, T, a_z)``.
            q: Atom single representation, ``(B * M, N, atom_s)``.
            p: Atom pair conditioning, ``(B, NW, W, H, atom_z)``.
            c: Conditioning activations, ``(B * M, T, a_s)``.
            atom_mask: Atom validity mask, ``(B, N)``.
            token_pad_mask: Token validity mask, ``(B, T)``.
            atom_to_token_ids: Maps each atom to its parent token, ``(B, N)``.
            is_padded_atom: True for padded (invalid) atoms, ``(B, N)``.
            token_center_atom_index: Center atom index per token, ``(B, T)``.
            multiplicity: Number of noise replicates per batch element.

        Returns:
            Updated ``(s, q)`` activations.
        """
        B = z.shape[0]

        assert p.shape[0] == z.shape[0], (
            f"Expected same batch size for atomic pair conditioning `p` and token pair conditioning `z`, received {p.shape[0]} and {z.shape[0]}"
        )
        assert s.shape[0] == q.shape[0], (
            f"Expected same batch size for s and q, received {s.shape[0]} and {q.shape[0]}"
        )
        assert q.shape[0] == B * multiplicity, (
            f"Expected q batch size to be `B x multiplicity = ({B * multiplicity})`, received {q.shape[0]} for {B = } and {multiplicity = }"
        )

        # broadcast_tokens_to_atoms is a precision-agnostic gather; no float32 guard needed.
        c_to_q = broadcast_tokens_to_atoms(
            c, atom_to_token_ids, is_padded_atom, multiplicity
        )  # (B * multiplicity, N, a_s)

        # Residual stream for atom tokens.
        q = self.atom_block(
            q,
            p,
            c_to_q,
            mask=atom_mask,
            multiplicity=multiplicity,
        )  # (B * multiplicity, N, atom_s)

        # Residual stream for coarse tokens.

        # APB expands over multiplicity internally, so pass the raw [B, N_token] mask.
        token_mask = token_pad_mask  # (B, N_token)

        s = self.token_block(
            s,
            z,
            c,
            mask=token_mask,
            multiplicity=multiplicity,
        )  # (B * multiplicity, N_token, a_s)

        # Coupling transitions.
        s, q = self.coupling(
            s,
            q,
            c,
            c_to_q,
            atom_to_token_ids,
            is_padded_atom,
            token_center_atom_index,
            token_pad_mask,
            multiplicity,
        )

        return s, q
