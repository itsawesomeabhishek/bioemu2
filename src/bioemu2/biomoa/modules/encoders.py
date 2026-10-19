# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Atom attention encoder and Fourier noise-level embedding for BioMoa."""

from dataclasses import dataclass
from math import pi

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import softplus
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from bioemu2.biomoa.modules.time import MultiExponentialTimeEmbedding
from bioemu2.biomoa.modules.transformer import ConditionedSlidingWindowTransformerLayer
from bioemu2.biomoa.ops.atom_token_ops import (
    broadcast_token_pair_to_atom_window,
    broadcast_tokens_to_atoms,
    scatter_mean_atoms_to_tokens,
    window_atom_single_to_pair_keys,
    windowed_atom_pair_geometry,
)
from bioemu2.biomoa.utils import LinearNoBias
from bioemu2.data_interface.const import NUM_TOKENS

NUM_TOKEN_TYPES: int = NUM_TOKENS
NUM_BINS_ATOM: int = 64
NUM_MSA_TOKENS: int = NUM_TOKENS
LEN_POCKET_CONTACT_INFO: int = 4
MAX_ELEMENTS: int = 128
NUM_ATOM_NAME_CHARS: int = 4


@dataclass
class AtomAttentionEncoderStaticContext:
    """Static atom representations returned by ``compute_static_context``.

    ``B`` is batch size, ``N`` is padded atom count, ``N_token`` is padded token
    count, ``NW`` is the number of atom query windows, ``W`` is the query-window
    size, and ``H`` is the key-window size. ``apply_dynamic`` combines these
    cached values with the current coordinates.

    Attributes:
        q: Initial atom representations, shape ``(B, N, atom_s)``.
        c: Atom conditioning representations, shape ``(B, N, atom_s)``.
        p: Windowed atom-pair representations, shape
            ``(B, NW, W, H, atom_z)``.
        atom_pad_mask: Shape ``(B, N)``; ``True`` means the atom is not padding.
        atom_to_token_ids: Parent-token index for each atom, shape ``(B, N)``.
        token_pad_mask: Shape ``(B, N_token)``; ``True`` means the token is not
            padding.
    """

    q: torch.Tensor
    c: torch.Tensor
    p: torch.Tensor
    atom_pad_mask: torch.Tensor
    atom_to_token_ids: torch.Tensor
    token_pad_mask: torch.Tensor


class AtomAttentionEncoder(nn.Module):
    """Atom attention encoder."""

    def __init__(
        self,
        atom_s,
        atom_z,
        token_s,
        token_z,
        atoms_per_window_queries,
        atoms_per_window_keys,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        structure_prediction=True,
        activation_checkpointing=False,
        use_atom_encoder: bool = True,
        use_atom_to_token_aggregation: bool = True,
        use_dynamics_conditioning: bool = False,
        downscale_condition_contrib: bool = False,
        dt_embed_num_timescales: int = 32,
        # Vocabulary sizes — fixed by the data format, not plumbed from higher-level modules.
        max_elements: int = MAX_ELEMENTS,
        num_bins_atom: int = NUM_BINS_ATOM,
        num_atom_name_chars: int = NUM_ATOM_NAME_CHARS,
    ):
        """Initialize the atom attention encoder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single representation dimension.
        token_z : int
            The pair representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_encoder_depth : int, optional
            The number of transformer layers, by default 3.
        atom_encoder_heads : int, optional
            The number of transformer heads, by default 4.
        structure_prediction : bool, optional
            Whether it is used in the diffusion module, by default True.
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False.
        use_atom_encoder : bool, optional
            Whether to instantiate and run the per-window stack of
            ``ConditionedSlidingWindowTransformerLayer`` (windowed atom self-attention).
            When False, atom features ``q`` are returned without further
            self-attention. By default True.
        use_atom_to_token_aggregation : bool, optional
            Whether to instantiate and run the atom -> token aggregation
            (`atom_to_token_trans` + bmm). When False, the returned ``a`` is
            ``None``. By default True.
        use_dynamics_conditioning : bool, optional
            If True, expect ``coords_cond`` (conditioning coordinates, same shape as the
            noised coordinates) and ``is_dynamics_conditioned`` and add a per-atom signal to the
            atom-level queries ``q``, built from the conditioning coordinates and gated per-sample
            by the mask.
        downscale_condition_contrib : bool, optional
            Only meaningful when ``use_dynamics_conditioning`` is True. If True, multiply the
            per-atom ``coords_cond`` contribution to ``q`` by a learned scalar gate built as a
            mixture of pure exponentials of the lag time ``dt``.
        dt_embed_num_timescales : int, optional
            Number of exponential timescales in the lag gate, by default 32.
        """
        super().__init__()

        # Separate projection/embedding per feature group, summed into atom_s.
        self.proj_atom_continuous = LinearNoBias(4, atom_s)  # ref_pos(3) + ref_charge(1)
        self.embed_element = nn.Embedding(max_elements, atom_s)
        self.embed_atom_name_chars = nn.ModuleList(
            [nn.Embedding(num_bins_atom, atom_s) for _ in range(num_atom_name_chars)]
        )
        self.embed_atompair_ref_pos = LinearNoBias(3, atom_z)
        self.embed_atompair_ref_dist = LinearNoBias(1, atom_z)
        self.embed_atompair_mask = LinearNoBias(1, atom_z)
        self.atoms_per_window_queries = atoms_per_window_queries
        self.atoms_per_window_keys = atoms_per_window_keys

        self.structure_prediction = structure_prediction
        if structure_prediction:
            self.s_to_c_trans = nn.Sequential(nn.LayerNorm(token_s), LinearNoBias(token_s, atom_s))

            self.z_to_p_trans = nn.Sequential(nn.LayerNorm(token_z), LinearNoBias(token_z, atom_z))

            self.r_to_q_trans = LinearNoBias(3, atom_s)

        self.use_dynamics_conditioning = use_dynamics_conditioning and structure_prediction
        self.downscale_condition_contrib = (
            downscale_condition_contrib and self.use_dynamics_conditioning
        )
        if self.use_dynamics_conditioning:
            # Per-atom linear that folds the conditioning signal into the atom query
            self.cond_to_q_trans = LinearNoBias(3, atom_s)

            if self.downscale_condition_contrib:
                self.dt_gate_embed = MultiExponentialTimeEmbedding(dim=dt_embed_num_timescales)
                n_taus = self.dt_gate_embed.taus.shape[0]
                init_log_w = float(torch.log(torch.expm1(torch.tensor(1.0 / n_taus))))
                self.dt_gate_log_w = nn.Parameter(torch.full((n_taus,), init_log_w))

        self.c_to_p_trans_k = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )

        self.c_to_p_trans_q = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_s, atom_z),
        )

        self.p_mlp = nn.Sequential(
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
            nn.ReLU(),
            LinearNoBias(atom_z, atom_z),
        )

        self.use_atom_encoder = use_atom_encoder
        self.use_atom_to_token_aggregation = use_atom_to_token_aggregation

        if use_atom_encoder:
            self.atom_encoder_checkpointing = activation_checkpointing
            self.atom_encoder_layers = nn.ModuleList(
                [
                    ConditionedSlidingWindowTransformerLayer(
                        dim=atom_s,
                        dim_pairwise=atom_z,
                        dim_cond=atom_s,
                        heads=atom_encoder_heads,
                        transition_hidden=4 * atom_s,
                        window_queries=atoms_per_window_queries,
                        window_keys=atoms_per_window_keys,
                    )
                    for _ in range(atom_encoder_depth)
                ]
            )

        if use_atom_to_token_aggregation:
            self.atom_to_token_trans = nn.Sequential(
                LinearNoBias(atom_s, 2 * token_s if structure_prediction else token_s),
                nn.ReLU(),
            )

    def compute_static_context(
        self,
        atom_ref_pos: torch.Tensor,  # (B, N, 3)
        ref_charge: torch.Tensor,  # (B, N)
        ref_element: torch.Tensor,  # (B, N) int64 index
        ref_atom_name_chars: torch.Tensor,  # (B, N, 4) int64 indices
        atom_ref_space_uid: torch.Tensor,  # (B, N)
        atom_pad_mask: torch.Tensor,  # (B, N)
        atom_to_token_ids: torch.Tensor,  # (B, N)
        token_pad_mask: torch.Tensor,  # (B, T)
        s_trunk: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> AtomAttentionEncoderStaticContext:
        B, N, _ = atom_ref_pos.shape
        is_padded_atom = ~atom_pad_mask.bool()  # (B, N)

        # Embed atom features via separate embeddings summed into atom_s.
        continuous = torch.cat([atom_ref_pos, ref_charge.unsqueeze(-1)], dim=-1)  # (B, N, 4)
        c = self.proj_atom_continuous(continuous)  # (B, N, atom_s)
        c = c + self.embed_element(ref_element)  # (B, N, atom_s)
        for i, emb in enumerate(self.embed_atom_name_chars):
            c = c + emb(ref_atom_name_chars[:, :, i])  # (B, N, atom_s)
        # Zero out padded atoms
        c.masked_fill_(is_padded_atom.unsqueeze(-1), 0.0)

        # NOTE: we are already creating the windows to make it more efficient
        W, H = self.atoms_per_window_queries, self.atoms_per_window_keys
        B, N = c.shape[:2]
        assert N % W == 0, f"Number of atoms {N} is not divisible by window size {W}."
        NW = N // W

        atom_pair_disp, atom_pair_inv_dist, atom_pair_is_same_ref_space = (
            windowed_atom_pair_geometry(
                atom_ref_pos=atom_ref_pos,
                atom_mask=atom_pad_mask,
                atom_uid=atom_ref_space_uid,
                atoms_per_window_queries=W,
                atoms_per_window_keys=H,
                out_dtype=c.dtype,
            )
        )

        p = self.embed_atompair_ref_pos(atom_pair_disp)
        p = p + self.embed_atompair_ref_dist(atom_pair_inv_dist)
        p = p + self.embed_atompair_mask(atom_pair_is_same_ref_space)
        p.masked_fill_(~atom_pair_is_same_ref_space.bool(), 0.0)

        q = c

        if self.structure_prediction:
            c = c + broadcast_tokens_to_atoms(
                self.s_to_c_trans(s_trunk),
                atom_to_token_ids=atom_to_token_ids,
                is_padded_atom=is_padded_atom,
            )

            z_to_p = self.z_to_p_trans(z)
            z_to_p = broadcast_token_pair_to_atom_window(
                z_to_p,
                atom_to_token_ids=atom_to_token_ids,
                is_padded_atom=is_padded_atom,
                W=W,
                H=H,
            )
            p = p + z_to_p

        p = p + self.c_to_p_trans_q(c.view(B, NW, W, 1, c.shape[-1]))
        p = p + self.c_to_p_trans_k(window_atom_single_to_pair_keys(c, W, H))
        p = p + self.p_mlp(p)

        return AtomAttentionEncoderStaticContext(
            q=q,
            c=c,
            p=p,
            atom_pad_mask=atom_pad_mask,
            atom_to_token_ids=atom_to_token_ids,
            token_pad_mask=token_pad_mask,
        )

    def apply_dynamic(
        self,
        context: AtomAttentionEncoderStaticContext,
        r: torch.Tensor | None,
        multiplicity: int,
        coords_cond: torch.Tensor | None = None,
        is_dynamics_conditioned: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = context.q
        c = context.c
        p = context.p

        if self.structure_prediction:
            # only here the multiplicity kicks in because we use the different positions r
            assert r is not None, "r must not be None when structure_prediction is True."
            q = q.repeat_interleave(multiplicity, 0)
            r_to_q = self.r_to_q_trans(r)
            q = q + r_to_q

            if self.use_dynamics_conditioning:
                assert coords_cond is not None and is_dynamics_conditioned is not None
                cond_contrib = self.cond_to_q_trans(coords_cond)  # [B * multiplicity, N, atom_s]

                if self.downscale_condition_contrib:
                    assert dt is not None
                    decay = self.dt_gate_embed(dt.expand(-1, multiplicity).reshape(-1, 1))
                    gate = (decay * softplus(self.dt_gate_log_w)).sum(-1, keepdim=True)
                    cond_contrib = cond_contrib * gate.unsqueeze(-1)

                # is_dynamics_conditioned is bool (B, M); expand to per-sample (B*M, 1, 1)
                cond_mask = ~is_dynamics_conditioned.expand(-1, multiplicity).reshape(-1, 1, 1)
                cond_contrib.masked_fill_(cond_mask, 0.0)
                q = q + cond_contrib

        c = c.repeat_interleave(multiplicity, 0)

        if self.use_atom_encoder:
            do_checkpointing = self.atom_encoder_checkpointing and q.requires_grad
            for layer in self.atom_encoder_layers:
                if do_checkpointing:
                    q = activation_checkpoint(
                        layer,
                        q,
                        p,
                        c,
                        context.atom_pad_mask,
                        multiplicity,
                        use_reentrant=False,
                    )
                else:
                    q = layer(
                        q,
                        p,
                        c,
                        mask=context.atom_pad_mask,
                        multiplicity=multiplicity,
                    )

        if self.use_atom_to_token_aggregation:
            is_padded_atom = ~context.atom_pad_mask.bool()
            a = scatter_mean_atoms_to_tokens(
                self.atom_to_token_trans(q),
                context.atom_to_token_ids,
                is_padded_atom,
                num_tokens=context.token_pad_mask.shape[1],
                multiplicity=multiplicity,
            )
        else:
            a = None

        return a, q, c, p

    def forward(
        self,
        atom_ref_pos: torch.Tensor,  # (B, N, 3)
        ref_charge: torch.Tensor,  # (B, N)
        ref_element: torch.Tensor,  # (B, N) int64 index
        ref_atom_name_chars: torch.Tensor,  # (B, N, 4) int64 indices
        atom_ref_space_uid: torch.Tensor,  # (B, N)
        atom_pad_mask: torch.Tensor,  # (B, N)
        atom_to_token_ids: torch.Tensor,  # (B, N)
        token_pad_mask: torch.Tensor,  # (B, T)
        s_trunk: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        r: torch.Tensor | None = None,
        multiplicity: int = 1,
        coords_cond: torch.Tensor | None = None,  # (B * M, N, 3) dynamics conditioning
        is_dynamics_conditioned: torch.Tensor | None = None,  # (B, M) dynamics conditioning
        dt: torch.Tensor | None = None,  # (B, M) lag times
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed atom features with windowed atom-pair attention.

        Args:
            atom_ref_pos: Reference atom positions, ``(B, N, 3)``.
            ref_charge: Reference atom charges, ``(B, N)``.
            ref_element: Element type index, ``(B, N)`` int64.
            ref_atom_name_chars: Atom name character indices, ``(B, N, 4)`` int64.
            atom_ref_space_uid: Reference space UID for pair masking, ``(B, N)``.
            atom_pad_mask: Atom validity mask (1=valid, 0=pad), ``(B, N)``.
            atom_to_token_ids: Maps each atom to its parent token, ``(B, N)``.
            token_pad_mask: Token validity mask, ``(B, T)``.
            s_trunk: Trunk single representation, ``(B, T, token_s)``. Required when
                ``structure_prediction=True``.
            z: Trunk pair representation, ``(B, T, T, token_z)``. Required when
                ``structure_prediction=True``.
            r: Noisy atom coordinates, ``(B * multiplicity, N, 3)``. Required when
                ``structure_prediction=True``.
            multiplicity: Number of noise replicates per batch element.
            coords_cond: Conditioning coordinates for dynamics, ``(B * M, N, 3)``.
            is_dynamics_conditioned: Per-sample conditioning flag, ``(B, M)``.
            dt: Lag time between ``coords_cond`` and the target, ``(B, M)``.

        Returns:
            Tuple of ``(a, q, c, p)`` where ``a`` is the token-aggregated atom
            features (or None), ``q`` is the atom single representation, ``c`` is
            the atom conditioning, and ``p`` is the atom pair conditioning.
        """
        context = self.compute_static_context(
            atom_ref_pos=atom_ref_pos,
            ref_charge=ref_charge,
            ref_element=ref_element,
            ref_atom_name_chars=ref_atom_name_chars,
            atom_ref_space_uid=atom_ref_space_uid,
            atom_pad_mask=atom_pad_mask,
            atom_to_token_ids=atom_to_token_ids,
            token_pad_mask=token_pad_mask,
            s_trunk=s_trunk,
            z=z,
        )
        return self.apply_dynamic(
            context,
            r=r,
            multiplicity=multiplicity,
            coords_cond=coords_cond,
            is_dynamics_conditioned=is_dynamics_conditioned,
            dt=dt,
        )


class FourierEmbedding(nn.Module):
    """Fourier embedding layer."""

    def __init__(self, dim: int):
        """Initialize the Fourier Embeddings.

        Parameters
        ----------
        dim : int
            The dimension of the embeddings.

        """

        super().__init__()
        self.proj = nn.Linear(1, dim)
        torch.nn.init.normal_(self.proj.weight, mean=0, std=1)
        torch.nn.init.normal_(self.proj.bias, mean=0, std=1)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times: torch.Tensor,
    ) -> torch.Tensor:
        times = rearrange(times, "b -> b 1")
        rand_proj = self.proj(times)
        return torch.cos(2 * pi * rand_proj)
