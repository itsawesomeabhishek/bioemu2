# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Input embedder and distogram head for the BioMoa trunk."""

import torch
import torch.nn as nn

from bioemu2.biomoa.modules.encoders import (
    NUM_MSA_TOKENS,
    NUM_TOKEN_TYPES,
    AtomAttentionEncoder,
)


class InputEmbedder(nn.Module):
    """Input embedder.

    Concatenates atom-level features (aggregated to token level by the atom
    attention encoder) with per-token features (residue type one-hot, MSA
    profile, deletion mean, pocket feature) into a single representation.

    The output dimension is ``token_s + num_token_types + num_msa_tokens + 1
    + num_pocket_classes`` (default: 384 + 33 + 33 + 1 + 4 = 455). Downstream
    consumers (``s_init``, ``z_init``, ``SingleConditioning``) each apply their
    own learned projection to the full vector, giving each consumer independent
    access to both atom-level and per-token features.
    """

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
        num_token_types: int = NUM_TOKEN_TYPES,
        num_msa_tokens: int = NUM_MSA_TOKENS,
        num_pocket_classes: int = 4,
    ) -> None:
        """Initialize the input embedder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single token representation dimension.
        token_z : int
            The pair token representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_encoder_depth : int
            The atom encoder depth.
        atom_encoder_heads : int
            The atom encoder heads.
        no_atom_encoder : bool, optional
            Whether to skip the atom encoder, by default False.
        num_token_types : int
            Number of residue/token types for one-hot encoding.
        num_msa_tokens : int
            Dimension of MSA profile features (continuous).
        num_pocket_classes : int
            Number of pocket feature classes for one-hot encoding.
        """
        super().__init__()
        self.token_s = token_s
        self.num_token_types = num_token_types
        self.num_msa_tokens = num_msa_tokens
        self.num_pocket_classes = num_pocket_classes
        self.no_atom_encoder = no_atom_encoder

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                token_z=token_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
                structure_prediction=False,
            )

    @property
    def output_dim(self) -> int:
        """Total output dimension: token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes."""
        return (
            self.token_s + self.num_token_types + self.num_msa_tokens + 1 + self.num_pocket_classes
        )

    def forward(
        self,
        res_type: torch.Tensor,
        profile: torch.Tensor,
        deletion_mean: torch.Tensor,
        pocket_feature: torch.Tensor,
        ref_pos: torch.Tensor,
        ref_charge: torch.Tensor,
        ref_element: torch.Tensor,
        ref_atom_name_chars: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        atom_to_token_ids: torch.Tensor,
        token_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Embed input features into a concatenated representation.

        Concatenates the atom-level token aggregation with per-token features
        (residue type one-hot, MSA profile, deletion mean, pocket one-hot) so
        that each downstream consumer can apply its own learned projection.

        Args:
            res_type: Residue / token type index, ``(B, T)`` int64.
            profile: MSA profile features (continuous), ``(B, T, NUM_MSA_TOKENS)``.
            deletion_mean: Mean deletion count per token, ``(B, T)``.
            pocket_feature: Pocket feature index, ``(B, T)`` int64.
            ref_pos: Reference atom positions, ``(B, N, 3)``.
            ref_charge: Reference atom charges, ``(B, N)``.
            ref_element: Element type index, ``(B, N)`` int64.
            ref_atom_name_chars: Atom name character indices, ``(B, N, 4)`` int64.
            ref_space_uid: Reference space UID for pair masking, ``(B, N)``.
            atom_pad_mask: Atom validity mask, ``(B, N)``.
            atom_to_token_ids: Maps each atom to its parent token, ``(B, N)``.
            token_pad_mask: Token validity mask, ``(B, T)``.

        Returns:
            ``(B, T, token_s + 71)`` concatenation of atom encoding and
            per-token features.
        """
        # Compute atom-level embedding aggregated to tokens  # a: (B, T, token_s)
        if self.no_atom_encoder:
            a = torch.zeros(
                (res_type.shape[0], res_type.shape[1], self.token_s),
                device=res_type.device,
                dtype=profile.dtype,
            )
        else:
            a, _, _, _ = self.atom_attention_encoder(
                atom_ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                atom_ref_space_uid=ref_space_uid,
                atom_pad_mask=atom_pad_mask,
                atom_to_token_ids=atom_to_token_ids,
                token_pad_mask=token_pad_mask,
            )

        # Per-token features as raw vectors (one-hot / continuous).
        profile = profile.to(a.dtype)
        deletion_mean = deletion_mean.to(a.dtype)

        res_onehot = torch.nn.functional.one_hot(res_type, num_classes=self.num_token_types).to(
            a.dtype
        )  # (B, T, num_token_types)
        pocket_onehot = torch.nn.functional.one_hot(
            pocket_feature, num_classes=self.num_pocket_classes
        ).to(a.dtype)  # (B, T, num_pocket_classes)

        return torch.cat(
            [a, res_onehot, profile, deletion_mean.unsqueeze(-1), pocket_onehot], dim=-1
        )  # (B, T, token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes)


class DistogramModule(nn.Module):
    """Distogram Module."""

    def __init__(self, token_z: int, num_bins: int) -> None:
        """Initialize the distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.
        num_bins : int
            The number of bins.

        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : torch.Tensor
            The pairwise embeddings

        Returns
        -------
        torch.Tensor
            The predicted distogram.

        """
        z = z + z.transpose(1, 2)
        return self.distogram(z)
