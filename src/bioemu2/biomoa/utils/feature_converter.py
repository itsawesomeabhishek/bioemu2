"""Converters from ``BatchAllAtomStructure`` (dense batch) to flat-dict feature dicts
consumed by the BioMoa score model and its sub-modules."""

import torch

from bioemu2.conditioning.contact import (
    make_unspecified_contact_conditioning,
    make_unspecified_contact_threshold,
)
from bioemu2.structure import BatchAllAtomStructure
from bioemu2.structure.batch import REQUIRES_MSA_PADDING

# Pad policy for index-valued fields. Fields consumed via torch.gather or as
# scatter targets should use -1 so that unmasked pad leaks fail loudly
# (scatter out-of-bounds) rather than silently aliasing real slot 0.
INDEX_PAD_VALUE: int = -1


def convert_densebatch_to_dictionary(
    dense_batch: BatchAllAtomStructure,
    *,
    build_pair_features: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert a dense batch to the flat feature dict consumed by the BioMoa score model.

    Starts from :meth:`~bioemu2.structure.TensorStore.to_dict` (zero-copy)
    and overrides the few keys that require transformation (bool casts, derived
    token-level features, token bonds, contact-conditioning fallback).

    Atom-dimension tensors are expected to be already padded to a window-aligned
    size by :func:`~bioemu2.structure.batch.batch_collate`.

    Args:
        dense_batch: A dense batch of structures.
        build_pair_features: If False, skip computing ``token_bonds`` and the
            no-contact ``contact_conditioning``/``contact_threshold``

    Returns:
        Feature dictionary keyed by the same names as ``BatchAllAtomStructure``
        fields wherever possible.
    """
    max_tokens = dense_batch.num_tokens.max().item()
    device = dense_batch.coords.device

    # Start from the batch's own dict — no copies, just tensor references.
    # Filter to tensors with a batch dimension (excludes scalars like batch_size).
    feats: dict[str, torch.Tensor] = {
        k: v
        for k, v in dense_batch.to_dict().items()
        if isinstance(v, torch.Tensor) and v.dim() >= 2
    }

    # The model expects strict bool dtype.
    feats["atom_pad_mask"] = dense_batch.atom_pad_mask.bool()
    feats["atom_mask"] = dense_batch.atom_mask.bool()
    feats["token_pad_mask"] = dense_batch.token_pad_mask.bool()

    # Contact conditioning fallback.
    if build_pair_features and "contact_conditioning" not in feats:
        max_tokens_int = int(max_tokens)
        feats["contact_conditioning"] = make_unspecified_contact_conditioning(
            max_tokens_int, device=device
        ).expand(dense_batch.batch_size, -1, -1, -1)
        feats["contact_threshold"] = make_unspecified_contact_threshold(
            max_tokens_int, device=device
        ).expand(dense_batch.batch_size, -1, -1)

    # The pocket feature is always zero in the current pipeline.
    feats["pocket_feature"] = torch.zeros_like(dense_batch.token_pad_mask).long()  # (B, T)

    # Reference-space uid (an index): padded atoms carry the -1 sentinel. The atom-attention
    # encoder re-masks these to its own pad uid, so the pad value is only a debugging aid here.
    feats["ref_space_uid"] = dense_batch.residue_idx

    # Derived token-level features.
    batch_aux_idx = torch.arange(dense_batch.batch_size, device=device)[:, None]

    # Residue type by gathering from atom-level via token center atom.
    feats["res_type"] = dense_batch.residue_type[batch_aux_idx, dense_batch.token_center_atom_index]
    feats["res_type"][~dense_batch.token_pad_mask] = 0

    # Residue index (token-level) by gathering from atom-level residue_idx.
    feats["residue_index"] = dense_batch.residue_idx[
        batch_aux_idx, dense_batch.token_center_atom_index
    ]
    feats["residue_index"][~dense_batch.token_pad_mask] = INDEX_PAD_VALUE

    if build_pair_features:
        # Token bonds.
        token_bonds = torch.zeros(
            dense_batch.batch_size,
            max_tokens,
            max_tokens,
            1,
            dtype=dense_batch.coords.dtype,
            device=device,
        )
        token_bond_index = dense_batch.token_idx_at_atom[
            batch_aux_idx[..., None], dense_batch.bond_index
        ]
        token_bonds[batch_aux_idx, token_bond_index[:, 0, :], token_bond_index[:, 1, :]] = 1
        # Symmetrize (some templates only have unique bond pairs [i,j] without [j,i]).
        token_bonds = token_bonds + token_bonds.transpose(1, 2)
        feats["token_bonds"] = torch.clamp(token_bonds, 0, 1)

    # MSA features.
    if "msa" in dense_batch:
        for msa_feature in [
            "msa",
            "msa_paired",
            "deletion_value",
            "has_deletion",
            "deletion_mean",
            "profile",
            "msa_mask",
        ]:
            # The batch stores token-major MSA features; the trunk expects MSA depth first.
            if msa_feature in REQUIRES_MSA_PADDING:
                feats[msa_feature] = dense_batch[msa_feature].transpose(1, 2)

    feats = {key: value.to(device) for key, value in feats.items()}
    return feats
