"""Collation of all-atom structures into padded batches."""

import functools
import math
from collections.abc import Iterable

import torch

from bioemu2.structure.all_atom_structure import (
    ATOM_LEVEL_PROPERTIES,
    AllAtomStructure,
    BatchAllAtomStructure,
)

# List of all MSA related quantities that require padding along the sequence depth dimension.
REQUIRES_MSA_PADDING = [
    "msa",
    "msa_paired",
    "deletion_value",
    "has_deletion",
    "msa_mask",
]


def pad_msa_features(
    tensor_list: Iterable[torch.Tensor],
    max_pad: int = 4096,
    pad_dim: int = 1,
    pad_value: float = 0.0,
) -> list[torch.Tensor]:
    """
    Pad tensors containing MSA derived features along the sequence depth dimension.

    Args:
        tensor_list: List of input tensors (shape is expected to be different along `pad_dim`).
        max_pad: Tensors will be padded to this value along the `pad_dim` dimension.
        pad_dim: Dimension to which padding should be applied.
        pad_value: Value used for padding.

    Returns:
        List of input tensors with `pad_dim` padded to `max_pad` using `pad_value`.
    """

    def _get_padding(x: torch.Tensor) -> list[int]:
        """
        Auxiliary function used for creating padding instructions for `torch.nn.functional.pad`.
        Notation is a little tricky, since the latter expects an input along the lines of:

        (padding_left_dim-1, padding_right_dim-1, padding_left_dim-2, padding_right_dim-2, ...)

        So the first two values are for padding the last dimension etc. We construct a list with
        twice as many entries as the tensor dimension and set the left padding value of the padding
        dim (2 * `pad_dim` in reverse counting) to the difference in the padding dim of the current
        tensor and `max_pad`.
        """
        pad_size = max(max_pad - x.shape[pad_dim], 0)
        return [0 if i != 2 * pad_dim else pad_size for i in range(2 * x.ndim - 1, -1, -1)]

    tensor_list_padded = [
        torch.nn.functional.pad(x, pad=_get_padding(x), mode="constant", value=pad_value)
        for x in tensor_list
    ]

    return tensor_list_padded


def batch_collate(batch: list[AllAtomStructure], window_size: int = 32) -> BatchAllAtomStructure:
    """
    Collate a list of `AllAtomStructure` objects into a single `BatchAllAtomStructure` object.

    This function takes a batch of `AllAtomStructure` instances and combines them into a batched structure,
    padding variable-length attributes (such as atoms, residues, and bonds) to the maximum length in the batch.
    The resulting object has leading batch dimensions for all tensor fields, and appropriate masks are generated
    for atoms, residues, and bonds to indicate valid entries.

    Atom-dimension tensors are right-padded to the next multiple of ``window_size``

    Args:
        batch: A list of ``AllAtomStructure`` objects to be collated.
        window_size: Atom-dimension padding is rounded up to the nearest  multiple of this value.

    Returns:
        BatchAllAtomStructure: A single batched structure with padded and collated fields, including
        atom, residue, and bond masks, as well as all other relevant attributes.

    Notes:
        - The function assumes all objects in the batch have the same set of keys/attributes.
        - Non-tensor fields such as `sequence` and `system_id` are collected as lists.
        - Tensor fields are padded along the appropriate dimensions to enable batching.
        - Masks are generated to distinguish valid entries from padding.
        - Embeddings may exist, or are added later

    """
    # Samples can optionally have a conditioning frame (coords_cond) and dt
    # NOTE this is used for time-lagged training (ITO)
    conditioning_flags = [
        (
            "coords_cond" in item
            and "dt" in item
            and item.coords_cond is not None
            and item.dt is not None
        )
        for item in batch
    ]
    batch = [
        item
        if flag
        else item.replace(
            coords_cond=torch.zeros_like(item.coords),
            # coords are either [N_atoms, 3] or [N_atoms, M, 3]
            dt=item.coords.new_zeros(item.coords.shape[1] if item.coords.ndim == 3 else 1),
        )
        for item, flag in zip(batch, conditioning_flags)
    ]

    keys_to_collate = list(batch[0].keys())

    if "residue_labels" in keys_to_collate:
        keys_to_collate.remove("residue_labels")

    if "interfaces" in keys_to_collate:
        keys_to_collate.remove("interfaces")

    # Collect list information into a dictionary dict[str, list[str | Tensor | int]]
    data = {key: [getattr(item, key) for item in batch] for key in keys_to_collate}

    if "edge_is_bonded" in data:
        del data["edge_is_bonded"]  # [BS, edge, 1]: bool, information is contained in bond_index
    if "edge_index" in data:
        del data["edge_index"]

    for key in ["num_nodes", "num_residues", "num_tokens"]:
        # Convert list[int] to tensor
        data[key] = torch.tensor(data[key], dtype=torch.int64).unsqueeze(-1)

    num_tokens_max = max(data["num_tokens"])  # maximum sequence length in the batch
    if "contact_conditioning" in data:
        assert "contact_threshold" in data, "Contact conditioning requires contact thresholds."
        for sample_idx, (S, cc) in enumerate(zip(data["num_tokens"], data["contact_conditioning"])):
            F = cc.shape[-1]
            padded_cc = torch.zeros((num_tokens_max, num_tokens_max, F), dtype=cc.dtype)
            padded_cc[:S, :S, :] = cc
            data["contact_conditioning"][sample_idx] = padded_cc

    if "contact_threshold" in data:
        assert "contact_conditioning" in data, "Contact threshold requires contact conditioning."
        for sample_idx, (S, ct) in enumerate(zip(data["num_tokens"], data["contact_threshold"])):
            padded_ct = torch.zeros((num_tokens_max, num_tokens_max), dtype=ct.dtype)
            padded_ct[:S, :S] = ct
            data["contact_threshold"][sample_idx] = padded_ct

    # Pad sequence depth dimension of MSA features and related quantities before token padding.
    for key in REQUIRES_MSA_PADDING:
        if key in data and data[key] is not None:
            max_pad = max([x.shape[1] for x in data[key]])
            data[key] = pad_msa_features(data[key], max_pad=max_pad, pad_dim=1, pad_value=0.0)

    # Define padding function for variable length entries, right padded with 0.0, batch dim prepended
    pad_fn = functools.partial(
        torch.nn.utils.rnn.pad_sequence,
        batch_first=True,
        padding_value=0.0,
        padding_side="right",
    )

    # Atom-dim padding: pads to at least max_atoms_aligned (next multiple of window_size).
    max_atoms = int(max(data["num_nodes"]).item())
    max_atoms_aligned = (
        math.ceil(max_atoms / window_size) * window_size if window_size > 1 else max_atoms
    )

    def atom_pad_fn(sequences: list[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
        """Pad sequences to ``max_atoms_aligned`` by extending the first item, then calling pad_sequence."""
        t = sequences[0]
        extra = max_atoms_aligned - t.shape[0]
        if extra > 0:
            sequences[0] = torch.cat([t, t.new_full((extra,) + t.shape[1:], pad_value)])
        return torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=pad_value, padding_side="right"
        )

    # Create atom:[BS, max_atoms_aligned], token:[BS, max(num_tokens)] and bond:[BS, max(num_bonds)] masks
    atom_pad_mask = atom_pad_fn(
        [torch.ones(size=(num_atoms,)).bool() for num_atoms in data["num_nodes"]]
    )
    token_pad_mask = pad_fn(
        sequences=[torch.ones(size=(num_tokens,)).bool() for num_tokens in data["num_tokens"]]
    )
    # bond_indices:[2, N_bonds] are transposed to [N_bonds, 2] for padding, such that variable lengths are in front
    bond_pad_mask = (
        pad_fn(
            sequences=[torch.ones(size=bond_index_.shape).T for bond_index_ in data["bond_index"]]
        )
        .bool()
        .transpose(2, 1)
    )

    # Collect atoms in dictionary with batched entries. Store padding masks.
    batched_data: dict[str, list[str] | torch.Tensor | list[None]] = {
        "atom_pad_mask": atom_pad_mask,
        "bond_pad_mask": bond_pad_mask,
        "token_pad_mask": token_pad_mask,
    }
    batched_data["bond_index"] = pad_fn([data_.T for data_ in data["bond_index"]]).transpose(2, 1)
    batched_data["template_bond_index"] = pad_fn(
        [data_.T for data_ in data["bond_index"]]
    ).transpose(2, 1)
    batched_data["sequences"] = data["sequence"]

    batched_data["system_id"] = data["system_id"] if "system_id" in data else [None] * len(batch)
    # broadcast is_dynamics_conditioned to match the shape of dt
    data["is_dynamics_conditioned"] = [
        torch.full_like(dt.reshape(-1), flag, dtype=torch.bool)
        for dt, flag in zip(data["dt"], conditioning_flags)
    ]

    for key in data:
        # data['sequence]: list[str] not batchable
        # data['system_id]: list[None] not batchable
        # data['bond_index]: list[torch.Tensor] with shape [2, N_bonds], already processed above
        # data['pair_embeds]: list[torch.Tensor] with shape [S, S, F], already processed above
        if key not in [
            "sequence",
            "system_id",
            "bond_index",
        ]:
            # Index fields use the -1 sentinel pad; every other field keeps the 0 pad.
            if key in ["token_idx_at_atom", "residue_idx"]:
                batched_data[key] = atom_pad_fn(data[key], pad_value=-1)
            elif key in ATOM_LEVEL_PROPERTIES:
                batched_data[key] = atom_pad_fn(data[key])
            else:
                batched_data[key] = pad_fn(data[key])

    batched_data["batch_size"] = torch.scalar_tensor(len(batch), dtype=torch.int64)  # [1]

    return BatchAllAtomStructure(**batched_data)
