# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""MSA data structures and array assembly for the sequence embedder."""

from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt
import torch
from numba import types

from bioemu2.data_interface.const import Token
from bioemu2.structure import AllAtomStructure

# MSA types
MSAResidue = [
    ("res_type", np.dtype("i1")),
]

MSADeletion = [
    ("res_idx", np.dtype("i2")),
    ("deletion", np.dtype("i2")),
]

MSASequence = [
    ("seq_idx", np.dtype("i2")),
    ("taxonomy", np.dtype("i4")),
    ("res_start", np.dtype("i4")),
    ("res_end", np.dtype("i4")),
    ("del_start", np.dtype("i4")),
    ("del_end", np.dtype("i4")),
]


@dataclass(frozen=True)
class EncodedMSA:
    """Encoded sequence, deletion, and residue arrays for MSA featurization."""

    sequences: np.ndarray  # shape=(n_sequences_in_msa, 6) dtype=[('seq_idx', '<i2'), ('taxonomy', '<i4'), ('res_start', '<i4'), ('res_end', '<i4'), ('del_start', '<i4'), ('del_end', '<i4')])
    deletions: np.ndarray  # shape=(n_deletions, 2) dtype=[('res_idx', '<i2'), ('deletion', '<i2')])
    residues: (
        np.ndarray
    )  # shape=(n_sequences_in_msa * len(ofeachsequence), 1) dtype=[('res_type', 'i1')])


def make_dummy_msa(residue_types: np.ndarray) -> EncodedMSA:
    """Create a dummy MSA for a chain. This is used for non protein
    entities.

    Parameters
    ----------
    residues_types : np.ndarray
        A numpy array containing residue type encodings (e.g. [2, 1, 1, 3, ...]).

    Returns
    -------
    EncodedMSA
        The dummy MSA.

    """
    deletions = []  # type: ignore
    sequences = [(0, -1, 0, len(residue_types), 0, 0)]
    return EncodedMSA(
        residues=np.array(residue_types, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )


def _local_residue_idx_at_tokens(structure: AllAtomStructure) -> np.ndarray:
    """
    Return local residue indices at the token level.

    Uses ``token_idx_at_atom`` (atom→token mapping) instead of
    ``downsample_to_tokens`` because ``token_center_atom_index`` can be shorter
    than ``num_tokens`` when some polymer residues lack their expected center
    atom (e.g. capping groups like ACE/NME without CA).
    """
    # Scatter atom-level residue_idx to tokens via the atom→token mapping.
    # Multiple atoms per token all carry the same residue_idx, so last-write is fine.
    global_res_idx_at_tokens = torch.zeros(structure.num_tokens, dtype=torch.long)
    global_res_idx_at_tokens.scatter_(0, structure.token_idx_at_atom, structure.residue_idx)

    # Convert global residue indices to local (per-chain, 0-based) indices.
    asym_id = structure.asym_id
    local_res_idx = torch.zeros_like(global_res_idx_at_tokens)
    for chain_id in torch.unique(asym_id):
        mask = asym_id == chain_id
        chain_global = global_res_idx_at_tokens[mask]
        local_res_idx[mask] = chain_global - chain_global[0]
    return local_res_idx.numpy()


def _prepare_msa_arrays(
    structure: AllAtomStructure,
    pairing: list[dict[int, int]],
    is_paired: list[dict[int, int]],
    deletions: dict[tuple[int, int, int], int],
    msa: dict[int, EncodedMSA],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Reshape data to play nicely with numba jit."""
    token_asym_ids_arr = structure.asym_id.numpy()
    token_res_idxs_arr = _local_residue_idx_at_tokens(structure)
    assert len(token_asym_ids_arr) == len(token_res_idxs_arr) == structure.num_tokens, (
        "Token arrays don't match num_tokens"
    )
    chain_ids = sorted(msa.keys())

    # chain_ids are not necessarily contiguous (e.g. they might be 0, 24, 25).
    # This allows us to look up a chain_id by its index in the chain_ids list.
    chain_id_to_idx = {chain_id: i for i, chain_id in enumerate(chain_ids)}
    token_asym_ids_idx_arr = np.array(
        [chain_id_to_idx[asym_id] for asym_id in token_asym_ids_arr], dtype=np.int64
    )

    pairing_arr = np.zeros((len(pairing), len(chain_ids)), dtype=np.int64)
    is_paired_arr = np.zeros((len(is_paired), len(chain_ids)), dtype=np.int64)

    for i, row_pairing in enumerate(pairing):
        for chain_id in chain_ids:
            pairing_arr[i, chain_id_to_idx[chain_id]] = row_pairing[chain_id]

    for i, row_is_paired in enumerate(is_paired):
        for chain_id in chain_ids:
            is_paired_arr[i, chain_id_to_idx[chain_id]] = row_is_paired[chain_id]

    max_seq_len = max(len(msa[chain_id].sequences) for chain_id in chain_ids)

    # we want res_start from sequences
    msa_sequences = np.full((len(chain_ids), max_seq_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        for i, seq in enumerate(msa[chain_id].sequences):
            msa_sequences[chain_id_to_idx[chain_id], i] = seq["res_start"]

    max_residues_len = max(len(msa[chain_id].residues) for chain_id in chain_ids)
    msa_residues = np.full((len(chain_ids), max_residues_len), -1, dtype=np.int64)
    for chain_id in chain_ids:
        residues = msa[chain_id].residues.astype(np.int64)
        idxs = np.arange(len(residues))
        chain_idx = chain_id_to_idx[chain_id]
        msa_residues[chain_idx, idxs] = residues

    deletions_dict = numba.typed.Dict.empty(
        key_type=numba.types.Tuple([numba.types.int64, numba.types.int64, numba.types.int64]),
        value_type=numba.types.int64,
    )
    deletions_dict.update(deletions)

    return _prepare_msa_arrays_inner(
        token_asym_ids_arr,
        token_res_idxs_arr,
        token_asym_ids_idx_arr,
        pairing_arr,
        is_paired_arr,
        deletions_dict,
        msa_sequences,
        msa_residues,
        Token.GAP.value,
    )


DeletionDict = types.DictType(types.UniTuple(types.int64, 3), types.int64)


@numba.njit(
    [
        types.Tuple(
            (
                types.int64[:, ::1],  # msa_data
                types.int64[:, ::1],  # del_data
                types.int64[:, ::1],  # paired_data
            )
        )(
            types.int64[::1],  # token_asym_ids
            types.int64[::1],  # token_res_idxs
            types.int64[::1],  # token_asym_ids_idx
            types.int64[:, ::1],  # pairing
            types.int64[:, ::1],  # is_paired
            DeletionDict,  # deletions
            types.int64[:, ::1],  # msa_sequences
            types.int64[:, ::1],  # msa_residues
            types.int64,  # gap_token
        )
    ],
    cache=True,
)
def _prepare_msa_arrays_inner(
    token_asym_ids: npt.NDArray[np.int64],
    token_res_idxs: npt.NDArray[np.int64],
    token_asym_ids_idx: npt.NDArray[np.int64],
    pairing: npt.NDArray[np.int64],
    is_paired: npt.NDArray[np.int64],
    deletions: dict[tuple[int, int, int], int],
    msa_sequences: npt.NDArray[np.int64],
    msa_residues: npt.NDArray[np.int64],
    gap_token: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    n_tokens = len(token_asym_ids)
    n_pairs = len(pairing)
    msa_data = np.full((n_tokens, n_pairs), gap_token, dtype=np.int64)
    paired_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)
    del_data = np.zeros((n_tokens, n_pairs), dtype=np.int64)

    # Add all the token MSA data
    for token_idx in range(n_tokens):
        chain_id_idx = token_asym_ids_idx[token_idx]
        chain_id = token_asym_ids[token_idx]
        res_idx = token_res_idxs[token_idx]

        for pair_idx in range(n_pairs):
            seq_idx = pairing[pair_idx, chain_id_idx]
            paired_data[token_idx, pair_idx] = is_paired[pair_idx, chain_id_idx]

            # Add residue type
            if seq_idx != -1:
                res_start = msa_sequences[chain_id_idx, seq_idx]
                res_type = msa_residues[chain_id_idx, res_start + res_idx]
                k = (chain_id, seq_idx, res_idx)
                if k in deletions:
                    del_data[token_idx, pair_idx] = deletions[k]
                msa_data[token_idx, pair_idx] = res_type

    return msa_data, del_data, paired_data
