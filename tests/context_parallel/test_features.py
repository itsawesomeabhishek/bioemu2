from functools import partial
from unittest.mock import Mock, patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.features import (
    build_shard_msa_feats,
    build_shard_pair_feats,
    pad_feats_for_cp,
)
from bioemu2.context_parallel.utils.dtensor import PL_PAIR, msa_placements
from bioemu2.context_parallel.utils.padding import (
    make_cp_pad_sizes,
    pad_dim,
    pad_pair_along_token,
)
from bioemu2.data_interface import Contact, ContactToken, System
from bioemu2.msa.alignment import MSA
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import REQUIRES_MSA_PADDING, batch_collate
from bioemu2.structure.preparation import prepare_structure
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    spawn_cp_test,
    spawn_cpu_cp_test,
)


@pytest.mark.parametrize("already_aligned", [False, True])
def test_padding_preserves_real_batch_features(already_aligned: bool) -> None:
    structure = AllAtomStructure.from_system(System.from_description(proteins=["YDPEGAK"]))
    structure.template_coords = torch.arange(structure.num_nodes * 3).reshape(-1, 3).float()
    structure.coords_cond = structure.template_coords + 0.5
    structure.dt = torch.tensor([0.5])
    mesh = Mock(spec=DeviceMesh)
    mesh.ndim = 3
    mesh.size.side_effect = (1, 2, 3).__getitem__
    sizes = make_cp_pad_sizes(
        n_token=7, n_atom=structure.num_nodes, n_msa=0, atoms_per_window=32, mesh=mesh
    )
    batch = batch_collate([structure], window_size=sizes.n_atom_final if already_aligned else 32)
    feats = convert_densebatch_to_dictionary(batch)
    atom_extent = batch.coords.shape[1]
    token_ids_before = batch.token_idx_at_atom.clone()
    padded = pad_feats_for_cp(batch, sizes)

    assert padded["residue_index"].shape == (1, 12)
    for key in ("template_coords", "coords_cond"):
        assert padded[key].shape == (1, sizes.n_atom_final, 3)
        torch.testing.assert_close(padded[key][:, :atom_extent], feats[key])
        assert torch.count_nonzero(padded[key][:, atom_extent:]) == 0
    torch.testing.assert_close(padded["dt"], feats["dt"])
    torch.testing.assert_close(padded["is_dynamics_conditioned"], feats["is_dynamics_conditioned"])
    assert padded["token_idx_at_atom"].shape == (1, sizes.n_atom_final)
    torch.testing.assert_close(
        padded["token_idx_at_atom"][:, :atom_extent], token_ids_before.clamp_min(0)
    )
    assert torch.count_nonzero(padded["token_idx_at_atom"][:, atom_extent:]) == 0
    torch.testing.assert_close(batch.token_idx_at_atom, token_ids_before)


def test_padding_distinguishes_equal_atom_and_token_extents() -> None:
    mesh = Mock(spec=DeviceMesh)
    mesh.ndim = 3
    mesh.size.side_effect = (1, 2, 3).__getitem__
    sizes = make_cp_pad_sizes(n_token=7, n_atom=7, n_msa=0, atoms_per_window=32, mesh=mesh)
    indices = torch.arange(7).unsqueeze(0)
    feats = {
        "template_coords": torch.arange(21).reshape(1, 7, 3).float(),
        "token_idx_at_atom": indices,
        "charge": indices.float() - 3,
        "atom_mask": indices % 3 != 0,
        "atom_pad_mask": indices < 6,
        "ref_space_uid": indices // 2,
        "asym_id": indices // 3,
        "profile": torch.arange(140).reshape(1, 7, 20).float(),
        "token_pad_mask": indices < 5,
        "method_feature": indices % 3,
    }
    with patch(
        "bioemu2.context_parallel.features.convert_densebatch_to_dictionary",
        return_value=dict(feats),
    ):
        padded = pad_feats_for_cp(Mock(spec=BatchAllAtomStructure), sizes)
    assert padded["template_coords"].shape == (1, 192, 3)
    for key in ("token_idx_at_atom", "charge", "atom_mask", "atom_pad_mask", "ref_space_uid"):
        assert padded[key].shape == (1, 192)
    for key in ("asym_id", "profile", "token_pad_mask", "method_feature"):
        assert padded[key].shape[:2] == (1, 12)
    for key, value in feats.items():
        original_region = tuple(slice(0, extent) for extent in value.shape)
        torch.testing.assert_close(padded[key][original_region], value)
        assert padded[key].dtype == value.dtype
        assert torch.count_nonzero(padded[key]) == torch.count_nonzero(value)


def _local_pair_feats_match_full_then_pad(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    with_contacts: bool,
) -> None:
    sequence = "YDPEG"
    system = System.from_description(proteins=[sequence])
    chain_id = system.chain_ids[0]
    contacts = [
        Contact(
            token1=ContactToken(asym_id=chain_id, res_id=1),
            token2=ContactToken(asym_id=chain_id, res_id=3),
            max_distance=5.0,
        )
    ]
    structure = prepare_structure(
        system,
        msa=MSA({chain_id: f">query\n{sequence}\n"}),
        contacts=contacts if with_contacts else [],
        mode="structure",
    )
    sizes = make_cp_pad_sizes(
        n_token=structure.num_tokens,
        n_atom=structure.num_nodes,
        n_msa=0,
        atoms_per_window=32,
        mesh=mesh,
    )
    assert sizes.n_token_pad > 0
    assert sizes.n_atom_pad > 0
    batch = batch_collate([structure], window_size=sizes.n_atom_final)
    # Directed token pairs (0, 2) and (1, 0): symmetrization must fill in the transposes.
    first_atom = (batch.token_idx_at_atom[:, None, :] == torch.arange(3)[None, :, None]).int()
    first_atom = first_atom.argmax(dim=-1)
    batch.bond_index = torch.stack([first_atom[:, [0, 1]], first_atom[:, [2, 0]]], dim=1)

    full = convert_densebatch_to_dictionary(batch)
    keys = ("token_bonds", "contact_conditioning", "contact_threshold")
    expected = pad_pair_along_token([full[key] for key in keys], sizes.n_token_pad)
    local = build_shard_pair_feats(batch, sizes, mesh, device=device)

    assert torch.count_nonzero(expected[0]) == 4
    for key, reference in zip(keys, expected, strict=True):
        feat = local[key]
        assert isinstance(feat, DTensor)
        assert feat.dtype == reference.dtype
        assert_output_placement(feat, expected_placements=PL_PAIR, label=key)
        gather_n_assert_parity_at_rank0(reference, feat, label=key, atol=0)


@pytest.mark.parametrize("with_contacts", [False, True])
def test_local_pair_feats_match_full_then_pad(with_contacts: bool) -> None:
    spawn_cpu_cp_test(
        partial(_local_pair_feats_match_full_then_pad, with_contacts=with_contacts),
        cp_size=2,
    )


def pad_along_msa(
    tensors: list[torch.Tensor], n_pad: int, *, n_token_pad: int
) -> list[torch.Tensor]:
    """Pad MSA depth (dim 1) and tokens (dim 2) in ``(B, S_msa, N_token, ...)``."""
    return [
        pad_dim(pad_dim(tensor, dim=1, pad_len=n_pad), dim=2, pad_len=n_token_pad)
        for tensor in tensors
    ]


def _local_msa_feats_match_full_then_pad(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
) -> None:
    seq = "YDPEG"
    system = System.from_description(proteins=[seq])
    # Lowercase A3M insertions give nonzero deletions at distinct rows and columns, so a
    # misplaced or dropped deletion slab cannot match the reference.
    structure = prepare_structure(
        system,
        msa=MSA({system.chain_ids[0]: ">q\nYDPEG\n>h1\nY-PEaG\n>h2\n-DkkPEG\n"}),
        contacts=[],
        mode="structure",
    )
    assert structure.has_deletion.sum() == 2
    assert structure.deletion_value.unique().numel() == 3
    sizes = make_cp_pad_sizes(
        n_token=structure.num_tokens,
        n_atom=structure.num_nodes,
        n_msa=structure.msa.shape[1],
        atoms_per_window=32,
        mesh=mesh,
    )
    assert sizes.n_token_pad > 0
    assert sizes.n_msa_pad > 0
    batch = batch_collate([structure], window_size=sizes.n_atom_final)

    full = convert_densebatch_to_dictionary(batch)
    keys = [key for key in REQUIRES_MSA_PADDING if key in batch]
    expected = pad_along_msa(
        [full[key] for key in keys], sizes.n_msa_pad, n_token_pad=sizes.n_token_pad
    )
    local = build_shard_msa_feats(batch, sizes, mesh, device=device)

    assert local.keys() == set(keys)
    for key, reference in zip(keys, expected, strict=True):
        feat = local[key]
        assert feat.dtype == reference.dtype
        assert_output_placement(
            feat, expected_placements=msa_placements(reference.shape, mesh), label=key
        )
        feat, reference = feat.float(), reference.to(device).float()
        assert_replicate_values_against_rank0(feat, atol=0)
        gather_n_assert_parity_at_rank0(reference, feat, label=key, atol=0)


def test_local_msa_feats_match_full_then_pad() -> None:
    spawn_cpu_cp_test(_local_msa_feats_match_full_then_pad, cp_size=2)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_local_msa_feats_match_full_then_pad_gpu() -> None:
    spawn_cp_test(_local_msa_feats_match_full_then_pad)
