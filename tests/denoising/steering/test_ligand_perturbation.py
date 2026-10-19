"""Ligand potentials on a real complex: energy fires and the gradient descends.

Myoglobin 101M (heme, sulfate, n-butyl isocyanide) with crystal coordinates.
Each ligand physicality potential is ~0 on the crystal, positive after jittering
only the ligand atoms, differentiable with respect to those atoms, and one
gradient step lowers the energy. Non-bonded cases use the full feature builder
and require the Boltz CCD pickle; the bonded cases use packaged reference tables.
"""

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

from bioemu2.data_interface import System
from bioemu2.data_interface.const import MolType
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
    InterChainVDWOverlapPotential,
    PlanarBondPotential,
    PoseBustersPotential,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate

_FIXTURE_101M = Path(__file__).resolve().parent / "fixtures" / "101m.cif.gz"


@pytest.fixture(scope="module")
def myoglobin():
    """(System, crystal coords [1, N, 3] Å, ligand-atom mask [N])."""
    with TemporaryDirectory() as tmpdir:
        cif = Path(tmpdir) / "101m.cif"
        cif.write_bytes(gzip.decompress(_FIXTURE_101M.read_bytes()))
        system = System.from_cif(cif)
        structure = AllAtomStructure.from_system(system, ref_struct_or_fpath=cif)
    batch = batch_collate([structure])
    n = int(batch.atom_mask.sum())
    ligand = batch.mol_type[0, :n] == MolType.NONPOLYMER.value
    assert ligand.sum() > 0
    return system, batch.coords[:, :n].clone(), ligand


# Buffers used by the default physicality potentials.
_CASES = [
    (PoseBustersPotential, {"bond_buffer": 0.2, "angle_buffer": 0.2, "clash_buffer": 0.15}),
    (BondLengthPotential, {}),
    (BondAnglePotential, {}),
    (InterChainVDWOverlapPotential, {"buffer": 0.225}),
    (PlanarBondPotential, {"buffer": 0.2618}),
]


@pytest.mark.parametrize(("cls", "parameters"), _CASES, ids=[c.__name__ for c, _ in _CASES])
def test_ligand_perturbation_triggers_energy_and_gradient(myoglobin, cls, parameters):
    """Test that moving ligand atoms raises the energy, produces finite gradients,
    and allows one gradient-descent step to lower the energy.
    """
    if cls not in (BondLengthPotential, BondAnglePotential):
        try:
            hf_hub_download(
                repo_id="boltz-community/boltz-1",
                filename="ccd.pkl",
                local_files_only=True,
            )
        except LocalEntryNotFoundError:
            pytest.skip("ccd.pkl not cached; hf_hub_download boltz-community/boltz-1 once")
    system, crystal, ligand = myoglobin
    potential = (
        cls(sigma_tolerance=10.0, parameters=parameters)
        if cls in (BondLengthPotential, BondAnglePotential)
        else cls(parameters=parameters)
    )
    potential.prepare(system)

    e_crystal = potential(crystal, t=0.0)
    if cls in (BondLengthPotential, BondAnglePotential):
        assert e_crystal.item() == 0
    torch.manual_seed(0)
    perturbed = crystal.clone()
    perturbed[:, ligand] += 1.0 * torch.randn_like(perturbed[:, ligand])
    prepared_coords = perturbed.clone().requires_grad_(True)
    e_perturbed = potential(prepared_coords, t=0.0)
    assert e_perturbed.item() > e_crystal.item() + 1.0, (e_crystal, e_perturbed)

    (grad,) = torch.autograd.grad(e_perturbed.sum(), prepared_coords)
    assert torch.isfinite(grad).all()
    assert grad[:, ligand].abs().sum() > 0
    if (
        cls is not InterChainVDWOverlapPotential
    ):  # intra-ligand terms: protein atoms get no gradient
        assert grad[:, ~ligand].abs().sum() == 0

    # Require both real and absent atoms; changing coordinates at masked-out positions
    # must leave the energy and real-atom gradients unchanged.
    atom_pad_mask = potential.feats["atom_pad_mask"][0]
    assert atom_pad_mask.any() and (~atom_pad_mask).any()
    padded_coords = perturbed.new_zeros((1, atom_pad_mask.numel(), 3))
    padded_coords[:, atom_pad_mask] = perturbed
    garbage_coords = (
        padded_coords + 1000.0 * torch.randn_like(padded_coords) * (~atom_pad_mask)[None, :, None]
    )
    padded_coords.requires_grad_(True)
    garbage_coords.requires_grad_(True)
    padded_energy = potential(padded_coords, t=0.0)
    garbage_energy = potential(garbage_coords, t=0.0)
    (padded_grad,) = torch.autograd.grad(padded_energy.sum(), padded_coords)
    (garbage_grad,) = torch.autograd.grad(garbage_energy.sum(), garbage_coords)
    torch.testing.assert_close(garbage_energy, padded_energy)
    torch.testing.assert_close(garbage_grad[:, atom_pad_mask], padded_grad[:, atom_pad_mask])
    assert not garbage_grad[:, ~atom_pad_mask].any()

    e_step = potential((prepared_coords - 0.05 * grad).detach(), t=0.0)
    assert e_step.item() < e_perturbed.item()


@pytest.mark.parametrize("bond_buffer,angle_buffer", [(0.1, 0.3), (0.3, 0.1)])
@pytest.mark.parametrize(
    "distance,expected_energy,expected_gradient",
    [(0.85, 0.05, -1.0), (0.95, 0.0, 0.0), (2.1, 0.0, 0.0), (2.3, 0.1, 1.0)],
)
def test_posebusters_bond_and_angle_pair_uses_smaller_buffer(
    bond_buffer, angle_buffer, distance, expected_energy, expected_gradient
):
    """A pair flagged as both bond and angle uses the smaller buffer on both bounds."""
    feats = {
        "rdkit_bounds_index": torch.tensor([[[0], [1]]]),
        "rdkit_lower_bounds": torch.tensor([[1.0]]),
        "rdkit_upper_bounds": torch.tensor([[2.0]]),
        "rdkit_bounds_bond_mask": torch.tensor([[True]]),
        "rdkit_bounds_angle_mask": torch.tensor([[True]]),
    }
    potential = PoseBustersPotential(
        parameters={
            "bond_buffer": bond_buffer,
            "angle_buffer": angle_buffer,
            "clash_buffer": 0.0,
        }
    )
    coords = torch.tensor([[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]], requires_grad=True)
    energy = potential.compute(coords, feats, potential.compute_parameters(0.0))
    torch.testing.assert_close(energy, torch.tensor([expected_energy]))
    (gradient,) = torch.autograd.grad(energy.sum(), coords)
    torch.testing.assert_close(
        gradient,
        torch.tensor([[[-expected_gradient, 0.0, 0.0], [expected_gradient, 0.0, 0.0]]]),
    )


def test_dihedral_exactly_planar_has_finite_gradient():
    """float32 planar dihedrals: phi is exact (pi trans, 0 cis) and autograd is finite.

    Upstream's sign(.) * arccos(clamp(., 1 - 1e-8)) returned phi = 0 for trans and a
    NaN gradient, which propagated into every coordinate under GD guidance.
    """
    coords = torch.tensor(
        [
            [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # trans
            [[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # cis
        ],
        requires_grad=True,
    )
    index = torch.tensor([[0], [1], [2], [3]])
    phi = PlanarBondPotential().compute_variable(coords, index)
    (grad,) = torch.autograd.grad(phi.sum(), coords)
    assert torch.allclose(phi.flatten(), torch.tensor([torch.pi, 0.0]))
    assert torch.isfinite(grad).all()
