"""Bonded-potential angle numerics and system preparation."""

import math

import pytest
import torch

from bioemu2.data_interface import DNA, RNA, Ligand, Protein, System
from bioemu2.denoising.steering.potentials import (
    BondAnglePotential,
    BondLengthPotential,
)


def _evaluated(potential, coords):
    coords = coords.clone().requires_grad_(True)
    energy = potential(coords, t=0.5)
    return energy.detach(), torch.autograd.grad(energy.sum(), coords)[0]


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_bonded_tolerance_must_be_finite_and_nonnegative(cls, tolerance):
    """Test that negative, NaN, and infinite tolerances are rejected before preparing bounds."""
    with pytest.raises(ValueError, match="sigma_tolerance"):
        cls(sigma_tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [0.0, 10.0])
def test_prepared_reference_bounds_use_requested_tolerance(tolerance):
    """Test that bond and angle bounds equal the reference mean plus/minus tolerance times sigma."""
    system = System(entities=[Protein(sequence="A", asym_ids="A")])
    atoms = {atom.name: atom.index for atom in system.topology.atoms}
    for cls, prefix, names, mean, sigma in (
        (BondLengthPotential, "bond", ("N", "CA"), 1.482, 0.0101),
        (BondAnglePotential, "angle", ("N", "C", "CA"), math.radians(109.627), math.radians(1.5)),
    ):
        potential = cls(sigma_tolerance=tolerance)
        potential.prepare(system)
        index = torch.tensor([atoms[name] for name in names])
        selected = (potential.feats[f"{prefix}_index"][0].T == index).all(dim=-1)
        assert selected.sum() == 1
        bounds = [
            potential.feats[f"{prefix}_{side}"][0, selected].item() for side in ("lower", "upper")
        ]
        assert bounds == pytest.approx(
            (mean - tolerance * sigma, mean + tolerance * sigma), abs=1e-6
        )


def _angle_coords(degrees, first=1.0, second=1.0):
    theta = math.radians(degrees)
    return torch.tensor(
        [[first, 0, 0], [0, 0, 0], [second * math.cos(theta), second * math.sin(theta), 0]],
        dtype=torch.float64,
    ).unsqueeze(0)


def _angle_feats(dtype=torch.float64):
    return {
        "angle_index": torch.tensor([[[0], [2], [1]]]),
        "angle_lower": torch.tensor([[math.radians(100)]], dtype=dtype),
        "angle_upper": torch.tensor([[math.radians(140)]], dtype=dtype),
    }


@pytest.mark.parametrize("angle,expected", [(90, 10), (100, 0), (120, 0), (140, 0), (150, 10)])
@pytest.mark.parametrize("first,second", [(1, 1), (0.7, 2.5), (4, 0.8)])
def test_angle_hinge_is_in_radians_and_arm_length_invariant(angle, expected, first, second):
    """Test radian penalties at and outside the angle bounds for different bond
    lengths, and check coordinate gradients against numerical derivatives.
    """
    coords = _angle_coords(angle, first, second).requires_grad_(True)
    potential = BondAnglePotential(sigma_tolerance=10.0)
    assert potential.compute(coords, _angle_feats(), {}).item() == pytest.approx(
        math.radians(expected), abs=1e-12
    )
    if expected:
        assert torch.autograd.gradcheck(
            lambda positions: potential.compute(positions, _angle_feats(), {}), (coords,)
        )


@pytest.mark.parametrize(
    "first,second", [(0, 0), (1, 2), (1, -2)], ids=["coincident", "collinear", "antiparallel"]
)
def test_degenerate_angle_has_finite_energy_and_gradient(first, second):
    """Test that coincident (undefined angle), parallel (0), and antiparallel (pi)
    atom triples produce neither NaN nor infinite energies or coordinate gradients.
    """
    coords = torch.tensor(
        [[[first, 0, 0], [0, 0, 0], [second, 0, 0]]], dtype=torch.float32, requires_grad=True
    )
    energy = BondAnglePotential(sigma_tolerance=10.0).compute(
        coords, _angle_feats(torch.float32), {}
    )
    assert torch.isfinite(energy).all()
    assert torch.isfinite(torch.autograd.grad(energy.sum(), coords)[0]).all()


def test_ligand_angle_fallback_allows_linear_but_penalizes_folded_geometry():
    """Test that a linear C-C-O angle has zero penalty, while a folded angle has
    a positive penalty and a nonzero, finite coordinate gradient.
    """
    potential = BondAnglePotential(sigma_tolerance=10.0)
    potential.prepare(System(entities=[Ligand(smiles="CCO", asym_ids="L")]))
    linear = torch.tensor([[[-1.5, 0, 0], [0, 0, 0], [1.5, 0, 0]]])
    folded = torch.tensor([[[1.5, 0, 0], [0, 0, 0], [1.5, 0.15, 0]]])
    assert not potential(linear, t=0.5).any()
    energy, gradient = _evaluated(potential, folded)
    assert energy.item() > 0 and torch.isfinite(gradient).all() and gradient.abs().max() > 0


@pytest.mark.parametrize("cls", [BondLengthPotential, BondAnglePotential])
def test_bonded_repreparation_and_padding_preserve_energies_and_gradients(cls):
    """Test that preparing a reused potential matches a fresh one for each system,
    and that padding atoms changes neither energies nor real-atom gradients.
    """
    potential = cls(sigma_tolerance=10.0)
    for polymer, sequence in ((Protein, "AXG"), (DNA, "AC"), (RNA, "GU"), (Protein, "A")):
        system = System(entities=[polymer(sequence=sequence, asym_ids="A")])
        potential.prepare(system)
        fresh = cls(sigma_tolerance=10.0)
        fresh.prepare(system)
        coords = torch.randn(
            2, system.topology.n_atoms, 3, generator=torch.Generator().manual_seed(83)
        )
        expected_energy, expected_gradient = _evaluated(fresh, coords)
        assert expected_energy.min() > 0 and expected_gradient.abs().max() > 0
        # Exercise both collate-window padding and additional context-parallel padding.
        padding = potential.feats["atom_pad_mask"].shape[1] + 5 - coords.shape[1]
        padded = torch.cat((coords, torch.full((2, padding, 3), 1000.0)), dim=1)
        energy, gradient = _evaluated(potential, padded)
        torch.testing.assert_close(energy, expected_energy, rtol=0, atol=0)
        torch.testing.assert_close(
            gradient[:, : coords.shape[1]], expected_gradient, rtol=0, atol=0
        )
        assert not gradient[:, coords.shape[1] :].any()
