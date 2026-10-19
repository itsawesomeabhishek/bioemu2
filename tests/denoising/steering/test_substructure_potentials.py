"""Behavioural tests for the ported substructure potentials
`ContactPotential` and `TemplateReferencePotential` (see
`bioemu2.denoising.steering.potentials`).

Each test asserts a real behavioural invariant, not just shapes:

1. Contact energy is 0 when the target pair is within threshold, positive when
   they are far apart, and its autograd gradient pulls them together.
2. Template energy is ~0 when coords match the reference template, positive
   after coord perturbation, and its autograd gradient points back toward the
   template.
"""

import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bioemu2.data_interface import Ligand, Protein, System
from bioemu2.denoising.steering.feats import build_potential_features
from bioemu2.denoising.steering.potentials import (
    BondLengthPotential,
    ContactPotential,
    InterChainVDWOverlapPotential,
    PiecewiseStepFunction,
    TemplateReferencePotential,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate

_FIXTURE_101M = Path(__file__).resolve().parent / "fixtures" / "101m.cif.gz"


def _protein_only_task() -> System:
    return System(entities=[Protein(asym_ids=["A"], sequence="GYDPETGTWG")])


@pytest.fixture(scope="module")
def batch_101m():
    """Full 101M myoglobin with crystallographic coordinates.

    The contact and template invariants below need real 3D coords: an
    inference-input structure carries all-zero placeholder coords (the positions
    are what the diffusion samples), which collapses every pairwise distance to
    zero and leaves the gradient direction undefined. We load the topology with
    ``System.from_cif`` and map the crystal coordinates onto it via
    ``ref_struct_or_fpath``.
    """
    with TemporaryDirectory() as tmpdir:
        cif_path = Path(tmpdir) / "101m.cif"
        cif_path.write_bytes(gzip.decompress(_FIXTURE_101M.read_bytes()))
        task = System.from_cif(cif_path)
        structure = AllAtomStructure.from_system(task, ref_struct_or_fpath=cif_path)
    return batch_collate([structure])


@pytest.fixture(scope="module")
def feats_101m(batch_101m):
    return build_potential_features(batch_101m)


def _inject_contact(feats, atom_i, atom_j, threshold):
    """Copy `feats` with contact_pair_index/contact_thresholds injected for one
    pair. Returns a fresh dict — caller keeps the module-scoped fixture clean."""
    device = feats["coords"].device
    out = dict(feats)
    out["contact_pair_index"] = torch.tensor(
        [[[atom_i], [atom_j]]], dtype=torch.long, device=device
    )  # [1, 2, 1]
    out["contact_thresholds"] = torch.tensor(
        [[threshold]], dtype=torch.float32, device=device
    )  # [1, 1]
    return out


def _pick_real_atoms(feats) -> tuple[int, int]:
    """Return two distinct unpadded atom indices from feats.

    Anchoring the tests to `atom_pad_mask` — instead of hard-coded 0/3 — keeps
    them meaningful if the canonical atom ordering ever shifts padding into
    the low indices.
    """
    real = feats["atom_pad_mask"][0].bool().nonzero().flatten().tolist()
    assert len(real) >= 2, "batch has fewer than 2 real atoms"
    return real[0], real[3] if len(real) > 3 else real[1]


def test_contact_potential_zero_when_within_threshold(feats_101m):
    # Two real atoms of chain 0, threshold set well above their crystal distance.
    atom_i, atom_j = _pick_real_atoms(feats_101m)
    coords = feats_101m["coords"]
    d = torch.linalg.norm(coords[0, atom_i] - coords[0, atom_j]).item()
    feats = _inject_contact(feats_101m, atom_i=atom_i, atom_j=atom_j, threshold=d + 5.0)
    potential = ContactPotential()
    potential.feats = feats
    energy = potential(coords, t=0.0)
    assert energy.shape == (1,)
    assert torch.allclose(energy, torch.zeros_like(energy), atol=1e-5), (
        f"expected zero energy inside flat bottom, got {energy}"
    )
    assert "x0_hat" not in feats


@pytest.mark.parametrize("n_atoms", [0, 5], ids=["empty-atoms", "no-contacts"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_contact_potential_empty_restraints_have_connected_zero(n_atoms, dtype):
    coords = torch.randn(2, n_atoms, 3, dtype=dtype, requires_grad=True)
    feats = {
        "contact_pair_index": torch.empty(1, 2, 0, dtype=torch.long),
        "contact_thresholds": torch.empty(1, 0),
    }
    energy = ContactPotential().compute(coords, feats)
    assert energy.shape == (2,) and energy.dtype == coords.dtype and energy.device == coords.device
    assert torch.equal(energy, torch.zeros_like(energy))
    (grad,) = torch.autograd.grad(energy.sum(), coords)
    assert torch.equal(grad, torch.zeros_like(coords))


def test_contact_potential_positive_and_gradient_pulls_together(feats_101m):
    # Same two atoms but with a tight threshold: energy should be positive and
    # its autograd gradient at atom_i should point along (atom_i - atom_j),
    # so a gradient-descent step pulls them together.
    atom_i, atom_j = _pick_real_atoms(feats_101m)
    coords = feats_101m["coords"].clone().detach().requires_grad_(True)
    d = torch.linalg.norm(coords[0, atom_i] - coords[0, atom_j]).detach().item()
    feats = _inject_contact(feats_101m, atom_i=atom_i, atom_j=atom_j, threshold=d - 1.0)
    potential = ContactPotential()
    potential.feats = feats
    energy = potential(coords, t=0.0).sum()
    assert energy.item() > 0.0, f"expected positive violation energy, got {energy.item()}"

    (grad,) = torch.autograd.grad(energy, coords)
    # A flat-bottom upper penalty gradient at atom_i points along (i - j) / |i - j|
    # — descending on this gradient shrinks their distance.
    r_ij_hat = (coords[0, atom_i] - coords[0, atom_j]).detach()
    r_ij_hat = r_ij_hat / r_ij_hat.norm()
    cos_sim = torch.dot(grad[0, atom_i], r_ij_hat) / grad[0, atom_i].norm()
    assert cos_sim > 0.99, f"gradient at atom_i not aligned with (i - j): cos={cos_sim}"


def _template_feats_from_batch(feats, threshold):
    """Inject template feats derived from the batch's own crystal coords.

    template_cb: sampled coords at each token's rep atom.
    template_mask_cb: any token whose rep atom is unpadded.
    """
    device = feats["coords"].device
    rep_atom_idx = feats["token_center_atom_index"][0]  # [T]
    ref = feats["coords"][0, rep_atom_idx]  # [T, 3]
    atom_pad = feats["atom_pad_mask"][0].bool()
    ref_mask = atom_pad[rep_atom_idx]  # [T]

    out = dict(feats)
    out["template_cb"] = ref[None].float()  # [1, T, 3]
    out["template_mask_cb"] = ref_mask[None]  # [1, T]
    out["template_force_threshold"] = torch.tensor([threshold], dtype=torch.float32, device=device)
    return out


def test_template_potential_zero_at_reference(feats_101m):
    # Self-template: reference = current coords. After weighted_rigid_align the
    # aligned ref should coincide with the sampled coords → per-token distance
    # is ~0 → energy is ~0.
    coords = feats_101m["coords"]
    feats = _template_feats_from_batch(feats_101m, threshold=1.0)
    potential = TemplateReferencePotential()
    potential.feats = feats
    energy = potential(coords, t=0.0)
    assert energy.shape == (1,)
    assert energy.item() < 1e-2, f"expected ~0 energy at self-template, got {energy.item()}"


@pytest.mark.parametrize("n_atoms", [0, 5], ids=["empty-atoms", "masked-reference"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_template_potential_empty_reference_has_connected_zero(n_atoms, dtype):
    coords = torch.randn(2, n_atoms, 3, dtype=dtype, requires_grad=True)
    feats = {
        "template_cb": torch.zeros(1, 1, 3),
        "template_mask_cb": torch.zeros(1, 1, dtype=torch.bool),
    }
    energy = TemplateReferencePotential().compute(coords, feats)
    assert energy.shape == (2,) and energy.dtype == coords.dtype and energy.device == coords.device
    assert torch.equal(energy, torch.zeros_like(energy))
    (grad,) = torch.autograd.grad(energy.sum(), coords)
    assert torch.equal(grad, torch.zeros_like(coords))


def test_template_potential_positive_and_gradient_points_back(feats_101m):
    # Perturb the sampled coords well past threshold → energy positive, autograd
    # gradient reduces energy when descended (checked via finite-difference sign).
    feats = _template_feats_from_batch(feats_101m, threshold=0.5)
    perturbed = feats_101m["coords"].clone().detach()
    torch.manual_seed(0)
    # 3 Å isotropic Gaussian jitter on real atoms only.
    atom_pad = feats["atom_pad_mask"][0].bool()
    perturbed[0, atom_pad] += 3.0 * torch.randn_like(perturbed[0, atom_pad])
    perturbed = perturbed.requires_grad_(True)

    potential = TemplateReferencePotential()
    potential.feats = feats
    energy = potential(perturbed, t=0.0).sum()
    assert energy.item() > 0.0, f"expected positive violation energy, got {energy.item()}"

    (grad,) = torch.autograd.grad(energy, perturbed)
    # Descend one small step and check the energy drops. A cleaner invariant
    # than picking a specific direction, and robust to rigid-align degeneracies.
    step = 1e-3
    with torch.no_grad():
        new_coords = perturbed - step * grad
    new_energy = potential(new_coords, t=0.0).sum()
    assert new_energy.item() < energy.item(), (
        f"gradient descent step should reduce energy, got {energy.item()} -> {new_energy.item()}"
    )


def test_potential_parameters_recursively_instantiated():
    # A potential whose `parameters` carries a nested ParameterSchedule `_target_`
    # must come out with the schedule resolved to a live object, not a DictConfig.
    cfg = OmegaConf.create(
        {
            "_target_": "bioemu2.denoising.steering.potentials.InterChainVDWOverlapPotential",
            "parameters": {
                "guidance_interval": 5,
                "guidance_weight": {
                    "_target_": ("bioemu2.denoising.steering.potentials.PiecewiseStepFunction"),
                    "thresholds": [0.4],
                    "values": [0.125, 0.0],
                },
                "buffer": 0.225,
            },
        }
    )
    pot = instantiate(cfg)
    assert isinstance(pot, InterChainVDWOverlapPotential)
    gw = pot.parameters["guidance_weight"]
    assert isinstance(gw, PiecewiseStepFunction)
    assert gw.compute(0.0) == 0.125 and gw.compute(1.0) == 0.0


def test_feats_potentials_prepare_independent_features():
    """Test that two potentials keep separate feature dictionaries after
    preparation and evaluation.
    """
    first = BondLengthPotential(sigma_tolerance=10.0, parameters={"guidance_weight": 0.05})
    second = InterChainVDWOverlapPotential()
    system = _protein_only_task()
    first.prepare(system)
    second.prepare(system)

    assert first.feats is not second.feats
    assert first.feats["asym_id"].shape[0] == 1
    coords = torch.zeros(2, first.feats["atom_pad_mask"].shape[1], 3)
    assert first(coords, t=0.0).shape == (2,)
    assert first.feats is not second.feats


@pytest.mark.parametrize("distance,energy,slope", [(0.5, 0.3, -1), (1.5, 0, 0), (3.0, 0.5, 1)])
def test_fixed_bond_window_changes_energy_and_gradient(distance, energy, slope):
    """Test that a two-atom ligand has zero penalty inside its bond-length bounds
    and the expected linear penalty and coordinate gradient outside either bound.
    """
    potential = BondLengthPotential(sigma_tolerance=10.0)
    potential.prepare(System(entities=[Ligand(smiles="CC", asym_ids="L")]))
    coords = torch.tensor([[[0.0, 0, 0], [distance, 0, 0]]], requires_grad=True)
    actual = potential(coords, t=0.4)
    assert actual.item() == pytest.approx(energy)
    gradient = torch.autograd.grad(actual.sum(), coords)[0]
    torch.testing.assert_close(gradient, coords.new_tensor([[[-slope, 0, 0], [slope, 0, 0]]]))
