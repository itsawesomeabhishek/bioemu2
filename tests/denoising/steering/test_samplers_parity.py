"""CPU parity test for the steered EDM sampler.

Asserts that `edm_steered_sampler` without potentials produces bit-exact
output identical to `edm_sampler`. Uses a stub score model so it runs
on CPU with no checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.data_interface import Protein, System
from bioemu2.denoising.edm import edm_sampler
from bioemu2.denoising.samplers import GuidedFKSampler
from bioemu2.denoising.steering.edm_steered import edm_steered_sampler
from bioemu2.denoising.steering.potentials import BondAnglePotential, BondLengthPotential
from bioemu2.model import InferenceModel
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate

_DUMMY_CONTEXT = SimpleNamespace(
    coords_cond=None,
    is_dynamics_conditioned=None,
    dt=None,
)
_DUMMY_TRUNK = cast(TrunkEmbeddings, object())


class _GaussianScoreModel:
    """Optimal denoiser for a unit-Gaussian prior p(x₀) = N(0, I).

    E[x₀ | xₜ, σ] = xₜ / (σ² + 1)

    This makes the Heun update non-trivial (coords shrink toward 0 each step),
    so the parity test exercises the full loop rather than a no-op.
    """

    edm_schedule = EDMSchedule()

    def eval(self):
        return self

    def compute_static_context(self, batch, trunk_embeddings, multiplicity):
        assert trunk_embeddings is _DUMMY_TRUNK
        return _DUMMY_CONTEXT

    def denoise(self, context, trunk_embeddings, coords, sigmas):
        assert trunk_embeddings is _DUMMY_TRUNK
        denoised = coords / (sigmas[:, None, None] ** 2 + 1)
        return SimpleNamespace(denoised_coords=denoised)


def _kwargs(num_steps=5):
    return dict(
        num_steps=num_steps,
        step_scale=1.0,
        noise_scale=1.0,
    )


def test_edm_steered_no_steering_is_bit_exact(test_structure):
    """`edm_steered_sampler` with no potentials must match `edm_sampler` exactly,
    on non-zero inputs and producing non-zero outputs."""
    # Seed the init draw so the starting coords are deterministic + non-zero —
    # an all-zeros tensor would let trivial shape/dtype bugs slip through.
    # The sampler-internal manual_seed(0) calls below reset RNG for parity.
    torch.manual_seed(42)
    test_structure = test_structure.replace(coords=torch.randn(test_structure.num_nodes, 3))
    assert test_structure.coords.abs().sum() > 0, "input coords must be non-zero"
    batch = batch_collate([test_structure])
    score_model = _GaussianScoreModel()
    torch.manual_seed(0)
    ref = edm_sampler(
        batch=batch,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        **_kwargs(),
    )
    torch.manual_seed(0)
    steered = edm_steered_sampler(
        batch=batch,
        score_model=cast(InferenceModel, score_model),
        trunk_embeddings=_DUMMY_TRUNK,
        **_kwargs(),
    )

    assert ref.coords.abs().sum() > 0, "edm_sampler produced all-zero coords"
    assert steered.coords.abs().sum() > 0, "edm_steered_sampler produced all-zero coords"
    assert torch.equal(ref.coords, steered.coords)


def test_edm_steered_rejects_batch_with_multiplicity(test_structure):
    batch_size = 2
    multiplicity = 3
    batch = batch_collate([test_structure] * batch_size)

    with pytest.raises(NotImplementedError, match="Steered samplers take one structure per call"):
        edm_steered_sampler(
            batch=batch,
            score_model=cast(InferenceModel, _GaussianScoreModel()),
            trunk_embeddings=_DUMMY_TRUNK,
            multiplicity=multiplicity,
            **_kwargs(num_steps=2),
        )


@pytest.mark.parametrize("potential_type", [BondLengthPotential, BondAnglePotential])
def test_default_protein_bond_guidance_changes_samples(potential_type):
    """Test that each default bonded potential changes protein samples from the
    same-seed unsteered run without producing NaN or infinite coordinates.
    """
    system = System(entities=[Protein(sequence="AAA", asym_ids="A")])
    batch = batch_collate([AllAtomStructure.from_system(system, mode="structure")])
    guided_potential = next(
        p for p in GuidedFKSampler().potentials if isinstance(p, potential_type)
    )
    guided_potential.prepare(system)

    def run(potentials):
        torch.manual_seed(17)
        return edm_steered_sampler(
            batch=batch,
            score_model=cast(InferenceModel, _GaussianScoreModel()),
            trunk_embeddings=_DUMMY_TRUNK,
            potentials=potentials,
            resampling_interval=None,
            num_gd_steps=2,
            alignment_reverse_diff=False,
            **_kwargs(num_steps=4),
        ).coords

    guided = run([guided_potential])
    unsteered = run([])
    assert torch.isfinite(guided).all()
    assert not torch.allclose(guided, unsteered)
