import torch

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.structure import AllAtomStructure, BatchAllAtomStructure
from bioemu2.structure.batch import batch_collate


def _ito_batch(structure: AllAtomStructure) -> BatchAllAtomStructure:
    structure = structure.replace(
        coords_cond=structure.coords - 1.0,
        dt=torch.tensor([10.0]),
    )
    return batch_collate([structure])


def test_conditioning_signal_reaches_score(
    test_structure: AllAtomStructure,
    toy_biomoa_ito_model: BioMoaCModel,
) -> None:
    batch = _ito_batch(test_structure)
    sigmas = torch.full((1,), 0.1)
    assert bool(batch.is_dynamics_conditioned.all())

    with torch.no_grad():
        score = toy_biomoa_ito_model(batch, sigmas).score
        other_frame = toy_biomoa_ito_model(
            batch.replace(coords_cond=batch.coords_cond + 3.0), sigmas
        ).score
        other_lag = toy_biomoa_ito_model(batch.replace(dt=batch.dt * 1000.0), sigmas).score
        masked = toy_biomoa_ito_model(
            batch.replace(is_dynamics_conditioned=torch.zeros_like(batch.is_dynamics_conditioned)),
            sigmas,
        ).score

    assert not torch.allclose(score, other_frame)
    assert not torch.allclose(score, other_lag)
    assert not torch.allclose(score, masked)


def test_unconditional_model_ignores_conditioning_fields(
    test_structure: AllAtomStructure,
    toy_biomoa_model: BioMoaCModel,
) -> None:
    batch = _ito_batch(test_structure)
    sigmas = torch.full((1,), 0.1)

    with torch.no_grad():
        score = toy_biomoa_model(batch, sigmas).score
        other_frame = toy_biomoa_model(
            batch.replace(coords_cond=batch.coords_cond + 3.0), sigmas
        ).score

    torch.testing.assert_close(score, other_frame)
