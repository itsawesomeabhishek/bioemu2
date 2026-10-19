"""End-to-end regression test: batched multiplicity matches sequential ``M=1``."""

import pytest
import torch

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate


class TestToyModelMultiplicityIntegration:
    def test_forward_composes_explicit_operations(
        self,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = toy_biomoa_model.to(device)
        batch = batch_collate([test_structure]).to(device)
        sigmas = torch.full((1,), 0.34, device=device)

        with torch.no_grad():
            expected = model(batch, sigmas)
            trunk_embeddings = model.compute_trunk_embeddings(batch)
            context = model.compute_static_context(batch, trunk_embeddings)
            actual = model.denoise(context, trunk_embeddings, batch.coords, sigmas)

        torch.testing.assert_close(actual.score, expected.score)
        torch.testing.assert_close(actual.denoised_coords, expected.denoised_coords)
        torch.testing.assert_close(trunk_embeddings.distogram, expected.distogram)

    @pytest.mark.parametrize("multiplicity", [3, 5, 7])
    def test_batched_score_matches_sequential(
        self,
        multiplicity: int,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = toy_biomoa_model.to(device)

        # Run all coordinate-independent work once on a template batch.
        with torch.no_grad():
            template_batch = batch_collate([test_structure]).to(device)
            trunk_embeddings = model.compute_trunk_embeddings(template_batch, num_steps_recycle=0)
            context = model.compute_static_context(template_batch, trunk_embeddings)

        # Each noise replicate gets distinct random coords; share the same sigma.
        torch.manual_seed(42)
        list_coords = [torch.randn_like(template_batch.coords) for _ in range(multiplicity)]

        coords_M = torch.cat(list_coords, dim=0)
        sigmas_M = torch.full((multiplicity,), 0.34, device=device)
        list_sigmas = [sigmas_M[i : i + 1] for i in range(multiplicity)]

        with torch.no_grad():
            out_batched = model.denoise(context, trunk_embeddings, coords_M, sigmas_M)
            out_sequential = torch.cat(
                [
                    model.denoise(context, trunk_embeddings, list_coords[i], list_sigmas[i]).score
                    for i in range(multiplicity)
                ],
                dim=0,
            )
        torch.testing.assert_close(out_batched.score, out_sequential, atol=1e-5, rtol=1e-5)

    def test_batch_size_and_multiplicity_are_independent(
        self,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = toy_biomoa_model.to(device)
        second_structure = test_structure.clone()
        assert second_structure.template_coords is not None
        second_structure.template_coords = second_structure.template_coords + 0.25
        structures = [test_structure, second_structure]
        batch_size = len(structures)
        multiplicity = 3

        batch = batch_collate(structures).to(device)
        torch.manual_seed(43)
        coords = torch.randn(
            batch_size * multiplicity,
            *batch.coords.shape[1:],
            device=device,
        )
        sigmas = torch.linspace(0.1, 0.6, batch_size * multiplicity, device=device)

        with torch.no_grad():
            trunk_embeddings = model.compute_trunk_embeddings(batch)
            context = model.compute_static_context(batch, trunk_embeddings)
            actual = model.denoise(context, trunk_embeddings, coords, sigmas)

            expected_scores = []
            expected_denoised_coords = []
            for batch_index, structure in enumerate(structures):
                single_batch = batch_collate([structure]).to(device)
                single_trunk_embeddings = model.compute_trunk_embeddings(single_batch)
                single_context = model.compute_static_context(single_batch, single_trunk_embeddings)
                start = batch_index * multiplicity
                stop = start + multiplicity
                output = model.denoise(
                    single_context,
                    single_trunk_embeddings,
                    coords[start:stop],
                    sigmas[start:stop],
                )
                expected_scores.append(output.score)
                expected_denoised_coords.append(output.denoised_coords)

        torch.testing.assert_close(
            actual.score,
            torch.cat(expected_scores),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            actual.denoised_coords,
            torch.cat(expected_denoised_coords),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_denoise_validates_batch_dimensions(
        self,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        batch = batch_collate([test_structure, test_structure.clone()])
        with torch.no_grad():
            trunk_embeddings = toy_biomoa_model.compute_trunk_embeddings(batch)
            context = toy_biomoa_model.compute_static_context(batch, trunk_embeddings)

        coords = torch.randn(3, *batch.coords.shape[1:])
        sigmas = torch.full((3,), 0.34)
        with pytest.raises(ValueError, match="not divisible by context batch size 2"):
            toy_biomoa_model.denoise(context, trunk_embeddings, coords, sigmas)

        coords = torch.randn(4, *batch.coords.shape[1:])
        with pytest.raises(ValueError, match="Sigma batch size 3"):
            toy_biomoa_model.denoise(context, trunk_embeddings, coords, sigmas)
