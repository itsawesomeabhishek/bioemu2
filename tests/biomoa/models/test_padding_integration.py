"""Regression test: a ragged batch must not change any system's score."""

import pytest
import torch

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.structure import AllAtomStructure, add_mode_conditioning, extract_substructure
from bioemu2.structure.batch import batch_collate
from tests.msa.features import add_dummy_msa_features
from tests.structure.conftest import build_test_structure


class TestToyModelPaddingIntegration:
    @pytest.mark.parametrize("n_residues_short", [1, 2, 3])
    def test_scores_are_padding_invariant(
        self,
        n_residues_short: int,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = toy_biomoa_model.to(device)

        # Short system (a residue prefix of YDPE) co-batched with the full YDPE
        # (test_structure); collate pads the short one up -> inter-sample padding.
        torch.manual_seed(0)
        long_system = test_structure.to(device)
        full = build_test_structure("YDPE")
        short_system = extract_substructure(
            full, token_mask=torch.arange(full.num_tokens) < n_residues_short
        )
        short_system.coords = torch.randn(short_system.num_nodes, 3)
        short_system.template_coords = torch.randn(short_system.num_nodes, 3)
        short_system.charge = torch.randn(short_system.num_nodes)
        short_system = add_mode_conditioning(
            add_dummy_msa_features(short_system), mode="NON-MD"
        ).to(device)
        assert short_system.num_nodes < long_system.num_nodes

        sigma = 0.34
        with torch.no_grad():
            batch_short_alone = batch_collate([short_system]).to(device)
            score_short_alone = model(
                batch_short_alone, torch.full((1,), sigma, device=device)
            ).score[0]

            batch_long_alone = batch_collate([long_system]).to(device)
            score_long_alone = model(
                batch_long_alone, torch.full((1,), sigma, device=device)
            ).score[0]

            batch_ragged = batch_collate([short_system, long_system]).to(device)
            score_ragged = model(batch_ragged, torch.full((2,), sigma, device=device)).score

        n_short = short_system.num_nodes
        n_long = long_system.num_nodes
        torch.testing.assert_close(
            score_ragged[0, :n_short], score_short_alone[:n_short], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            score_ragged[1, :n_long], score_long_alone[:n_long], atol=1e-5, rtol=1e-5
        )
