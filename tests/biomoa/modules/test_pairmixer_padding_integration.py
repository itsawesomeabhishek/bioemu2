import torch

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.structure import AllAtomStructure, add_mode_conditioning, extract_substructure
from bioemu2.structure.batch import batch_collate
from tests.conftest import set_matmul_precision
from tests.msa.features import add_dummy_msa_features
from tests.structure.conftest import build_test_structure

S_PADDING_ATOL = 5e-7
Z_PADDING_ATOL = 5e-6


class TestPairMixerPaddingIntegration:
    def test_padding_does_not_change_embeddings(
        self,
        test_structure: AllAtomStructure,
        toy_biomoa_model: BioMoaCModel,
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = toy_biomoa_model.to(device)

        torch.manual_seed(0)
        long_system = test_structure.to(device)
        full = build_test_structure("YDPE")
        short_system = extract_substructure(full, token_mask=torch.arange(full.num_tokens) < 2)
        short_system.coords = torch.randn(short_system.num_nodes, 3)
        short_system.template_coords = torch.randn(short_system.num_nodes, 3)
        short_system.charge = torch.randn(short_system.num_nodes)
        short_system = add_mode_conditioning(
            add_dummy_msa_features(short_system), mode="NON-MD"
        ).to(device)
        assert short_system.num_tokens < long_system.num_tokens

        with torch.no_grad(), set_matmul_precision("highest"):
            batch_short_alone = batch_collate([short_system]).to(device)
            embeddings_short_alone = model.compute_trunk_embeddings(
                batch_short_alone, num_steps_recycle=0
            )

            batch_long_alone = batch_collate([long_system]).to(device)
            embeddings_long_alone = model.compute_trunk_embeddings(
                batch_long_alone, num_steps_recycle=0
            )

            batch_ragged = batch_collate([short_system, long_system]).to(device)
            embeddings_ragged = model.compute_trunk_embeddings(batch_ragged, num_steps_recycle=0)

        for index, embeddings_alone, num_tokens in [
            (0, embeddings_short_alone, short_system.num_tokens),
            (1, embeddings_long_alone, long_system.num_tokens),
        ]:
            torch.testing.assert_close(
                embeddings_ragged.single[index, :num_tokens],
                embeddings_alone.single[0, :num_tokens],
                atol=S_PADDING_ATOL,
                rtol=0.0,
            )
            torch.testing.assert_close(
                embeddings_ragged.pair[index, :num_tokens, :num_tokens],
                embeddings_alone.pair[0, :num_tokens, :num_tokens],
                atol=Z_PADDING_ATOL,
                rtol=0.0,
            )
