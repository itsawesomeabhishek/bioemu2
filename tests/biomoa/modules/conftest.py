"""Fixtures for diffusion module tests.

Builds a minimal ``feats`` dict from the synthetic 4-residue YDPE structure
(no network access) via ``build_test_structure → batch_collate →
convert_densebatch_to_dictionary``. The atom dimension is padded to a multiple
of 32 by ``batch_collate``, so ``atoms_per_window_queries=32`` works out of the box.
"""

import pytest
import torch

from bioemu2.biomoa.modules.encoders import LEN_POCKET_CONTACT_INFO, NUM_MSA_TOKENS, NUM_TOKEN_TYPES
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.structure.batch import batch_collate
from tests.msa.features import add_dummy_msa_features
from tests.structure.conftest import build_test_structure

# Module-level constants used by tests.  Kept small so forward passes are fast.
TOKEN_S = 32
TOKEN_Z = 16
ATOM_S = 16
ATOM_Z = 8
# Matches InputEmbedder output: token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes
S_INPUT_DIM = TOKEN_S + NUM_TOKEN_TYPES + NUM_MSA_TOKENS + 1 + LEN_POCKET_CONTACT_INFO
ATOMS_PER_WINDOW_QUERIES = 32
ATOMS_PER_WINDOW_KEYS = 128


@pytest.fixture(scope="session")
def real_feats() -> dict[str, torch.Tensor]:
    """Minimal ``feats`` dict for diffusion forward passes (no network access).

    Uses the synthetic YDPE 4-residue structure with dummy MSA features, giving 4 tokens and
    64 atoms (padded to window size). Reference positions (``template_coords``) and ``charge``
    are supplied as random per-atom features to keep the fixture self-contained.
    """
    s = build_test_structure("YDPE")
    s.coords = torch.randn(s.num_nodes, 3)
    s.template_coords = torch.randn(s.num_nodes, 3)
    s.charge = torch.randn(s.num_nodes)
    s = add_dummy_msa_features(s)
    batch = batch_collate([s])
    return convert_densebatch_to_dictionary(batch)


@pytest.fixture
def feats_shapes(real_feats: dict[str, torch.Tensor]) -> dict[str, int]:
    """Convenience accessor for the key shape constants of the fixture."""
    B, N_atoms, _ = real_feats["template_coords"].shape
    N_token = real_feats["token_pad_mask"].shape[1]
    return {"B": B, "N_atoms": N_atoms, "N_token": N_token}
