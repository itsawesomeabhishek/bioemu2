"""Shared fixtures for biomoa tests.

Provides a test structure for unit tests, matching the sequence_distill pattern.
"""

from typing import Any

import pytest
import torch

from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.biomoa.modules.encoders import LEN_POCKET_CONTACT_INFO, NUM_MSA_TOKENS, NUM_TOKEN_TYPES
from bioemu2.biomoa.modules.pairmixer import PairMixerSequenceEmbedder
from bioemu2.structure import AllAtomStructure, add_mode_conditioning
from tests.msa.features import add_dummy_msa_features
from tests.structure.conftest import build_test_structure


@pytest.fixture
def test_structure() -> AllAtomStructure:
    """YDPE 4-residue structure with random coordinates and dummy MSA features."""
    s = build_test_structure("YDPE")
    s.coords = torch.randn(s.num_nodes, 3)
    return add_mode_conditioning(add_dummy_msa_features(s), mode="NON-MD")


# Small CPU-only BioMoaCModel.

_TOY_TOKEN_S = 12
_TOY_S_INPUT_DIM = _TOY_TOKEN_S + NUM_TOKEN_TYPES + NUM_MSA_TOKENS + 1 + LEN_POCKET_CONTACT_INFO

# Tiny diffusion trunk with deliberately mismatched dims to probe boundary conditions.
_TOY_DIFFUSION: dict[str, Any] = dict(
    atom_s=20,  # = 4 · 5; divisible by atom_attention_heads (4) and prime 5
    atom_z=4,
    atom_attention_heads=4,
    atoms_per_window_queries=32,
    atoms_per_window_keys=32,
    token_s=_TOY_TOKEN_S,
    token_z=11,
    s_input_dim=_TOY_S_INPUT_DIM,
    token_attention_heads=3,
    a_s=21,
    a_z=11,
    diffusion_transformer_depth=3,
    dim_fourier=7,
    conditioning_transition_layers=1,
    relpos_cfg={},
    use_lns=True,
    num_bins_distogram=5,
    max_atoms=None,
    compile_score=False,
)

# Tiny PairMixer sequence-embedder trunk.
_TOY_PAIRMIXER: dict[str, Any] = dict(
    token_s=_TOY_DIFFUSION["token_s"],  # 12
    token_z=_TOY_DIFFUSION["token_z"],  # 11
    s_input_dim=_TOY_S_INPUT_DIM,
    msa_s=5,  # prime
    msa_blocks=1,
    num_blocks=1,
    trimul_variant="full",  # full works on CPU
    opm_variant="vanilla",
)


def build_toy_biomoa_model(**overrides: Any) -> BioMoaCModel:
    """Create a small CPU model with active residual projections.

    The numerical tests need nonzero outputs from these paths.
    """
    torch.manual_seed(0)
    sequence_embedder = PairMixerSequenceEmbedder(**_TOY_PAIRMIXER)
    edm_schedule = EDMSchedule(sigma_min=0.0004, sigma_max=160.0, sigma_data=16.0)
    model = BioMoaCModel(
        sequence_embedder=sequence_embedder,
        edm_schedule=edm_schedule,
        **{**_TOY_DIFFUSION, **overrides},
    ).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                torch.nn.init.normal_(p, std=0.02)
    return model


@pytest.fixture
def toy_biomoa_model() -> BioMoaCModel:
    """Fresh tiny ``BioMoaCModel`` on CPU."""
    return build_toy_biomoa_model()


@pytest.fixture
def toy_biomoa_ito_model() -> BioMoaCModel:
    """Tiny ``BioMoaCModel`` with MD-sequential conditioning enabled."""
    return build_toy_biomoa_model(
        use_dynamics_conditioning=True,
        downscale_condition_contrib=True,
        dt_embed_num_timescales=4,
    )
