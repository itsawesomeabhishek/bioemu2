"""Test-only MSA feature helpers."""

import torch

from bioemu2.msa.features import NUM_MSA_TOKENS, process_msa_features
from bioemu2.structure import AllAtomStructure


def add_dummy_msa_features(
    data: AllAtomStructure,
    num_features_msa: int = 249,
    num_tokens_msa: int = NUM_MSA_TOKENS,
) -> AllAtomStructure:
    """Zero-valued MSA features for testing without network access."""
    num_tokens = data.num_tokens
    return data.replace(
        **process_msa_features(
            msa=torch.zeros(num_tokens, num_features_msa, dtype=torch.long),
            deletion=torch.zeros(num_tokens, num_features_msa),
            paired=torch.zeros(num_tokens, num_features_msa),
            num_tokens_msa=num_tokens_msa,
        )
    )
