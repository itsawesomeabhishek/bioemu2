import pytest
import torch
import torch.nn as nn

from bioemu2.biomoa.modules.pairmixer import PairMixerSequenceEmbedder

TOKEN_S = 4
TOKEN_Z = 4
S_INPUT_DIM = 5
BATCH_SIZE = 1
NUM_TOKENS = 3
NUM_MSA_ROWS = 2


class _IdentityMSA(nn.Module):
    def forward(
        self,
        s,
        z,
        emb,
        msa,
        msa_mask,
        has_deletion,
        deletion_value,
        msa_paired,
        token_pad_mask,
    ):
        return s, z


class _IdentityPairMixer(nn.Module):
    def forward(self, s, z, pair_mask):
        return s, z


def _make_embedder() -> PairMixerSequenceEmbedder:
    embedder = PairMixerSequenceEmbedder(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        s_input_dim=S_INPUT_DIM,
        msa_s=4,
        msa_blocks=1,
        num_blocks=1,
        trimul_variant="full",
        use_contact_conditioning=False,
        use_method_conditioning=True,
        embed_method_into_pairs=True,
    )
    embedder.msa_module = _IdentityMSA()
    embedder.pairmixer_module = _IdentityPairMixer()
    with torch.no_grad():
        for parameter in embedder.parameters():
            parameter.zero_()
        embedder.method_conditioning.embedding.weight[1].fill_(1.0)
        embedder.method_conditioning.embedding.weight[2].fill_(2.0)
        assert embedder.method_conditioning.pair_embedding is not None
        embedder.method_conditioning.pair_embedding.weight[1].fill_(10.0)
        embedder.method_conditioning.pair_embedding.weight[2].fill_(20.0)
    return embedder


def _make_feats() -> dict[str, torch.Tensor]:
    return {
        "method_feature": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "token_pad_mask": torch.tensor([[True, True, False]]),
        "token_bonds": torch.zeros((BATCH_SIZE, NUM_TOKENS, NUM_TOKENS, 1)),
        "residue_index": torch.arange(NUM_TOKENS, dtype=torch.long).unsqueeze(0),
        "asym_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "entity_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "sym_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "msa": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS, 1)),
        "msa_mask": torch.ones((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS)),
        "has_deletion": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS)),
        "deletion_value": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS)),
        "msa_paired": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS)),
    }


def test_pairmixer_injects_method_conditioning_into_singles_and_pairs() -> None:
    embedder = _make_embedder()
    feats = _make_feats()
    expected_s, expected_z = embedder.method_conditioning(
        method_feature=feats["method_feature"],
        token_pad_mask=feats["token_pad_mask"],
    )

    s_inputs = torch.zeros((BATCH_SIZE, NUM_TOKENS, S_INPUT_DIM))
    s, z = embedder(s_inputs, feats)

    torch.testing.assert_close(s, expected_s)
    torch.testing.assert_close(z, expected_z)

    (s.sum() + z.sum()).backward()
    for name, parameter in embedder.method_conditioning.named_parameters():
        assert parameter.grad is not None, f"method_conditioning.{name} has no gradient"
        assert torch.any(parameter.grad != 0), (
            f"method_conditioning.{name} received only zero gradients"
        )


def test_pair_embedding_requires_method_conditioning() -> None:
    with pytest.raises(
        ValueError,
        match="embed_method_into_pairs requires use_method_conditioning=True",
    ):
        PairMixerSequenceEmbedder(
            token_s=TOKEN_S,
            token_z=TOKEN_Z,
            s_input_dim=S_INPUT_DIM,
            use_method_conditioning=False,
            embed_method_into_pairs=True,
        )
