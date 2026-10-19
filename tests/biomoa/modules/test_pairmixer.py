import torch

from bioemu2.biomoa.modules.pairmixer import PairMixerLayer, PairMixerMSALayer

TOKEN_S = 384
FEATURE_DIM = 64
# cuEquivariance falls back to PyTorch for sequences of at most 100 tokens
# (CUEQ_TRIMUL_FALLBACK_THRESHOLD), so the kernel comparison needs more tokens.
NUM_TOKENS = 128
MSA_S = 16


class TestPairMixerLayer:
    def test_full_and_cueq_have_same_state_dict_keys(self) -> None:
        torch.manual_seed(42)
        full = PairMixerLayer(token_s=TOKEN_S, token_z=FEATURE_DIM, trimul_variant="full")
        cueq = PairMixerLayer(token_s=TOKEN_S, token_z=FEATURE_DIM, trimul_variant="cueq")
        assert full.state_dict().keys() == cueq.state_dict().keys()

    def test_cueq_matches_full(
        self,
        gpu_device: torch.device,
    ) -> None:
        torch.manual_seed(42)
        s = torch.randn(1, NUM_TOKENS, TOKEN_S)
        z = torch.randn(1, NUM_TOKENS, NUM_TOKENS, FEATURE_DIM)
        token_mask = torch.ones(1, NUM_TOKENS, dtype=torch.bool)
        token_mask[:, -2:] = False
        mask = token_mask.unsqueeze(-1) & token_mask.unsqueeze(-2)
        full = PairMixerLayer(token_s=TOKEN_S, token_z=FEATURE_DIM, trimul_variant="full").eval()
        cueq = PairMixerLayer(token_s=TOKEN_S, token_z=FEATURE_DIM, trimul_variant="cueq").eval()
        with torch.no_grad():
            for parameter in full.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=0.02)
        cueq.load_state_dict(full.state_dict())

        s, z, mask = s.to(gpu_device), z.to(gpu_device), mask.to(gpu_device)
        full, cueq = full.to(gpu_device), cueq.to(gpu_device)

        with torch.no_grad():
            _, z_full = full(s, z, mask)
            _, z_cueq = cueq(s, z, mask)

        # CuEQ and einsum use different accumulation orders; max diff over 100 seeds is ~1.4e-6.
        diff = (z_full - z_cueq).abs().max().item()
        assert diff < 1e-5, f"cueq vs full max diff {diff}"


def test_msa_mask_controls_pair_update() -> None:
    torch.manual_seed(0)
    layer = PairMixerMSALayer(
        token_s=TOKEN_S,
        msa_s=MSA_S,
        token_z=FEATURE_DIM,
        msa_dropout=0.0,
        z_dropout=0.0,
        trimul_variant="full",
    ).eval()
    with torch.no_grad():
        layer.outer_product_mean.proj_o.weight.normal_(std=0.02)

    batch_size, num_msa_rows, num_tokens = 1, 2, 4
    s = torch.randn(batch_size, num_tokens, TOKEN_S)
    z = torch.randn(batch_size, num_tokens, num_tokens, FEATURE_DIM)
    m = torch.randn(batch_size, num_msa_rows, num_tokens, MSA_S)
    token_mask = torch.ones(batch_size, num_tokens, num_tokens, dtype=torch.bool)
    msa_mask = torch.tensor([[[True] * num_tokens, [False] * num_tokens]])

    with torch.no_grad():
        _, z_base, _ = layer(s, z, m, token_mask, msa_mask)

        masked_row_changed = m.clone()
        masked_row_changed[:, 1] += 10.0
        _, z_masked, _ = layer(s, z, masked_row_changed, token_mask, msa_mask)

        valid_row_changed = m.clone()
        valid_row_changed[:, 0] += 10.0
        _, z_valid, _ = layer(s, z, valid_row_changed, token_mask, msa_mask)

    torch.testing.assert_close(z_masked, z_base)
    assert not torch.allclose(z_valid, z_base)
