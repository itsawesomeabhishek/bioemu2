"""Gradient flow tests for contact conditioning wired into ``PairMixerSequenceEmbedder``."""

import pytest
import torch
import torch.nn as nn

from bioemu2.biomoa.modules.conditioning import contact_conditioning_info
from bioemu2.biomoa.modules.encoders import LEN_POCKET_CONTACT_INFO, NUM_MSA_TOKENS, NUM_TOKEN_TYPES
from bioemu2.biomoa.modules.pairmixer import PairMixerSequenceEmbedder
from bioemu2.conditioning.method import method_types_ids

TOKEN_S = 4
TOKEN_Z = 4
S_INPUT_DIM = TOKEN_S + NUM_TOKEN_TYPES + NUM_MSA_TOKENS + 1 + LEN_POCKET_CONTACT_INFO
BATCH_SIZE = 1
NUM_TOKENS = 3


class _IdentityMSA(nn.Module):
    def forward(
        self, s, z, emb, msa, msa_mask, has_deletion, deletion_value, msa_paired, token_pad_mask
    ):
        return s, z


class _IdentityPairMixer(nn.Module):
    def forward(self, s, z, pair_mask):
        return s, z


@pytest.fixture
def embedder() -> PairMixerSequenceEmbedder:
    """Tiny ``PairMixerSequenceEmbedder`` with contact conditioning enabled.

    The MSA and PairMixer stacks are replaced by identity passthroughs so the
    forward path under test reduces to the ``z_init`` composition that adds
    the contact-conditioning contribution.
    """
    torch.manual_seed(0)
    embedder = PairMixerSequenceEmbedder(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        s_input_dim=S_INPUT_DIM,
        msa_blocks=1,
        num_blocks=1,
        trimul_variant="full",
        use_contact_conditioning=True,
    )
    embedder.msa_module = _IdentityMSA()
    embedder.pairmixer_module = _IdentityPairMixer()
    return embedder


@pytest.fixture
def feats() -> dict[str, torch.Tensor]:
    """Minimal feats dict required by the reduced forward path."""
    contact_conditioning = torch.zeros(
        (BATCH_SIZE, NUM_TOKENS, NUM_TOKENS, len(contact_conditioning_info)),
        dtype=torch.float32,
    )
    contact_conditioning[0, 0, 1, contact_conditioning_info["CONTACT"]] = 1.0
    contact_conditioning[0, 1, 0, contact_conditioning_info["CONTACT"]] = 1.0
    contact_conditioning[0, 0, 0, contact_conditioning_info["UNSPECIFIED"]] = 1.0

    NUM_MSA_ROWS = 2  # dummy MSA depth

    return {
        "contact_conditioning": contact_conditioning,
        "contact_threshold": torch.full((BATCH_SIZE, NUM_TOKENS, NUM_TOKENS), 8.0),
        "token_pad_mask": torch.ones((BATCH_SIZE, NUM_TOKENS), dtype=torch.bool),
        "token_bonds": torch.zeros((BATCH_SIZE, NUM_TOKENS, NUM_TOKENS, 1), dtype=torch.float32),
        "residue_index": torch.arange(NUM_TOKENS, dtype=torch.long).unsqueeze(0),
        "asym_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "entity_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "sym_id": torch.zeros((BATCH_SIZE, NUM_TOKENS), dtype=torch.long),
        "method_feature": torch.full(
            (BATCH_SIZE, NUM_TOKENS), method_types_ids["non-md"], dtype=torch.long
        ),
        # Dummy MSA tensors — _IdentityMSA ignores them, but the explicit call
        # convention in PairMixerSequenceEmbedder.forward unpacks them from feats.
        "msa": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS, 1), dtype=torch.float32),
        "msa_mask": torch.ones((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS), dtype=torch.float32),
        "has_deletion": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS), dtype=torch.float32),
        "deletion_value": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS), dtype=torch.float32),
        "msa_paired": torch.zeros((BATCH_SIZE, NUM_MSA_ROWS, NUM_TOKENS), dtype=torch.float32),
    }


def _forward_and_backward(
    embedder: PairMixerSequenceEmbedder, feats: dict[str, torch.Tensor]
) -> None:
    torch.manual_seed(0)
    s_inputs = torch.randn(BATCH_SIZE, NUM_TOKENS, S_INPUT_DIM)
    s, z = embedder(s_inputs, feats, num_steps_recycle=0)
    (s.sum() + z.sum()).backward()


def test_pairmixer_contact_conditioning_receives_gradient(
    embedder: PairMixerSequenceEmbedder, feats: dict[str, torch.Tensor]
) -> None:
    """With ``use_contact_conditioning=True`` the embedder must back-propagate into it.

    Guards against silent rewiring (e.g., dropping the ``z_init += contact_conditioning(...)``
    addition or accidentally detaching it).
    """
    _forward_and_backward(embedder, feats)

    cc_params = list(embedder.contact_conditioning.named_parameters())
    assert cc_params, "ContactConditioning has no parameters — test setup is wrong"

    received_any = False
    for name, param in cc_params:
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"contact_conditioning.{name} has no gradient"
        if torch.any(param.grad != 0):
            received_any = True
    assert received_any, "No ContactConditioning parameter received a non-zero gradient"


def test_pairmixer_disconnected_contact_conditioning_receives_no_gradient(
    embedder: PairMixerSequenceEmbedder, feats: dict[str, torch.Tensor]
) -> None:
    """A detached contact path gives no parameter gradients.

    This check confirms that the positive test depends on contact conditioning.
    """
    embedder.contact_conditioning.forward = (  # type: ignore[method-assign]
        lambda contact_conditioning, contact_threshold, token_pad_mask: torch.zeros(
            BATCH_SIZE, NUM_TOKENS, NUM_TOKENS, TOKEN_Z
        ).detach()
    )

    _forward_and_backward(embedder, feats)

    for name, param in embedder.contact_conditioning.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is None or torch.all(param.grad == 0), (
            f"Disconnected contact_conditioning.{name} unexpectedly received a non-zero gradient"
        )
