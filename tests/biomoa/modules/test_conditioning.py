import torch

from bioemu2.biomoa.modules.conditioning import (
    ContactConditioning,
    contact_conditioning_info,
)


def test_contact_conditioning_zeroes_padded_pairs_and_flows_grads() -> None:
    """Padded contact pairs are masked without blocking gradients."""
    torch.manual_seed(0)
    token_z, batch_size, num_tokens = 4, 1, 4

    module = ContactConditioning(token_z=token_z)

    contact_cond = torch.zeros(
        (batch_size, num_tokens, num_tokens, len(contact_conditioning_info)),
        dtype=torch.float32,
    )
    contact_cond[0, 0, 0, contact_conditioning_info["UNSPECIFIED"]] = 1.0
    contact_cond[0, 1, 1, contact_conditioning_info["UNSELECTED"]] = 1.0
    contact_cond[0, 0, 1, contact_conditioning_info["CONTACT"]] = 1.0
    contact_cond[0, 1, 0, contact_conditioning_info["CONTACT"]] = 1.0

    token_pad_mask = torch.tensor([[True, True, True, False]])
    contact_threshold = torch.full((batch_size, num_tokens, num_tokens), 8.0)

    out = module(contact_cond, contact_threshold, token_pad_mask)

    assert torch.any(out[0, :3, :3] != 0.0)
    assert torch.all(out[0, 3] == 0.0)
    assert torch.all(out[0, :, 3] == 0.0)

    out.sum().backward()
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.any(param.grad != 0), f"{name} received an all-zero gradient"
