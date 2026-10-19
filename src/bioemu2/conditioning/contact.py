# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Contact-conditioning feature helpers for AllAtomStructure samples."""

import torch

# NOTE: At the moment, only UNSPECIFIED, UNSELECTED, and CONTACT are used.
contact_conditioning_info: dict[str, int] = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}

_NUM_CONDITIONING_CLASSES = len(contact_conditioning_info)


def make_unspecified_contact_conditioning(
    max_num_tokens: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """One-hot ``contact_conditioning`` matrix with all pairs marked ``UNSPECIFIED``.

    Represents the "no contact conditioning" case, where conditioning is inactive for
    every token pair.

    Args:
        max_num_tokens: Token dimension ``T`` of the returned matrix.
        device: Device on which to allocate the tensor.

    Returns:
        ``(T, T, len(contact_conditioning_info))`` ``torch.long`` one-hot tensor.
    """
    cc = torch.full(
        (max_num_tokens, max_num_tokens),
        contact_conditioning_info["UNSPECIFIED"],
        dtype=torch.long,
        device=device,
    )
    return torch.nn.functional.one_hot(cc, num_classes=_NUM_CONDITIONING_CLASSES).to(torch.long)


def make_unspecified_contact_threshold(
    max_num_tokens: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``contact_threshold`` matrix (Å) for the no-contacts case (all zeros).

    Represents the "no contact conditioning" case, where conditioning is inactive for
    every token pair.

    Args:
        max_num_tokens: Token dimension ``T`` of the returned matrix.
        device: Device on which to allocate the tensor.

    Returns:
        ``(T, T)`` ``torch.float32`` tensor of zeros.
    """
    return torch.zeros((max_num_tokens, max_num_tokens), dtype=torch.float32, device=device)


def count_revealed_contacts(contact_conditioning: torch.Tensor) -> torch.Tensor:
    """Per-sample count of revealed (CONTACT) token pairs in a one-hot batch.

    Args:
        contact_conditioning: ``(B, T, T, len(contact_conditioning_info))`` one-hot
            tensor as produced by ``batch_collate``.

    Returns:
        ``(B,)`` ``torch.long`` tensor with the number of *unordered* revealed
        contact pairs per sample. Padded positions contribute zero because
        ``batch_collate`` pads with literal zeros, so the CONTACT channel is
        false there.
    """
    assert contact_conditioning.ndim == 4, (
        f"expected (B, T, T, C); got shape {tuple(contact_conditioning.shape)}"
    )
    contact_channel = contact_conditioning_info["CONTACT"]
    contact_mask = contact_conditioning[..., contact_channel].bool()  # (B, T, T)
    upper = torch.triu(contact_mask, diagonal=1)
    return upper.sum(dim=(1, 2)).to(torch.long)
