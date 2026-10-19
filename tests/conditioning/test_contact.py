"""Tests for `bioemu2.conditioning.contact`."""

import torch

from bioemu2.conditioning.contact import (
    contact_conditioning_info,
    count_revealed_contacts,
)

CONTACT = contact_conditioning_info["CONTACT"]
UNSPECIFIED = contact_conditioning_info["UNSPECIFIED"]


def test_count_revealed_contacts_counts_upper_triangle_only() -> None:
    """Counts each unordered (i,j) pair once, ignoring padding (UNSPECIFIED)."""
    num_classes = len(contact_conditioning_info)

    # Two samples, T=4. Sample 0 has 2 contacts ((0,1) and (2,3)); sample 1 has none.
    cc = torch.zeros((2, 4, 4, num_classes), dtype=torch.long)
    cc[..., UNSPECIFIED] = 1  # baseline
    for s, i, j in [(0, 0, 1), (0, 2, 3)]:
        cc[s, i, j, UNSPECIFIED] = 0
        cc[s, j, i, UNSPECIFIED] = 0
        cc[s, i, j, CONTACT] = 1
        cc[s, j, i, CONTACT] = 1

    counts = count_revealed_contacts(cc)
    assert counts.shape == (2,)
    assert counts.dtype == torch.long
    assert counts.tolist() == [2, 0]
