# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Contact and method conditioning modules for the BioMoa pair and single representations."""

import torch
import torch.nn as nn

from bioemu2.biomoa.modules.encoders import FourierEmbedding
from bioemu2.conditioning.method import num_method_types

contact_conditioning_info: dict[str, int] = {
    "UNSPECIFIED": 0,
    "UNSELECTED": 1,
    "POCKET>BINDER": 2,
    "BINDER>POCKET": 3,
    "CONTACT": 4,
}


class ContactConditioning(nn.Module):
    def __init__(self, token_z: int, cutoff_min: float = 4.0, cutoff_max: float = 20.0):
        super().__init__()

        self.fourier_embedding = FourierEmbedding(token_z)
        self.encoder = nn.Linear(token_z + len(contact_conditioning_info) - 1, token_z)
        self.encoding_unspecified = nn.Parameter(torch.zeros(token_z))
        self.encoding_unselected = nn.Parameter(torch.zeros(token_z))
        self.cutoff_min = cutoff_min
        self.cutoff_max = cutoff_max

        assert contact_conditioning_info["UNSPECIFIED"] == 0
        assert contact_conditioning_info["UNSELECTED"] == 1

    def forward(
        self,
        contact_conditioning: torch.Tensor,  # (B, T, T, C)
        contact_threshold: torch.Tensor,  # (B, T, T)
        token_pad_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """Compute contact conditioning bias for the pair representation.

        Args:
            contact_conditioning: Raw contact conditioning features, ``(B, T, T, C)``.
                Channels 0–1 are unspecified/unselected flags; channels 2+ are used.
            contact_threshold: Contact distance threshold, ``(B, T, T)``.
            token_pad_mask: Boolean token validity mask, ``(B, T)``.

        Returns:
            Pair bias, ``(B, T, T, token_z)``.
        """
        cc_features = contact_conditioning[:, :, :, 2:]  # (B, T, T, C-2)
        contact_threshold_normalized = (contact_threshold - self.cutoff_min) / (
            self.cutoff_max - self.cutoff_min
        )
        contact_threshold_fourier = self.fourier_embedding(
            contact_threshold_normalized.flatten()
        ).reshape(contact_threshold_normalized.shape + (-1,))

        cc_features = torch.cat(
            [
                cc_features,
                contact_threshold_normalized.unsqueeze(-1),
                contact_threshold_fourier,
            ],
            dim=-1,
        )
        cc_features = self.encoder(cc_features)

        cc_features = (
            cc_features
            * (
                1 - contact_conditioning[:, :, :, 0:2].sum(dim=-1, keepdim=True)
            )  # Zero out unspecified / unselected contacts.
            + self.encoding_unspecified
            * contact_conditioning[:, :, :, 0:1]  # Encode unspecified contacts.
            + self.encoding_unselected
            * contact_conditioning[:, :, :, 1:2]  # Encode unselected contacts.
        )

        token_mask = token_pad_mask.unsqueeze(-1) & token_pad_mask.unsqueeze(-2)  # (B, T, T)
        cc_features.masked_fill_(~token_mask.unsqueeze(-1), 0.0)
        return cc_features


class MethodConditioning(nn.Module):
    """Method-dependent biases for single and pair representations."""

    def __init__(self, token_s: int, token_z: int, embed_into_pairs: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_method_types, token_s)

        self.embed_into_pairs = embed_into_pairs
        self.pair_embedding: nn.Embedding | None = None
        if embed_into_pairs:
            self.pair_embedding = nn.Embedding(num_method_types, token_z)

    def forward(
        self,
        method_feature: torch.Tensor,  # (B, T) long
        token_pad_mask: torch.Tensor,  # (B, T)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute method conditioning for the single and pair representations.

        Args:
            method_feature: Per-token method type index, ``(B, T)`` long.
            token_pad_mask: Boolean token validity mask, ``(B, T)``.

        Returns:
            Additive single bias, ``(B, T, token_s)``, and optionally pair bias,
            ``(B, T, T, token_z)``.
        """
        token_mask = token_pad_mask.unsqueeze(-1)  # (B, T, 1)
        single_method = self.embedding(method_feature)
        single_method.masked_fill_(~token_mask, 0.0)

        if not self.embed_into_pairs:
            return single_method, None

        assert self.pair_embedding is not None
        pair_method_per_token = self.pair_embedding(method_feature)  # (B, T, token_z)
        pair_method = pair_method_per_token[:, :, None] + pair_method_per_token[:, None, :]
        pair_mask = token_pad_mask.unsqueeze(-1) & token_pad_mask.unsqueeze(-2)  # (B, T, T)
        pair_method.masked_fill_(~pair_mask.unsqueeze(-1), 0.0)

        return single_method, pair_method
