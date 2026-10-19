"""Time embeddings used by BioMoa dynamics conditioning."""

from math import pi

import torch
import torch.nn as nn


class MultiExponentialTimeEmbedding(nn.Module):
    """Multi-exponential decay embedding for lag times."""

    def __init__(self, dim: int, t_min: float = 1.0, t_max: float = 1e9):
        super().__init__()
        assert dim > 0
        assert 0 < t_min < t_max
        taus = torch.logspace(
            start=torch.log10(torch.tensor(t_min)).item(),
            end=torch.log10(torch.tensor(t_max)).item(),
            steps=dim,
        )
        self.register_buffer("taus", taus)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Return embeddings with shape ``(B, dim)``."""
        times = times.reshape(-1, 1)
        return torch.exp(-times / self.taus)


class DampedHarmonicTimeEmbedding(nn.Module):
    """Embed lag times with exponentially damped harmonic features."""

    QUALITY_FACTORS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

    def __init__(
        self,
        num_timescales: int,
        t_min: float = 1.0,
        t_max: float = 1e9,
        quality_factors: tuple[float, ...] | None = None,
    ) -> None:
        """Initialize the embedding.

        Args:
          num_timescales: Number of log-spaced timescales.
          t_min: Shortest timescale, in the same units as the input.
          t_max: Longest timescale, in the same units as the input.
          quality_factors: Oscillation frequencies relative to each timescale.
        """
        super().__init__()
        assert num_timescales > 0
        assert 0 < t_min < t_max
        taus = torch.logspace(
            start=torch.log10(torch.tensor(t_min)).item(),
            end=torch.log10(torch.tensor(t_max)).item(),
            steps=num_timescales,
        )
        factors = self.QUALITY_FACTORS if quality_factors is None else quality_factors
        self.register_buffer("taus", taus)
        self.register_buffer("quality_factors", torch.tensor(factors, dtype=torch.float32))

    @property
    def output_dim(self) -> int:
        """Return the number of embedding features."""
        return self.taus.shape[0] * (1 + 2 * self.quality_factors.shape[0])

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Embed lag times from a column tensor with shape ``(B, 1)``."""
        scaled_time = times / self.taus[None, :]  # [B, T]
        envelope = torch.exp(-scaled_time)  # [B, T]

        quality_factors = self.quality_factors.to(dtype=times.dtype)
        phase = 2 * pi * quality_factors[None, :, None] * scaled_time[:, None, :]  # [B, Q, T]
        modulated = envelope[:, None, :] * torch.stack(
            (torch.sin(phase), torch.cos(phase)), dim=0
        )  # [2, B, Q, T]
        sin_features = modulated[0].transpose(1, 2).reshape(times.shape[0], -1)
        cos_features = modulated[1].transpose(1, 2).reshape(times.shape[0], -1)
        return torch.cat([envelope, sin_features, cos_features], dim=-1)
