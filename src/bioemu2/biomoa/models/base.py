from dataclasses import dataclass

import torch


@dataclass
class ScoreModelOutput:
    """Output container shared by BioMoa score models."""

    score: torch.Tensor  # unitless, Karras EDM
    distogram: torch.Tensor | None  # per-Cα-pair scores over distance bins; None if no head
    denoised_coords: torch.Tensor | None  # angstroms; None from _forward, set by .forward
