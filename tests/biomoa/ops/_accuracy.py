from dataclasses import dataclass, field
from typing import Any

import torch

Tolerance = tuple[float, float]


@dataclass(frozen=True)
class VariantAccuracyCase:
    name: str
    variant: str
    reference_variant: str
    dtype: torch.dtype = torch.float32
    reference_dtype: torch.dtype = torch.float32
    autocast_dtype: torch.dtype | None = None
    matmul_precision: str = "highest"
    options: dict[str, Any] = field(default_factory=dict)
    forward_tolerance: Tolerance = (0.0, 0.0)
    input_gradient_tolerance: Tolerance = (0.0, 0.0)
    parameter_gradient_tolerance: Tolerance = (0.0, 0.0)
    inference_only: bool = False
