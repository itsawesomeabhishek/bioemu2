"""Precision contexts shared by sampling entrypoints."""

import contextlib
from typing import Any

import torch


def get_autocast_dtype(precision: str) -> torch.dtype | None:
    """Return the autocast dtype for a supported inference precision."""
    match precision:
        case "32-true":
            return None
        case "bf16-mixed":
            return torch.bfloat16
        case _:
            raise ValueError(
                f"Unknown precision setting {precision!r}; expected '32-true' or 'bf16-mixed'."
            )


def autocast_ctx(device_type: str, dtype: torch.dtype | None) -> Any:
    """Enable dtype-specific autocasting on CUDA, or otherwise do nothing."""
    if (device_type != "cuda") or (dtype is None):
        return contextlib.nullcontext()
    return torch.autocast(device_type=device_type, dtype=dtype)


class _Float32MatmulPrecision:
    """Set ``torch.set_float32_matmul_precision`` and restore the previous value on exit."""

    def __init__(self, precision: str) -> None:
        self.precision = precision

    def __enter__(self) -> None:
        self.previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision(self.precision)

    def __exit__(self, *exc_info: object) -> None:
        torch.set_float32_matmul_precision(self.previous)


def matmul_precision_ctx(precision: str | None) -> Any:
    """Set the float32 matmul precision, or do nothing if ``precision`` is ``None``."""
    if precision is None:
        return contextlib.nullcontext()
    return _Float32MatmulPrecision(precision)
