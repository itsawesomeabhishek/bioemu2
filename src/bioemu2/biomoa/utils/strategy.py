"""Centralized factory for variant module instantiation.

The ``make()`` function is the single entry point for creating variant
module instances from a registry dict.
"""

import importlib
import inspect

import torch.nn as nn


def make(
    registry: dict, variant: str, *, matmul_precision: str | None = None, **kwargs
) -> nn.Module:
    """Instantiate a variant from a registry dict.

    Args:
        registry: Maps enum → (module_path, class_name).
        variant: Variant string or enum value.
        matmul_precision: Optional kernel matmul precision (``"ieee"`` /
            ``"tf32"`` / ``"tf32x3"``). Forwarded to the resolved class
            iff its ``__init__`` accepts a ``matmul_precision`` kwarg —
            variants without a precision knob (vanilla / transition)
            silently ignore it, so callers can pass a single
            family-specific ``matmul_precision`` uniformly across all
            variants in that family.
        **kwargs: Forwarded to the variant class constructor.
    """
    for key in registry:
        if (key == variant) or (getattr(key, "value", None) == variant):
            entry = registry[key]
            if len(entry) == 3:
                mod_path, starting_cls, ending_cls = entry
                starting = kwargs.pop("starting", True)
                cls_name = starting_cls if starting else ending_cls
            else:
                mod_path, cls_name = entry
            cls = getattr(importlib.import_module(mod_path), cls_name)
            # matmul_precision is None only on direct/test calls; under Hydra the
            # precision_config defaults always supply a value, so this guard is
            # rarely the active route in production.
            if (
                matmul_precision is not None
                and "matmul_precision" in inspect.signature(cls).parameters
            ):
                kwargs["matmul_precision"] = matmul_precision
            return cls(**kwargs)
    available = [k.value for k in registry]
    raise ValueError(f"Unknown variant: {variant!r}, available: {available}")
