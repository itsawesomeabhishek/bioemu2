"""Forward-hook diagnostics for context-parallel models."""

import logging

import torch.nn as nn

from bioemu2.context_parallel.utils.session import TRACE_LEVEL

logger = logging.getLogger(__name__)


# Class-name substrings for the CP-collective-heavy stack; forward hooks on these
# emit enter/exit at DEBUG to localise hangs without touching model code.
_TRACE_MODULE_PATTERNS = (
    "Triangle",
    "OuterProduct",
    "PairMixer",
    "Attention",
    "Distributed",
    "Fold",
    "Diffusion",
)


def attach_trace_hooks(model: nn.Module) -> None:
    """At TRACE, register fwd pre/post hooks on CP-heavy submodules to trace execution.

    Filters by class-name substring so we don't drown in Linear/LayerNorm; hooks are
    pure observation and leave model code untouched.
    """
    if not logger.isEnabledFor(TRACE_LEVEL):
        return
    count = 0
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if not any(p in cls for p in _TRACE_MODULE_PATTERNS):
            continue
        mod.register_forward_pre_hook(
            lambda _m, _inp, _n=name, _c=cls: logger.debug("enter %s (%s)", _n, _c)
        )
        mod.register_forward_hook(
            lambda _m, _inp, _out, _n=name, _c=cls: logger.debug("exit  %s (%s)", _n, _c)
        )
        count += 1
    logger.debug("trace hooks attached to %d modules", count)
