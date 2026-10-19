"""Resolve one matmul-precision knob from a nested ``precision_cfg``.

A *knob* is a per-kernel matmul precision: ``"ieee"``, ``"tf32"`` or ``"tf32x3"``
(cueq TriMul also accepts ``None``). The kernels validate the string themselves;
this module only routes it.

We assume the yaml always lists every knob a model needs, at least as ``null``.
Given that:

- A knob set to a real value uses that value.
- A knob left ``null`` uses the active variant's default from
  ``DEFAULT_PRECISION_KEYS``, which ``strategy.make`` forwards to the kernel.
- A value set on a variant that lacks the knob (see
  ``VARIANTS_WITH_PRECISION_KNOB``) is returned as-is but warned about — we never
  switch the variant, because precision changes numerics.

``DEFAULT_PRECISION_KEYS`` mirrors each variant class's ``matmul_precision``
``__init__`` default and is drift-tested against it.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Variants that have each precision knob. Any other variant lacks it.
VARIANTS_WITH_PRECISION_KNOB: dict[str, set[str]] = {
    "opm_matmul_precision": {"triton_memory_saving"},
    "trimul_matmul_precision": {"cueq", "cueq_tiled"},
}

# Default precision per variant when the knob is left unset.
DEFAULT_PRECISION_KEYS: dict[str, dict[str, str | None]] = {
    "opm_matmul_precision": {"triton_memory_saving": "ieee"},
    "trimul_matmul_precision": {"cueq": "ieee", "cueq_tiled": "ieee"},
}

# Structural invariant: the two tables must agree on which variants honour each knob.
assert {k: set(v) for k, v in DEFAULT_PRECISION_KEYS.items()} == VARIANTS_WITH_PRECISION_KNOB, (
    "DEFAULT_PRECISION_KEYS and VARIANTS_WITH_PRECISION_KNOB are out of sync"
)


def resolve_precision_cfg(
    precision_cfg: dict[str, Any] | None,
    *,
    site: str,
    variant: str,
) -> str | None:
    """Resolve the knob named by ``site`` for the active ``variant``.

    Args:
        precision_cfg: The (possibly scoped) precision dict the caller holds, or
            ``None``.
        site: Dotted path to the knob within ``precision_cfg``, e.g.
            ``"msa_module.opm_matmul_precision"``.
        variant: Active variant for the knob's kernel family (enum or string).

    Example:
        >>> resolve_precision_cfg(
        ...     {"msa_module": {"opm_matmul_precision": "tf32"}},
        ...     site="msa_module.opm_matmul_precision",
        ...     variant="triton_memory_saving",
        ... )
        'tf32'
    """
    *path, knob = site.split(".")
    node: Any = precision_cfg or {}
    for segment in path:
        node = node.get(segment) or {}
    value = node.get(knob)
    variant = variant.value if hasattr(variant, "value") else variant

    if value is None:
        return DEFAULT_PRECISION_KEYS[knob].get(variant)
    if variant not in VARIANTS_WITH_PRECISION_KNOB[knob]:
        logger.warning(
            "precision %s=%r at %r ignored: variant %r does not honour it (honoured by %s).",
            knob,
            value,
            site,
            variant,
            sorted(VARIANTS_WITH_PRECISION_KNOB[knob]),
        )
    return value
