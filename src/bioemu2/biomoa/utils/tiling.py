"""Resolve and validate tiling / chunking configurations.

The structure module (``ScoreModel``) and sequence embedder each accept a
``tiling_cfg`` dictionary. The dictionary shape is::

    tiling_cfg:
      tile_size: <int|None>            # optional global default for *_tile_size keys
      chunk_size: <int|None>           # optional global default for *_chunk_size keys
      attn_pair_bias_tile_size: <int|None>
      opm_tile_size: <int|None>
      trimul_tile_size: <int|None>
      transition_chunk_size: <int|None>
      pwa_chunk_heads: <bool>

The set of recognised per-key sizes is module-specific — see
``MODULE_KNOWN_KEYS`` for the exact subset honoured by each module
(e.g. ``trimul_tile_size`` only applies to the pairmixer embedder,
``attn_pair_bias_tile_size`` only to the structure module).

Some tile/chunk sizes are honoured only by specific kernel variants
(e.g. ``opm_tile_size`` requires ``opm_variant=triton_memory_saving``;
``trimul_tile_size`` requires ``trimul_variant=cueq_tiled``). When the
caller passes the active ``variants`` mapping, the resolver auto-switches
the variant to the entry in :data:`PREFERRED_VARIANT_FOR_TILING` and logs
a warning whenever a non-default tile/chunk size is requested on a
variant that does not honour it. ``required_variants(module_name)``
exposes the same compatibility table for callers / tests.

The resolver returns ``(sizes, variants)``: ``sizes`` is the resolved
tile/chunk dict and ``variants`` is the (possibly auto-switched) variant
mapping the caller should pass to its sub-modules.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

GLOBAL_TILE_KEY = "tile_size"
GLOBAL_CHUNK_KEY = "chunk_size"
GLOBAL_KEYS: frozenset[str] = frozenset({GLOBAL_TILE_KEY, GLOBAL_CHUNK_KEY})

STRUCTURE_KEYS: frozenset[str] = frozenset(
    {
        "attn_pair_bias_tile_size",
        "transition_chunk_size",
    }
)

SEQUENCE_KEYS: frozenset[str] = frozenset(
    {
        "opm_tile_size",
        "transition_chunk_size",
        "trimul_tile_size",
        "pwa_chunk_heads",
    }
)

# Maps the public ``module_name`` argument (the consuming class's
# ``type(self).__name__``) to the set of recognised tiling keys.
MODULE_KNOWN_KEYS: dict[str, frozenset[str]] = {
    "BioMoaCModel": STRUCTURE_KEYS,
    "PairMixerSequenceEmbedder": SEQUENCE_KEYS,
}

# Keys whose values are booleans (default False) rather than ints (default None).
_BOOL_KEYS: frozenset[str] = frozenset({"pwa_chunk_heads"})

# Per-key variant compatibility. A tile/chunk knob is honoured only when the
# active variant of the corresponding sub-module is in the listed tuple. Keys
# not present in this table have no variant gating (e.g. ``transition_chunk_size``
# is honoured by every transition variant; ``pwa_chunk_heads`` is a
# kernel-internal flag with no variant choice).
TILING_KEY_VARIANT_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "opm_tile_size": ("triton_memory_saving",),
    "trimul_tile_size": ("cueq_tiled",),
}

# Maps each gated tiling key to the variant kwarg name a caller would supply
# in the ``variants`` dict passed to :func:`resolve_tiling_cfg`.
TILING_KEY_TO_VARIANT_KWARG: dict[str, str] = {
    "opm_tile_size": "opm_variant",
    "trimul_tile_size": "trimul_variant",
}

# Variant the resolver will switch to when the active variant doesn't honour a
# requested tile/chunk size. Picked among the entries of
# ``TILING_KEY_VARIANT_REQUIREMENTS[key]`` to be the fastest tiling-capable
# option in production.
PREFERRED_VARIANT_FOR_TILING: dict[str, str] = {
    "opm_tile_size": "triton_memory_saving",
    "trimul_tile_size": "cueq_tiled",
}


def _default_for(key: str) -> Any:
    return False if key in _BOOL_KEYS else None


def _is_set(key: str, value: Any) -> bool:
    """True when *value* is a non-default request for *key*."""
    if key in _BOOL_KEYS:
        return bool(value)
    return value is not None


def required_variants(module_name: str) -> dict[str, tuple[str, ...]]:
    """Return ``{variant_kwarg_name: supported_variants}`` for every gated
    tile/chunk key exposed by *module_name*.

    Keys that have no variant gating (e.g. ``transition_chunk_size``,
    ``pwa_chunk_heads``) are omitted.

    Raises:
        ValueError: ``module_name`` is unknown.
    """
    try:
        known_keys = MODULE_KNOWN_KEYS[module_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown module_name {module_name!r}; expected one of {sorted(MODULE_KNOWN_KEYS)}."
        ) from exc

    return {
        TILING_KEY_TO_VARIANT_KWARG[key]: TILING_KEY_VARIANT_REQUIREMENTS[key]
        for key in known_keys
        if key in TILING_KEY_VARIANT_REQUIREMENTS
    }


def resolve_tiling_cfg(
    tiling_cfg: dict[str, Any] | None,
    *,
    module_name: str,
    variants: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate a ``tiling_cfg`` dict and return resolved sizes + variants.

    Args:
        tiling_cfg: Raw config (typically loaded from yaml). ``None`` or
            an empty dict is treated as "no tiling anywhere".
        module_name: Name of the consuming class — typically
            ``type(self).__name__``. Must be a key of
            :data:`MODULE_KNOWN_KEYS` (e.g. ``"BioMoaCModel"``,
            ``"PairMixerSequenceEmbedder"``). Selects the matching
            known-keys set.
        variants: Optional mapping from variant kwarg name (e.g.
            ``"opm_variant"``) to the currently selected variant string.
            When provided, the resolver checks every gated tile/chunk key
            (see :data:`TILING_KEY_VARIANT_REQUIREMENTS`) and, when a
            non-default size is requested on a variant that does not honour
            it, **switches** the variant to the matching entry in
            :data:`PREFERRED_VARIANT_FOR_TILING` and emits a
            ``logger.warning``. Variant kwargs not present in the dict are
            skipped. When ``variants`` is ``None``, no checks/switches run
            and the returned variants dict is empty.

    Returns:
        ``(sizes, variants)`` tuple.
        - ``sizes`` has exactly one entry per known key for ``module_name``.
          Boolean keys default to ``False``; integer keys default to
          ``None``. Globals dominate: when ``tile_size`` is set, every
          ``*_tile_size`` key takes that value (overriding any per-key
          entry, including explicit ``null``); ``chunk_size`` likewise for
          ``*_chunk_size``.
        - ``variants`` is a (possibly modified) copy of the input variants
          dict with auto-switched entries applied. Empty when
          ``variants=None`` was passed.

    Raises:
        ValueError: ``module_name`` is unknown, or ``tiling_cfg``
            contains a key that is neither ``tile_size``/``chunk_size``
            nor a member of the selected known-keys set.
    """
    try:
        known_keys = MODULE_KNOWN_KEYS[module_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown module_name {module_name!r}; expected one of {sorted(MODULE_KNOWN_KEYS)}."
        ) from exc

    cfg = dict(tiling_cfg or {})
    global_tile = cfg.pop(GLOBAL_TILE_KEY, None)
    global_chunk = cfg.pop(GLOBAL_CHUNK_KEY, None)

    unknown = set(cfg) - known_keys
    if unknown:
        raise ValueError(
            f"Unknown tiling_cfg keys {sorted(unknown)} for module_name={module_name!r}; "
            f"expected a subset of {sorted(known_keys)} (plus optional {sorted(GLOBAL_KEYS)})."
        )

    resolved: dict[str, Any] = {}
    for key in known_keys:
        # Globals, when present, dominate every per-key value (including
        # explicit ``null``). Per-key values only apply when no matching
        # global is set. To use operation-dependent sizes the caller must
        # set every per-key value explicitly and leave the global unset.
        if key.endswith("_tile_size") and global_tile is not None:
            resolved[key] = global_tile
        elif key.endswith("_chunk_size") and global_chunk is not None:
            resolved[key] = global_chunk
        elif key in cfg:
            resolved[key] = cfg[key]
        else:
            resolved[key] = _default_for(key)

    resolved_variants: dict[str, str] = {}
    if variants is not None:
        resolved_variants = _switch_variants_for_tiling(resolved, module_name, variants)

    return resolved, resolved_variants


def _switch_variants_for_tiling(
    resolved: dict[str, Any],
    module_name: str,
    variants: dict[str, str],
) -> dict[str, str]:
    """Return a copy of *variants* with auto-switches applied.

    For every gated key whose resolved size is a non-default request and
    whose currently active variant does not honour it, replace the variant
    with :data:`PREFERRED_VARIANT_FOR_TILING` ``[key]`` and log a warning.
    Variant kwargs not present in *variants* are skipped (no switch, no
    warning) so callers don't have to plumb every variant kwarg.
    """
    out = dict(variants)
    for key, supported in TILING_KEY_VARIANT_REQUIREMENTS.items():
        if key not in resolved:
            continue
        if not _is_set(key, resolved[key]):
            continue
        variant_kwarg = TILING_KEY_TO_VARIANT_KWARG[key]
        if variant_kwarg not in out:
            continue
        active = out[variant_kwarg]
        # str(active) handles both raw strings and StrEnum members.
        if str(active) in supported:
            continue
        preferred = PREFERRED_VARIANT_FOR_TILING[key]
        logger.warning(
            "Switching %s from %r to %r because tiling_cfg[%s]=%r requires "
            "a tiling-capable variant (supported: %s, module_name=%r).",
            variant_kwarg,
            str(active),
            preferred,
            key,
            resolved[key],
            sorted(supported),
            module_name,
        )
        out[variant_kwarg] = preferred
    return out
