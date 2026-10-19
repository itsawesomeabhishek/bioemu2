"""Tests for ``bioemu2.biomoa.utils.tiling.resolve_tiling_cfg``.

Pure-Python tests — no torch / GPU needed. They exercise validation,
defaults, the global ``tile_size`` / ``chunk_size`` broadcast, and
immutability of the input dict, parametrized over every public
``module_name``.
"""

import pytest

from bioemu2.biomoa.ops.outer_product_mean import OPMVariant
from bioemu2.biomoa.utils.tiling import (
    _BOOL_KEYS,
    MODULE_KNOWN_KEYS,
    PREFERRED_VARIANT_FOR_TILING,
    TILING_KEY_TO_VARIANT_KWARG,
    TILING_KEY_VARIANT_REQUIREMENTS,
    required_variants,
    resolve_tiling_cfg,
)

ALL_MODULES = sorted(MODULE_KNOWN_KEYS)
STRUCTURE = "BioMoaCModel"
PAIRMIXER = "PairMixerSequenceEmbedder"


def test_unknown_module_name_raises():
    with pytest.raises(ValueError, match="Unknown module_name"):
        resolve_tiling_cfg(None, module_name="not_a_module")


@pytest.mark.parametrize("module_name", ALL_MODULES)
@pytest.mark.parametrize("cfg", [None, {}])
def test_defaults_one_entry_per_known_key(module_name, cfg):
    resolved, _ = resolve_tiling_cfg(cfg, module_name=module_name)
    assert set(resolved) == set(MODULE_KNOWN_KEYS[module_name])
    for key, value in resolved.items():
        if key in _BOOL_KEYS:
            assert value is False, f"{key} should default to False"
        else:
            assert value is None, f"{key} should default to None"


def test_per_key_override_wins_over_default():
    resolved, _ = resolve_tiling_cfg(
        {"transition_chunk_size": 64, "attn_pair_bias_tile_size": 128},
        module_name=STRUCTURE,
    )
    assert resolved["transition_chunk_size"] == 64
    assert resolved["attn_pair_bias_tile_size"] == 128


def test_explicit_none_is_overruled_by_global():
    """Globals dominate: an explicit per-key ``None`` is overruled by a
    global ``tile_size``/``chunk_size`` of the matching family."""
    resolved, _ = resolve_tiling_cfg(
        {"tile_size": 256, "attn_pair_bias_tile_size": None},
        module_name=STRUCTURE,
    )
    assert resolved["attn_pair_bias_tile_size"] == 256


def test_global_tile_size_broadcasts_only_to_tile_size_keys():
    resolved, _ = resolve_tiling_cfg({"tile_size": 256}, module_name=PAIRMIXER)
    for key in MODULE_KNOWN_KEYS[PAIRMIXER]:
        if key.endswith("_tile_size"):
            assert resolved[key] == 256, f"{key} should pick up global tile_size"
        elif key in _BOOL_KEYS:
            assert resolved[key] is False, f"{key} should NOT be touched by tile_size"
        else:
            assert resolved[key] is None, f"{key} should NOT be touched by tile_size"


def test_global_chunk_size_broadcasts_only_to_chunk_size_keys():
    resolved, _ = resolve_tiling_cfg({"chunk_size": 32}, module_name=PAIRMIXER)
    for key in MODULE_KNOWN_KEYS[PAIRMIXER]:
        if key.endswith("_chunk_size"):
            assert resolved[key] == 32, f"{key} should pick up global chunk_size"
        elif key in _BOOL_KEYS:
            assert resolved[key] is False, f"{key} should NOT be touched by chunk_size"
        else:
            assert resolved[key] is None, f"{key} should NOT be touched by chunk_size"


def test_globals_independent():
    resolved, _ = resolve_tiling_cfg({"tile_size": 256, "chunk_size": 32}, module_name=PAIRMIXER)
    for key in MODULE_KNOWN_KEYS[PAIRMIXER]:
        if key.endswith("_tile_size"):
            assert resolved[key] == 256
        elif key.endswith("_chunk_size"):
            assert resolved[key] == 32
        elif key in _BOOL_KEYS:
            assert resolved[key] is False


def test_global_tile_overrules_per_key_tile():
    """Global ``tile_size`` overrules per-key ``*_tile_size`` values."""
    resolved, _ = resolve_tiling_cfg(
        {"tile_size": 256, "opm_tile_size": 512},
        module_name=PAIRMIXER,
    )
    assert resolved["opm_tile_size"] == 256
    assert resolved["trimul_tile_size"] == 256


def test_global_chunk_overrules_per_key_chunk():
    """Global ``chunk_size`` overrules per-key ``*_chunk_size`` values."""
    resolved, _ = resolve_tiling_cfg(
        {"chunk_size": 32, "transition_chunk_size": 64},
        module_name=PAIRMIXER,
    )
    assert resolved["transition_chunk_size"] == 32


def test_per_key_applies_only_when_global_unset():
    """Without a matching global, per-key entries take effect."""
    resolved, _ = resolve_tiling_cfg(
        {"opm_tile_size": 512, "transition_chunk_size": 64},
        module_name=PAIRMIXER,
    )
    assert resolved["opm_tile_size"] == 512
    assert resolved["transition_chunk_size"] == 64
    assert resolved["trimul_tile_size"] is None


@pytest.mark.parametrize(
    "module_name",
    [m for m, ks in MODULE_KNOWN_KEYS.items() if "pwa_chunk_heads" in ks],
)
def test_pwa_chunk_heads_not_affected_by_globals(module_name):
    for global_cfg in (
        {"tile_size": 256},
        {"chunk_size": 32},
        {"tile_size": 256, "chunk_size": 32},
    ):
        resolved, _ = resolve_tiling_cfg(global_cfg, module_name=module_name)
        assert resolved["pwa_chunk_heads"] is False, global_cfg


@pytest.mark.parametrize(
    "module_name",
    [m for m, ks in MODULE_KNOWN_KEYS.items() if "pwa_chunk_heads" in ks],
)
def test_pwa_chunk_heads_only_true_when_explicit(module_name):
    assert resolve_tiling_cfg(None, module_name=module_name)[0]["pwa_chunk_heads"] is False
    resolved, _ = resolve_tiling_cfg(
        {"tile_size": 256, "transition_chunk_size": 32},
        module_name=module_name,
    )
    assert resolved["pwa_chunk_heads"] is False
    resolved, _ = resolve_tiling_cfg({"pwa_chunk_heads": True}, module_name=module_name)
    assert resolved["pwa_chunk_heads"] is True


def test_unknown_key_raises_valueerror():
    with pytest.raises(ValueError) as exc_info:
        resolve_tiling_cfg({"made_up_tile_size": 128}, module_name=STRUCTURE)
    msg = str(exc_info.value)
    assert "made_up_tile_size" in msg
    for key in MODULE_KNOWN_KEYS[STRUCTURE]:
        assert key in msg


def test_global_keys_are_never_unknown():
    resolve_tiling_cfg({"tile_size": 128}, module_name=STRUCTURE)
    resolve_tiling_cfg({"chunk_size": 32}, module_name=STRUCTURE)
    resolve_tiling_cfg({"tile_size": 128, "chunk_size": 32}, module_name=STRUCTURE)


def test_unknown_keys_listed_sorted():
    with pytest.raises(ValueError) as exc_info:
        resolve_tiling_cfg(
            {"zebra_tile_size": 1, "apple_tile_size": 1},
            module_name=STRUCTURE,
        )
    msg = str(exc_info.value)
    assert msg.find("apple_tile_size") < msg.find("zebra_tile_size")


def test_per_module_key_validation():
    # `trimul_tile_size` is only valid for the pairmixer module.
    resolve_tiling_cfg({"trimul_tile_size": 128}, module_name=PAIRMIXER)
    with pytest.raises(ValueError, match="trimul_tile_size"):
        resolve_tiling_cfg({"trimul_tile_size": 128}, module_name=STRUCTURE)


def test_input_dict_not_mutated():
    cfg = {"tile_size": 256, "opm_tile_size": 64}
    snapshot = dict(cfg)
    resolve_tiling_cfg(cfg, module_name=PAIRMIXER)
    assert cfg == snapshot


def test_resolved_keys_exactly_known_keys():
    resolved, _ = resolve_tiling_cfg({"tile_size": 128, "chunk_size": 32}, module_name=PAIRMIXER)
    assert set(resolved) == set(MODULE_KNOWN_KEYS[PAIRMIXER])
    assert "tile_size" not in resolved
    assert "chunk_size" not in resolved


_LOGGER = "bioemu2.biomoa.utils.tiling"


def test_required_variants_unknown_module_raises():
    with pytest.raises(ValueError, match="Unknown module_name"):
        required_variants("not_a_module")


def test_required_variants_per_module_shape():
    # Only gated keys appear; ungated keys (transition_chunk_size,
    # pwa_chunk_heads) and keys not exposed by the module are absent.
    assert required_variants(STRUCTURE) == {}
    assert required_variants("BioMoaCModel") == {}
    assert required_variants(PAIRMIXER) == {
        "opm_variant": ("triton_memory_saving",),
        "trimul_variant": ("cueq_tiled",),
    }


def test_required_variants_table_keys_are_consistent():
    assert set(TILING_KEY_VARIANT_REQUIREMENTS) == set(TILING_KEY_TO_VARIANT_KWARG)


def test_compatible_variant_emits_no_warning_and_passes_through(caplog):
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"opm_tile_size": 64, "trimul_tile_size": 256},
            module_name=PAIRMIXER,
            variants={
                "opm_variant": "triton_memory_saving",
                "trimul_variant": "cueq_tiled",
            },
        )
    assert not any(_LOGGER in r.name for r in caplog.records)
    # Compatible variants are returned unchanged.
    assert variants == {
        "opm_variant": "triton_memory_saving",
        "trimul_variant": "cueq_tiled",
    }


def test_incompatible_variant_with_set_size_switches_and_warns(caplog):
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"opm_tile_size": 64},
            module_name=PAIRMIXER,
            variants={"opm_variant": "vanilla"},
        )
    assert variants["opm_variant"] == "triton_memory_saving"  # switched
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER]
    assert len(msgs) == 1
    msg = msgs[0]
    assert "Switching" in msg
    assert "opm_variant" in msg
    assert "vanilla" in msg
    assert "triton_memory_saving" in msg
    assert "opm_tile_size" in msg


def test_default_size_on_incompatible_variant_no_switch_no_warning(caplog):
    # opm_tile_size left at default (None) -> no switch, no warning even
    # though opm_variant=vanilla doesn't honour it.
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {},
            module_name=PAIRMIXER,
            variants={"opm_variant": "vanilla"},
        )
    assert not any(_LOGGER in r.name for r in caplog.records)
    assert variants["opm_variant"] == "vanilla"


def test_global_tile_size_switches_each_gated_key_variant(caplog):
    # global tile_size dominates per-key entries; if the active variant
    # doesn't honour it, the gated *_tile_size variant should be switched.
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"tile_size": 128},
            module_name=PAIRMIXER,
            variants={"opm_variant": "vanilla", "trimul_variant": "full"},
        )
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER]
    assert len(msgs) == 2
    switched_kwarg_names = {m.split()[1] for m in msgs}
    assert switched_kwarg_names == {"opm_variant", "trimul_variant"}
    assert variants["opm_variant"] == "triton_memory_saving"
    assert variants["trimul_variant"] == "cueq_tiled"


def test_warning_skipped_when_variant_kwarg_absent(caplog):
    # Variants not present in the dict are silently skipped (caller may
    # not have plumbed every variant kwarg).
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"opm_tile_size": 64},
            module_name=PAIRMIXER,
            variants={"trimul_variant": "cueq_tiled"},  # opm_variant absent
        )
    assert not any(_LOGGER in r.name for r in caplog.records)
    # opm_variant absent -> stays absent in returned dict.
    assert "opm_variant" not in variants


def test_switch_handles_str_enum_variants(caplog):
    # Variant enums are str subclasses; ensure str(active) comparison works.
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"opm_tile_size": 64},
            module_name=PAIRMIXER,
            variants={"opm_variant": OPMVariant.VANILLA},
        )
    msgs = [r.getMessage() for r in caplog.records if r.name == _LOGGER]
    assert len(msgs) == 1
    assert "opm_tile_size" in msgs[0]
    assert variants["opm_variant"] == "triton_memory_saving"


def test_pwa_chunk_heads_never_warns_or_switches(caplog):
    # pwa_chunk_heads has no variant gating.
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"pwa_chunk_heads": True},
            module_name=PAIRMIXER,
            variants={"opm_variant": "vanilla"},
        )
    assert not any(_LOGGER in r.name for r in caplog.records)
    assert variants["opm_variant"] == "vanilla"


def test_transition_chunk_size_never_warns_or_switches(caplog):
    # transition_chunk_size is honoured by all transition variants.
    with caplog.at_level("WARNING", logger=_LOGGER):
        _, variants = resolve_tiling_cfg(
            {"transition_chunk_size": 8},
            module_name=STRUCTURE,
            variants={},
        )
    assert not any(_LOGGER in r.name for r in caplog.records)
    assert variants == {}


@pytest.mark.parametrize(
    ("module_name", "size_key", "variant_kwarg", "incompatible_variant"),
    [
        (PAIRMIXER, "opm_tile_size", "opm_variant", "vanilla"),
        (PAIRMIXER, "trimul_tile_size", "trimul_variant", "full"),
    ],
)
def test_auto_switch_uses_preferred_variant(
    module_name, size_key, variant_kwarg, incompatible_variant
):
    """For each gated key, an incompatible active variant gets switched
    to ``PREFERRED_VARIANT_FOR_TILING[size_key]``."""
    _, variants = resolve_tiling_cfg(
        {size_key: 32},
        module_name=module_name,
        variants={variant_kwarg: incompatible_variant},
    )
    assert variants[variant_kwarg] == PREFERRED_VARIANT_FOR_TILING[size_key]


def test_preferred_variant_table_consistent_with_requirements():
    assert set(PREFERRED_VARIANT_FOR_TILING) == set(TILING_KEY_VARIANT_REQUIREMENTS)
    for key, preferred in PREFERRED_VARIANT_FOR_TILING.items():
        assert preferred in TILING_KEY_VARIANT_REQUIREMENTS[key], (
            f"{preferred!r} not in supported variants for {key!r}"
        )


def test_input_variants_dict_not_mutated():
    # The resolver must copy the variants dict before applying switches.
    variants = {"opm_variant": "vanilla"}
    snapshot = dict(variants)
    resolve_tiling_cfg(
        {"opm_tile_size": 64},
        module_name=PAIRMIXER,
        variants=variants,
    )
    assert variants == snapshot
