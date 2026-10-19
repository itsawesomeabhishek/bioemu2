import logging

import pytest

from bioemu2.biomoa.utils.precision import (
    resolve_precision_cfg,
)


@pytest.mark.parametrize(
    "cfg, site, variant, expected, warns",
    [
        # Explicit override on a honouring variant: returned as-is, no warning.
        pytest.param(
            {"msa_module": {"opm_matmul_precision": "ieee"}},
            "msa_module.opm_matmul_precision",
            "triton_memory_saving",
            "ieee",
            False,
            id="explicit-override-honoured",
        ),
        # `None` knob → variant default substituted (cueq_tiled → "ieee").
        pytest.param(
            {"msa_module": {"trimul_matmul_precision": None}},
            "msa_module.trimul_matmul_precision",
            "cueq_tiled",
            "ieee",
            False,
            id="none-falls-back-to-variant-default",
        ),
        # Whole cfg missing → variant default (cueq → "ieee").
        pytest.param(
            None,
            "pairformer_module.trimul_matmul_precision",
            "cueq",
            "ieee",
            False,
            id="missing-cfg-falls-back-to-variant-default",
        ),
        # Value set on a variant that lacks the knob: passed through, warns.
        pytest.param(
            {"msa_module": {"opm_matmul_precision": "tf32"}},
            "msa_module.opm_matmul_precision",
            "vanilla",
            "tf32",
            True,
            id="mismatch-passes-through-and-warns",
        ),
        # No value + variant without the knob: default lookup returns None silently
        # (substitution path bypasses the warn branch).
        pytest.param(
            None,
            "msa_module.opm_matmul_precision",
            "vanilla",
            None,
            False,
            id="no-value-on-non-honouring-variant-silent",
        ),
        # Recomposition: a sibling site's knob (opm) is present in the dict but
        # never read at the trimul site — must not raise, only the resolved key
        # matters.
        pytest.param(
            {
                "pairformer_module": {
                    "opm_matmul_precision": "ieee",
                    "trimul_matmul_precision": None,
                }
            },
            "pairformer_module.trimul_matmul_precision",
            "cueq",
            "ieee",
            False,
            id="unowned-sibling-key-ignored",
        ),
    ],
)
def test_resolve_precision_cfg(cfg, site, variant, expected, warns, caplog):
    with caplog.at_level(logging.WARNING):
        result = resolve_precision_cfg(cfg, site=site, variant=variant)
    assert result == expected
    saw_warn = any("does not honour it" in r.message for r in caplog.records)
    assert saw_warn == warns


def test_unknown_knob_in_site_raises():
    """A typo'd knob in the developer-supplied ``site`` literal must fail loudly."""
    with pytest.raises(KeyError):
        resolve_precision_cfg(
            None,
            site="msa_module.opm_matmul_precison",  # typo
            variant="vanilla",
        )
