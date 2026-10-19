"""Tests for conditioned transformer residual calculations and gradients."""

import pytest
import torch
from torch import nn

from bioemu2.biomoa.modules.transformer import (
    AttentionLayer,
    ConditionedSlidingWindowTransformerLayer,
    ConditionedTransformerLayer,
    SlidingWindowAttentionLayer,
    ada_ln,
    ada_scale,
)

from .conftest import (
    ATOM_S,
    ATOM_Z,
    ATOMS_PER_WINDOW_KEYS,
    ATOMS_PER_WINDOW_QUERIES,
)

# Residual reconstruction changes operation order, so allow small absolute round-off.
TRANSFORMER_ATOL = 1e-8


def _activate_conditioning(layer: ConditionedTransformerLayer) -> None:
    with torch.no_grad():
        layer.cond[2].weight.normal_(std=0.02)
        layer.cond[2].bias.normal_(std=0.02)


def _manual_residual(
    layer: ConditionedTransformerLayer,
    x: torch.Tensor,
    pairs: torch.Tensor,
    c: torch.Tensor,
    mask: torch.Tensor,
    multiplicity: int,
) -> torch.Tensor:
    """Reconstruct the layer output by hand from ``c`` via ``layer.cond``."""
    a_a, a_b, a_g, m_a, m_b, m_g = layer.cond(c).chunk(6, dim=-1)
    h = layer.attention(
        tokens=layer.lns_scale * ada_ln(x, a_a, a_b),
        pairs=pairs,
        mask=mask,
        multiplicity=multiplicity,
    )
    ref = x + ada_scale(h, a_g)
    h = layer.transition(layer.lns_scale * ada_ln(ref, m_a, m_b))
    return ref + ada_scale(h, m_g)


@pytest.mark.parametrize(
    "lns_scale",
    [pytest.param(1.0, id="lns=1"), pytest.param(0.5, id="lns=0.5")],
)
def test_full_attention_layer_matches_manual_residual(lns_scale: float) -> None:
    """The full layer matches a direct residual calculation."""
    torch.manual_seed(0)
    B, N, M = 1, 4, 1
    layer = ConditionedTransformerLayer(
        dim=ATOM_S,
        dim_pairwise=ATOM_Z,
        dim_cond=ATOM_S,
        heads=2,
        transition_hidden=4 * ATOM_S,
        is_sliding_window=False,
        lns_scale=lns_scale,
    ).eval()
    _activate_conditioning(layer)

    x = torch.randn(B * M, N, ATOM_S)
    pairs = torch.randn(B, N, N, ATOM_Z)
    c = torch.randn(B * M, N, ATOM_S)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -1] = False

    out = layer(x, pairs, c, mask=mask, multiplicity=M)
    ref = _manual_residual(layer, x, pairs, c, mask, M)

    torch.testing.assert_close(out, ref, rtol=0, atol=TRANSFORMER_ATOL)
    assert not torch.allclose(out, x)


def test_sliding_window_layer_matches_manual_residual() -> None:
    """The sliding-window layer matches a direct residual calculation."""
    torch.manual_seed(0)
    B, N, M = 1, ATOMS_PER_WINDOW_QUERIES, 3
    layer = ConditionedSlidingWindowTransformerLayer(
        dim=ATOM_S,
        dim_pairwise=ATOM_Z,
        dim_cond=ATOM_S,
        heads=2,
        transition_hidden=4 * ATOM_S,
        window_queries=ATOMS_PER_WINDOW_QUERIES,
        window_keys=ATOMS_PER_WINDOW_KEYS,
    ).eval()
    _activate_conditioning(layer.transformer_layer)

    x = torch.randn(B * M, N, ATOM_S)
    pairs = torch.randn(B, N, ATOMS_PER_WINDOW_KEYS, ATOM_Z)
    c = torch.randn(B * M, N, ATOM_S)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, -4:] = False

    out = layer(x, pairs, c, mask=mask, multiplicity=M)
    # The wrapper delegates to the inner ConditionedTransformerLayer.
    ref = _manual_residual(layer.transformer_layer, x, pairs, c, mask, M)

    torch.testing.assert_close(out, ref, rtol=0, atol=TRANSFORMER_ATOL)
    assert not torch.allclose(out, x)


@pytest.mark.parametrize("is_sliding_window", [False, True])
def test_attention_layer_zero_init_output(is_sliding_window: bool) -> None:
    """``zero_init_output`` zero-inits the APB output projection weight for standalone use."""

    def _proj_o(attn: nn.Module) -> torch.Tensor:
        # SlidingWindowAttentionLayer wraps an inner AttentionLayer.
        pair_bias_attn = getattr(attn, "attention_layer", attn).pair_bias_attn
        return pair_bias_attn.proj_o.weight

    if is_sliding_window:
        make = lambda zero: SlidingWindowAttentionLayer(  # noqa: E731
            attn_window_queries=ATOMS_PER_WINDOW_QUERIES,
            attn_window_keys=ATOMS_PER_WINDOW_KEYS,
            heads=2,
            dim=ATOM_S,
            dim_pairwise=ATOM_Z,
            zero_init_output=zero,
        )
    else:
        make = lambda zero: AttentionLayer(  # noqa: E731
            heads=2,
            dim=ATOM_S,
            dim_pairwise=ATOM_Z,
            zero_init_output=zero,
        )

    assert torch.all(_proj_o(make(True)) == 0.0)
    assert torch.any(_proj_o(make(False)) != 0.0)


def test_layer_grad_flows_to_input_and_conditioning() -> None:
    """Gradients propagate through both the input and the conditioning projection params."""
    torch.manual_seed(0)
    B, N, M = 1, 4, 1
    layer = ConditionedTransformerLayer(
        dim=ATOM_S,
        dim_pairwise=ATOM_Z,
        dim_cond=ATOM_S,
        heads=2,
        transition_hidden=4 * ATOM_S,
        is_sliding_window=False,
    )
    _activate_conditioning(layer)
    x = torch.randn(B * M, N, ATOM_S, requires_grad=True)
    pairs = torch.randn(B, N, N, ATOM_Z)
    c = torch.randn(B * M, N, ATOM_S)
    mask = torch.ones(B, N, dtype=torch.bool)

    out = layer(x, pairs, c, mask=mask, multiplicity=M)
    out.sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.cond[2].weight.grad is not None
    assert torch.isfinite(layer.cond[2].weight.grad).all()
    for module_name in ("attention", "transition"):
        parameters = list(getattr(layer, module_name).parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)
        assert any(torch.any(parameter.grad != 0) for parameter in parameters)
