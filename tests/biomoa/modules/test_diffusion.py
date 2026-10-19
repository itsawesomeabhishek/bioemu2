"""Tests for `biomoa/modules/diffusion.py`.

Layout:
    * Helper / pure-function tests (`ada_ln`, `ada_scale`)
    * Submodule shape tests that don't need the real feats dict
      (`ReadoutSuperposition`, conditioning, attention)
    * Submodule tests that need the realistic feats dict (`AtomAttentionEncoder`
      configured as a feature embedder, `DualStreamDiffusionLayer`)
    * Integration tests for `DualStreamDiffusionModule`, parametrized over the
      relevant config knobs.

Tests use small dimensions defined in `conftest.py` so they run quickly on CPU.
A single `gpu_device` smoke test exercises the full forward on CUDA when available.
"""

from typing import Any

import pytest
import torch
from torch.nn import functional as F

from bioemu2.biomoa.modules.diffusion import (
    DualStreamDiffusionLayer,
    DualStreamDiffusionModule,
    PairwiseConditioning,
    ReadoutSuperposition,
    SingleConditioning,
)
from bioemu2.biomoa.modules.encoders import AtomAttentionEncoder
from bioemu2.biomoa.modules.transformer import (
    AttentionLayer,
    SlidingWindowAttentionLayer,
    ada_ln,
    ada_scale,
)

from .conftest import (
    ATOM_S,
    ATOM_Z,
    ATOMS_PER_WINDOW_KEYS,
    ATOMS_PER_WINDOW_QUERIES,
    S_INPUT_DIM,
    TOKEN_S,
    TOKEN_Z,
)

# Helper / pure-function tests


class TestAdaLn:
    def test_shape_preserved(self) -> None:
        torch.manual_seed(0)
        x = torch.randn(2, 4, 8)
        alpha = torch.randn(2, 4, 8)
        beta = torch.randn(2, 4, 8)
        out = ada_ln(x, alpha=alpha, beta=beta)
        assert out.shape == x.shape

    def test_zero_alpha_returns_ln_plus_beta(self) -> None:
        """With alpha=0, ada_ln returns LN(x) * (1 + 0) + beta = LN(x) + beta."""
        x = torch.randn(2, 4, 8)
        alpha = torch.zeros(2, 4, 8)
        beta = torch.full((2, 4, 8), 3.0)
        out = ada_ln(x, alpha=alpha, beta=beta)
        expected = F.layer_norm(x, (8,)) + beta
        torch.testing.assert_close(out, expected)


class TestAdaScale:
    def test_shape_and_zero_gamma(self) -> None:
        x = torch.randn(2, 4, 8)
        gamma = torch.zeros(2, 4, 8)
        assert ada_scale(x, gamma).shape == x.shape
        torch.testing.assert_close(ada_scale(x, gamma), torch.zeros_like(x))

    def test_unit_gamma_is_identity(self) -> None:
        x = torch.randn(2, 4, 8)
        gamma = torch.ones(2, 4, 8)
        torch.testing.assert_close(ada_scale(x, gamma), x)


class TestReadoutSuperposition:
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        B, N_atoms, N_token = 1, 32, 4
        layer = ReadoutSuperposition(atom_s=ATOM_S, token_s=TOKEN_S)
        a = torch.randn(B, N_token, TOKEN_S)
        q = torch.randn(B, N_atoms, ATOM_S)
        atom_to_token_ids = torch.zeros(B, N_atoms, dtype=torch.long)
        # Non-trivial mask: last 4 atoms are padded.
        atom_pad_mask = torch.ones(B, N_atoms)
        atom_pad_mask[:, -4:] = 0
        is_padded_atom = ~atom_pad_mask.bool()
        out = layer(
            a=a,
            q=q,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=1,
        )
        assert out.shape == (B, N_atoms, 3)

    def test_grad_flow(self) -> None:
        torch.manual_seed(0)
        B, N_atoms, N_token = 1, 32, 4
        layer = ReadoutSuperposition(atom_s=ATOM_S, token_s=TOKEN_S)
        a = torch.randn(B, N_token, TOKEN_S, requires_grad=True)
        q = torch.randn(B, N_atoms, ATOM_S, requires_grad=True)
        atom_to_token_ids = torch.zeros(B, N_atoms, dtype=torch.long)
        # Non-trivial mask: last 4 atoms are padded.
        atom_pad_mask = torch.ones(B, N_atoms)
        atom_pad_mask[:, -4:] = 0
        is_padded_atom = ~atom_pad_mask.bool()
        out = layer(
            a=a,
            q=q,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=1,
        )
        out.sum().backward()
        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert q.grad is not None and torch.isfinite(q.grad).all()

    def test_first_replicate_matches_standalone(self) -> None:
        torch.manual_seed(0)
        B, N_atoms, N_token = 1, 32, 4
        multiplicity = 2
        layer = ReadoutSuperposition(atom_s=ATOM_S, token_s=TOKEN_S)
        atom_to_token_ids = torch.randint(0, N_token, (B, N_atoms))
        is_padded_atom = torch.zeros(B, N_atoms, dtype=torch.bool)

        a0 = torch.randn(B, N_token, TOKEN_S)
        q0 = torch.randn(B, N_atoms, ATOM_S)
        out_standalone = layer(
            a=a0,
            q=q0,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=1,
        )
        a = torch.cat([a0, torch.randn(multiplicity - 1, N_token, TOKEN_S)], dim=0)
        q = torch.cat([q0, torch.randn(multiplicity - 1, N_atoms, ATOM_S)], dim=0)
        out_batched = layer(
            a=a,
            q=q,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=multiplicity,
        )
        torch.testing.assert_close(out_batched[:1], out_standalone, atol=0.0, rtol=0.0)

    def test_unreferenced_token_has_no_influence(self) -> None:
        torch.manual_seed(0)
        B, N_atoms, N_token = 1, 32, 4
        layer = ReadoutSuperposition(atom_s=ATOM_S, token_s=TOKEN_S)
        atom_to_token_ids = torch.randint(0, N_token - 1, (B, N_atoms))
        is_padded_atom = torch.zeros(B, N_atoms, dtype=torch.bool)
        q = torch.randn(B, N_atoms, ATOM_S)
        a = torch.randn(B, N_token, TOKEN_S)

        out = layer(
            a=a,
            q=q,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=1,
        )
        a_perturbed = a.clone()
        a_perturbed[:, N_token - 1, :] = torch.randn(B, TOKEN_S)
        out_perturbed = layer(
            a=a_perturbed,
            q=q,
            atom_to_token_ids=atom_to_token_ids,
            is_padded_atom=is_padded_atom,
            multiplicity=1,
        )
        torch.testing.assert_close(out_perturbed, out, atol=0.0, rtol=0.0)


class TestSingleConditioning:
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        B, N_token = 2, 6
        layer = SingleConditioning(
            sigma_data=16.0,
            token_s=TOKEN_S,
            s_input_dim=S_INPUT_DIM,
            a_s=TOKEN_S,
            dim_fourier=32,
            num_transitions=1,
        )
        s_trunk = torch.randn(B, N_token, TOKEN_S)
        s_inputs = torch.randn(B, N_token, S_INPUT_DIM)
        times = torch.randn(B)
        s_out, normed_fourier = layer(
            times=times,
            s_trunk=s_trunk,
            s_inputs=s_inputs,
        )
        assert s_out.shape == (B, N_token, TOKEN_S)
        assert normed_fourier.shape == (B, 32)


class TestPairwiseConditioning:
    def test_forward_shape(self) -> None:
        torch.manual_seed(0)
        B, N_token = 2, 5
        layer = PairwiseConditioning(
            token_z=TOKEN_Z, dim_token_rel_pos_feats=TOKEN_Z, a_z=TOKEN_Z, num_transitions=1
        )
        z_trunk = torch.randn(B, N_token, N_token, TOKEN_Z)
        rel_pos = torch.randn(B, N_token, N_token, TOKEN_Z)
        out = layer(z_trunk=z_trunk, token_rel_pos_feats=rel_pos)
        assert out.shape == (B, N_token, N_token, TOKEN_Z)


class TestAttentionLayer:
    @pytest.mark.parametrize(
        "multiplicity",
        [pytest.param(1, id="M=1"), pytest.param(3, id="M=3")],
    )
    def test_forward_shape(self, multiplicity: int) -> None:
        torch.manual_seed(0)
        B, N, M = 1, 4, multiplicity
        layer = AttentionLayer(heads=2, dim=ATOM_S, dim_pairwise=ATOM_Z)
        tokens = torch.randn(B * M, N, ATOM_S)
        pairs = torch.randn(B, N, N, ATOM_Z)
        # Non-trivial mask: last position is padded.
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -1] = False
        out = layer(tokens=tokens, pairs=pairs, mask=mask, multiplicity=M)
        assert out.shape == (B * M, N, ATOM_S)

    def test_grad_flow(self) -> None:
        torch.manual_seed(0)
        B, N, M = 1, 4, 1
        layer = AttentionLayer(heads=2, dim=ATOM_S, dim_pairwise=ATOM_Z)
        tokens = torch.randn(B * M, N, ATOM_S, requires_grad=True)
        pairs = torch.randn(B, N, N, ATOM_Z, requires_grad=True)
        # Non-trivial mask: last position is padded.
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, -1] = False
        out = layer(tokens=tokens, pairs=pairs, mask=mask, multiplicity=M)
        out.sum().backward()
        assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


class TestSlidingWindowAttentionLayer:
    @pytest.mark.parametrize(
        "multiplicity",
        [pytest.param(1, id="M=1"), pytest.param(3, id="M=3")],
    )
    def test_forward_shape_with_windowing(self, multiplicity: int) -> None:
        """When window sizes are set, output preserves the (B*M, N_atoms, D) shape."""
        torch.manual_seed(0)
        B, N_atoms = 1, ATOMS_PER_WINDOW_QUERIES  # one window
        M = multiplicity
        layer = SlidingWindowAttentionLayer(
            attn_window_queries=ATOMS_PER_WINDOW_QUERIES,
            attn_window_keys=ATOMS_PER_WINDOW_KEYS,
            heads=2,
            dim=ATOM_S,
            dim_pairwise=ATOM_Z,
        )
        # tokens carry M; pairs and mask do not (APB broadcasts M internally).
        tokens = torch.randn(B * M, N_atoms, ATOM_S)
        pairs = torch.randn(B, N_atoms, ATOMS_PER_WINDOW_KEYS, ATOM_Z)
        # Non-trivial mask: last 4 atoms are padded. Shape is [B, N_atoms] (no M).
        mask = torch.ones(B, N_atoms, dtype=torch.bool)
        mask[:, -4:] = False

        out = layer(tokens=tokens, pairs=pairs, mask=mask, multiplicity=M)
        assert out.shape == (B * M, N_atoms, ATOM_S)


# AtomAttentionEncoder without atom encoding or token aggregation.


def _make_feature_embedder(
    atoms_per_window_queries: int = ATOMS_PER_WINDOW_QUERIES,
) -> AtomAttentionEncoder:
    return AtomAttentionEncoder(
        atom_s=ATOM_S,
        atom_z=ATOM_Z,
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
        structure_prediction=True,
        use_atom_encoder=False,
        use_atom_to_token_aggregation=False,
    )


class TestFeatureEmbedder:
    """`AtomAttentionEncoder` configured as a feature embedder."""

    def test_first_pass_shapes(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        torch.manual_seed(0)
        layer = _make_feature_embedder()
        B, N_atoms, N_token = (
            feats_shapes["B"],
            feats_shapes["N_atoms"],
            feats_shapes["N_token"],
        )
        r = torch.randn(B, N_atoms, 3)
        s_trunk = torch.randn(B, N_token, TOKEN_S)
        z_trunk = torch.randn(B, N_token, N_token, TOKEN_Z)

        a, q, c, p = layer(
            atom_ref_pos=real_feats["template_coords"],
            ref_charge=real_feats["charge"],
            ref_element=real_feats["element"],
            ref_atom_name_chars=real_feats["name_encoding"],
            atom_ref_space_uid=real_feats["ref_space_uid"],
            atom_pad_mask=real_feats["atom_pad_mask"],
            atom_to_token_ids=real_feats["token_idx_at_atom"],
            token_pad_mask=real_feats["token_pad_mask"],
            s_trunk=s_trunk,
            z=z_trunk,
            r=r,
            multiplicity=1,
        )

        assert a is None
        assert q.shape == (B, N_atoms, ATOM_S)
        assert c.shape == (B, N_atoms, ATOM_S)
        assert p.dim() == 5  # (B, K, W, H, atom_z)
        assert p.shape[-1] == ATOM_Z
        # No transformer / aggregation modules should be attached.
        assert not hasattr(layer, "atom_encoder_layers")
        assert not hasattr(layer, "atom_to_token_trans")

    def test_misaligned_window_raises(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        # Use an apw that does not divide N_atoms (which is a multiple of 32).
        bad_apw = 7
        layer = _make_feature_embedder(atoms_per_window_queries=bad_apw)
        B, N_atoms, N_token = (
            feats_shapes["B"],
            feats_shapes["N_atoms"],
            feats_shapes["N_token"],
        )
        r = torch.randn(B, N_atoms, 3)
        s_trunk = torch.randn(B, N_token, TOKEN_S)
        z_trunk = torch.randn(B, N_token, N_token, TOKEN_Z)
        with pytest.raises(AssertionError, match="not divisible"):
            layer(
                atom_ref_pos=real_feats["template_coords"],
                ref_charge=real_feats["charge"],
                ref_element=real_feats["element"],
                ref_atom_name_chars=real_feats["name_encoding"],
                atom_ref_space_uid=real_feats["ref_space_uid"],
                atom_pad_mask=real_feats["atom_pad_mask"],
                atom_to_token_ids=real_feats["token_idx_at_atom"],
                token_pad_mask=real_feats["token_pad_mask"],
                s_trunk=s_trunk,
                z=z_trunk,
                r=r,
                multiplicity=1,
            )


class TestAtomAttentionEncoderFlagCombinations:
    """Cover the four (use_atom_encoder, use_atom_to_token_aggregation) combinations."""

    @pytest.mark.parametrize(
        "use_atom_encoder,use_atom_to_token_aggregation",
        [(True, True), (False, True), (True, False), (False, False)],
    )
    def test_module_attrs_and_outputs(
        self,
        real_feats: dict[str, torch.Tensor],
        feats_shapes: dict[str, int],
        use_atom_encoder: bool,
        use_atom_to_token_aggregation: bool,
    ) -> None:
        torch.manual_seed(0)
        layer = AtomAttentionEncoder(
            atom_s=ATOM_S,
            atom_z=ATOM_Z,
            token_s=TOKEN_S,
            token_z=TOKEN_Z,
            atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
            atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
            atom_encoder_depth=1,
            atom_encoder_heads=2,
            structure_prediction=True,
            use_atom_encoder=use_atom_encoder,
            use_atom_to_token_aggregation=use_atom_to_token_aggregation,
        )
        assert hasattr(layer, "atom_encoder_layers") is use_atom_encoder
        assert hasattr(layer, "atom_to_token_trans") is use_atom_to_token_aggregation

        B, N_atoms, N_token = (
            feats_shapes["B"],
            feats_shapes["N_atoms"],
            feats_shapes["N_token"],
        )
        r = torch.randn(B, N_atoms, 3)
        s_trunk = torch.randn(B, N_token, TOKEN_S)
        z_trunk = torch.randn(B, N_token, N_token, TOKEN_Z)

        a, q, c, p = layer(
            atom_ref_pos=real_feats["template_coords"],
            ref_charge=real_feats["charge"],
            ref_element=real_feats["element"],
            ref_atom_name_chars=real_feats["name_encoding"],
            atom_ref_space_uid=real_feats["ref_space_uid"],
            atom_pad_mask=real_feats["atom_pad_mask"],
            atom_to_token_ids=real_feats["token_idx_at_atom"],
            token_pad_mask=real_feats["token_pad_mask"],
            s_trunk=s_trunk,
            z=z_trunk,
            r=r,
            multiplicity=1,
        )

        assert q.shape == (B, N_atoms, ATOM_S)
        assert c.shape == (B, N_atoms, ATOM_S)
        assert p.shape[-1] == ATOM_Z
        if use_atom_to_token_aggregation:
            assert a is not None
            assert a.shape == (B, N_token, 2 * TOKEN_S)
        else:
            assert a is None


# Helpers for the layer-level tests below


def _build_layer_inputs(
    real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct (s, z, q, p, c, atom_mask) suitable for DualStreamDiffusionLayer."""
    torch.manual_seed(0)
    B, N_atoms, N_token = feats_shapes["B"], feats_shapes["N_atoms"], feats_shapes["N_token"]
    embedder = _make_feature_embedder()
    r = torch.randn(B, N_atoms, 3)
    s_trunk_in = torch.randn(B, N_token, TOKEN_S)
    z_trunk_in = torch.randn(B, N_token, N_token, TOKEN_Z)
    _, q, _, p = embedder(
        atom_ref_pos=real_feats["template_coords"],
        ref_charge=real_feats["charge"],
        ref_element=real_feats["element"],
        ref_atom_name_chars=real_feats["name_encoding"],
        atom_ref_space_uid=real_feats["ref_space_uid"],
        atom_pad_mask=real_feats["atom_pad_mask"],
        atom_to_token_ids=real_feats["token_idx_at_atom"],
        token_pad_mask=real_feats["token_pad_mask"],
        s_trunk=s_trunk_in,
        z=z_trunk_in,
        r=r,
        multiplicity=1,
    )
    atom_mask = real_feats["atom_pad_mask"]
    s = torch.randn(B, N_token, TOKEN_S)
    z = torch.randn(B, N_token, N_token, TOKEN_Z)
    c = torch.randn(B, N_token, TOKEN_S)
    return s, z, q, p, c, atom_mask


class TestDualStreamDiffusionLayer:
    def test_forward_shapes(
        self,
        real_feats: dict[str, torch.Tensor],
        feats_shapes: dict[str, int],
    ) -> None:
        torch.manual_seed(0)
        s, z, q, p, c, atom_mask = _build_layer_inputs(real_feats, feats_shapes)
        layer = DualStreamDiffusionLayer(
            a_s=TOKEN_S,
            a_z=TOKEN_Z,
            atom_s=ATOM_S,
            atom_z=ATOM_Z,
            atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
            atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
            atom_attention_heads=2,
            token_attention_heads=2,
        )
        s_out, q_out = layer(
            s,
            z,
            q,
            p,
            c,
            atom_mask,
            token_pad_mask=real_feats["token_pad_mask"],
            atom_to_token_ids=real_feats["token_idx_at_atom"],
            is_padded_atom=~real_feats["atom_pad_mask"].bool(),
            token_center_atom_index=real_feats["token_center_atom_index"],
            multiplicity=1,
        )
        assert s_out.shape == s.shape
        assert q_out.shape == q.shape

    def test_grad_flow(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        torch.manual_seed(0)
        s, z, q, p, c, atom_mask = _build_layer_inputs(real_feats, feats_shapes)
        s = s.detach().requires_grad_(True)
        q = q.detach().requires_grad_(True)
        layer = DualStreamDiffusionLayer(
            a_s=TOKEN_S,
            a_z=TOKEN_Z,
            atom_s=ATOM_S,
            atom_z=ATOM_Z,
            atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
            atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
            atom_attention_heads=2,
            token_attention_heads=2,
        )
        s_out, q_out = layer(
            s,
            z,
            q,
            p,
            c,
            atom_mask,
            token_pad_mask=real_feats["token_pad_mask"],
            atom_to_token_ids=real_feats["token_idx_at_atom"],
            is_padded_atom=~real_feats["atom_pad_mask"].bool(),
            token_center_atom_index=real_feats["token_center_atom_index"],
            multiplicity=1,
        )
        (s_out.sum() + q_out.sum()).backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()
        assert q.grad is not None and torch.isfinite(q.grad).all()


# DualStreamDiffusionModule integration tests


def _build_diffusion_module_inputs(
    real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int], multiplicity: int = 1
) -> dict[str, torch.Tensor]:
    """Construct the kwargs needed for ``DualStreamDiffusionModule.forward``."""
    B = feats_shapes["B"]
    N_atoms = feats_shapes["N_atoms"]
    N_token = feats_shapes["N_token"]
    return {
        "s_trunk": torch.randn(B, N_token, TOKEN_S),
        "s_inputs": torch.randn(B, N_token, S_INPUT_DIM),
        "z_trunk": torch.randn(B, N_token, N_token, TOKEN_Z),
        "relative_position_encoding": torch.randn(B, N_token, N_token, TOKEN_Z),
        "r_noisy": torch.randn(B * multiplicity, N_atoms, 3),
        "times": torch.randn(B * multiplicity),
    }


def _make_diffusion_module(**overrides: Any) -> DualStreamDiffusionModule:
    kwargs: dict[str, Any] = dict(
        token_s=TOKEN_S,
        token_z=TOKEN_Z,
        atom_s=ATOM_S,
        atom_z=ATOM_Z,
        s_input_dim=S_INPUT_DIM,
        num_layers=1,
        atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
        atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
        dim_fourier=32,
        atom_attention_heads=2,
        token_attention_heads=2,
        conditioning_transition_layers=1,
    )
    kwargs.update(overrides)
    return DualStreamDiffusionModule(**kwargs)


class TestDualStreamDiffusionModule:
    def test_forward_shapes(
        self,
        real_feats: dict[str, torch.Tensor],
        feats_shapes: dict[str, int],
    ) -> None:
        torch.manual_seed(0)
        B, N_atoms, N_token = (
            feats_shapes["B"],
            feats_shapes["N_atoms"],
            feats_shapes["N_token"],
        )
        module = _make_diffusion_module()
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes)
        out = module(feats=real_feats, multiplicity=1, **inputs)

        assert set(out.keys()) == {"r_update", "token_a"}
        assert out["r_update"].shape == (B, N_atoms, 3)
        assert out["token_a"].shape == (B, N_token, TOKEN_S)


class TestDualStreamDiffusionDenoising:
    def _build(self, **overrides: Any) -> DualStreamDiffusionModule:
        kwargs: dict[str, Any] = dict(
            token_s=TOKEN_S,
            token_z=TOKEN_Z,
            atom_s=ATOM_S,
            atom_z=ATOM_Z,
            s_input_dim=S_INPUT_DIM,
            num_layers=1,
            atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
            atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
            dim_fourier=32,
            atom_attention_heads=2,
            token_attention_heads=2,
            conditioning_transition_layers=1,
        )
        kwargs.update(overrides)
        return DualStreamDiffusionModule(**kwargs)

    def test_forward_shape(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        torch.manual_seed(0)
        B, N_atoms = feats_shapes["B"], feats_shapes["N_atoms"]
        model = self._build()
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes)
        score = model(
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            z_trunk=inputs["z_trunk"],
            relative_position_encoding=inputs["relative_position_encoding"],
            feats=real_feats,
            r_noisy=inputs["r_noisy"],
            times=inputs["times"],
            multiplicity=1,
        )["r_update"]
        assert score.shape == (B, N_atoms, 3)

    def test_multiplicity_expansion(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        torch.manual_seed(0)
        M = 3
        B, N_atoms = feats_shapes["B"], feats_shapes["N_atoms"]
        model = self._build()
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes, multiplicity=M)
        # Expand time tensor to the per-noise-sample length.
        t = torch.randn(B * M)
        score = model(
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            z_trunk=inputs["z_trunk"],
            relative_position_encoding=inputs["relative_position_encoding"],
            feats=real_feats,
            r_noisy=inputs["r_noisy"],
            times=t,
            multiplicity=M,
        )["r_update"]
        assert score.shape == (B * M, N_atoms, 3)

    def test_atom_resolved_mask_zeros_input_coords(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        """The diffusion module zeros noisy coordinates by ``atom_mask`` before scoring.

        Verify the masked positions are not propagated as if they had non-zero values: scoring
        with an all-zero mask must give an output that does not depend on the coordinates
        passed in (only on feats and zero-coords).
        """
        torch.manual_seed(0)
        B, N_atoms = feats_shapes["B"], feats_shapes["N_atoms"]
        model = self._build().eval()
        feats_zero = dict(real_feats)
        feats_zero["atom_mask"] = torch.zeros_like(real_feats["atom_mask"])
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes)

        with torch.no_grad():
            score_a = model(
                s_inputs=inputs["s_inputs"],
                s_trunk=inputs["s_trunk"],
                z_trunk=inputs["z_trunk"],
                relative_position_encoding=inputs["relative_position_encoding"],
                feats=feats_zero,
                r_noisy=inputs["r_noisy"],
                times=inputs["times"],
                multiplicity=1,
            )["r_update"]
            score_b = model(
                s_inputs=inputs["s_inputs"],
                s_trunk=inputs["s_trunk"],
                z_trunk=inputs["z_trunk"],
                relative_position_encoding=inputs["relative_position_encoding"],
                feats=feats_zero,
                r_noisy=torch.randn(B, N_atoms, 3),
                times=inputs["times"],
                multiplicity=1,
            )["r_update"]
        torch.testing.assert_close(score_a, score_b)

    def test_grad_flow(
        self, real_feats: dict[str, torch.Tensor], feats_shapes: dict[str, int]
    ) -> None:
        torch.manual_seed(0)
        # Use ≥2 layers so z_trunk's contribution (which only enters `s` per layer) can
        # cycle into `q` through the second layer's atom-attention conditioning path.
        model = self._build(num_layers=2)
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes)
        for k in ("s_trunk", "s_inputs", "z_trunk", "r_noisy"):
            inputs[k] = inputs[k].requires_grad_(True)
        score = model(
            s_inputs=inputs["s_inputs"],
            s_trunk=inputs["s_trunk"],
            z_trunk=inputs["z_trunk"],
            relative_position_encoding=inputs["relative_position_encoding"],
            feats=real_feats,
            r_noisy=inputs["r_noisy"],
            times=inputs["times"],
            multiplicity=1,
        )["r_update"]
        score.sum().backward()
        for k in ("s_trunk", "s_inputs", "z_trunk", "r_noisy"):
            grad = inputs[k].grad
            assert grad is not None and torch.isfinite(grad).all(), (
                f"missing or non-finite gradient for {k}"
            )

    def test_cuda_smoke(
        self,
        gpu_device: torch.device,
        real_feats: dict[str, torch.Tensor],
        feats_shapes: dict[str, int],
    ) -> None:
        torch.manual_seed(0)
        B, N_atoms = feats_shapes["B"], feats_shapes["N_atoms"]
        model = self._build().to(gpu_device).eval()
        feats_gpu = {k: v.to(gpu_device) for k, v in real_feats.items()}
        inputs = _build_diffusion_module_inputs(real_feats, feats_shapes)
        inputs = {k: v.to(gpu_device) for k, v in inputs.items()}
        with torch.no_grad():
            score = model(
                s_inputs=inputs["s_inputs"],
                s_trunk=inputs["s_trunk"],
                z_trunk=inputs["z_trunk"],
                relative_position_encoding=inputs["relative_position_encoding"],
                feats=feats_gpu,
                r_noisy=inputs["r_noisy"],
                times=inputs["times"],
                multiplicity=1,
            )["r_update"]
        assert score.shape == (B, N_atoms, 3)
        assert torch.isfinite(score).all()


def _repeat_feats(feats: dict[str, torch.Tensor], B: int) -> dict[str, torch.Tensor]:
    """Repeat a B=1 feats dict to create a batch of size B."""
    return {k: v.repeat_interleave(B, dim=0) for k, v in feats.items()}


def _apply_nontrivial_mask(feats: dict[str, torch.Tensor], N_atoms: int) -> dict[str, torch.Tensor]:
    """Zero out trailing atom/token positions with variable padding per batch element.

    Ensures at least one full window (ATOMS_PER_WINDOW_QUERIES) of valid atoms remains
    so that attention softmax does not see all-masked rows.
    """
    feats = dict(feats)
    B = feats["atom_pad_mask"].shape[0]
    atom_pad_mask = feats["atom_pad_mask"].clone()
    atom_resolved_mask = feats["atom_mask"].clone()
    token_pad_mask = feats["token_pad_mask"].clone()
    N_token = token_pad_mask.shape[1]

    for b in range(B):
        # Pad trailing atoms — ensure at least one window remains valid.
        atom_pad_len = min((b + 1) * ATOMS_PER_WINDOW_QUERIES, N_atoms - ATOMS_PER_WINDOW_QUERIES)
        if atom_pad_len > 0:
            atom_pad_mask[b, N_atoms - atom_pad_len :] = False
            atom_resolved_mask[b, N_atoms - atom_pad_len :] = False
        # Pad trailing tokens — keep at least 2 valid.
        token_pad_len = min((b + 1), N_token - 2)
        if token_pad_len > 0:
            token_pad_mask[b, N_token - token_pad_len :] = False

    feats["atom_pad_mask"] = atom_pad_mask
    feats["atom_mask"] = atom_resolved_mask
    feats["token_pad_mask"] = token_pad_mask
    return feats


def _ground_truth_sequential_diffusion(
    model: DualStreamDiffusionModule,
    feats: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
    B: int,
    M: int,
) -> torch.Tensor:
    """Ground truth: loop B=1, M=1 calls and stack results (no batching or multiplicity packing)."""
    outs: list[torch.Tensor] = []
    for b in range(B):
        for m in range(M):
            # Slice feats to B=1.
            feats_b: dict[str, torch.Tensor] = {k: v[b : b + 1] for k, v in feats.items()}

            score_bm = model(
                s_inputs=inputs["s_inputs"][b : b + 1],
                s_trunk=inputs["s_trunk"][b : b + 1],
                z_trunk=inputs["z_trunk"][b : b + 1],
                relative_position_encoding=inputs["relative_position_encoding"][b : b + 1],
                feats=feats_b,
                r_noisy=inputs["r_noisy"][b * M + m : b * M + m + 1],
                times=inputs["times"][b * M + m : b * M + m + 1],
                multiplicity=1,
            )["r_update"]
            outs.append(score_bm)
    return torch.cat(outs, dim=0)


class TestBatchMultiplicityParity:
    """Batched (B, M) forward matches B=1 M=1 ground truth computed sequentially."""

    def _build(self, **overrides: Any) -> DualStreamDiffusionModule:
        kwargs: dict[str, Any] = dict(
            token_s=TOKEN_S,
            token_z=TOKEN_Z,
            atom_s=ATOM_S,
            atom_z=ATOM_Z,
            s_input_dim=S_INPUT_DIM,
            num_layers=1,
            atoms_per_window_queries=ATOMS_PER_WINDOW_QUERIES,
            atoms_per_window_keys=ATOMS_PER_WINDOW_KEYS,
            dim_fourier=32,
            atom_attention_heads=2,
            token_attention_heads=2,
            conditioning_transition_layers=1,
        )
        kwargs.update(overrides)
        return DualStreamDiffusionModule(**kwargs)

    @pytest.mark.parametrize("B", [1, 2], ids=["B1", "B2"])
    @pytest.mark.parametrize("M", [1, 2, 3], ids=["M1", "M2", "M3"])
    def test_forward_parity(
        self,
        real_feats: dict[str, torch.Tensor],
        feats_shapes: dict[str, int],
        B: int,
        M: int,
    ) -> None:
        """Batched forward with non-trivial masks matches sequential B=1, M=1 ground truth."""
        torch.manual_seed(42)
        N_atoms = feats_shapes["N_atoms"]
        N_token = feats_shapes["N_token"]

        model = self._build().eval()

        # Create a batch and apply non-trivial masks.
        feats_batched = _repeat_feats(real_feats, B)
        feats_batched = _apply_nontrivial_mask(feats_batched, N_atoms)

        inputs = {
            "s_trunk": torch.randn(B, N_token, TOKEN_S),
            "s_inputs": torch.randn(B, N_token, S_INPUT_DIM),
            "z_trunk": torch.randn(B, N_token, N_token, TOKEN_Z),
            "relative_position_encoding": torch.randn(B, N_token, N_token, TOKEN_Z),
            "r_noisy": torch.randn(B * M, N_atoms, 3),
            "times": torch.randn(B * M),
        }

        with torch.no_grad():
            # Batched forward.
            score_batched = model(
                s_inputs=inputs["s_inputs"],
                s_trunk=inputs["s_trunk"],
                z_trunk=inputs["z_trunk"],
                relative_position_encoding=inputs["relative_position_encoding"],
                feats=feats_batched,
                r_noisy=inputs["r_noisy"],
                times=inputs["times"],
                multiplicity=M,
            )["r_update"]

            # Sequential ground truth (no batching, no multiplicity packing).
            score_ref = _ground_truth_sequential_diffusion(model, feats_batched, inputs, B, M)

        assert score_batched.shape == score_ref.shape == (B * M, N_atoms, 3)
        assert score_batched.abs().max().item() > 0, "output is trivially zero"
        torch.testing.assert_close(score_batched, score_ref, atol=1e-5, rtol=1e-5)
