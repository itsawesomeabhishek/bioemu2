"""Tests full-attention AttentionPairBias implementations."""

import pytest
import torch
from _pytest.mark.structures import ParameterSet

from bioemu2.biomoa.ops.attention_pair_bias import AttentionPairBias
from tests.conftest import set_matmul_precision

from ._accuracy import VariantAccuracyCase

C_S = 384
C_Z = 128
NUM_HEADS = 16
N = 64

# These head counts cover the generic and specialized kernels.
NUM_HEADS_VARIANTS = [16, 8]

_FP32_ACCURACY_CASES = [
    VariantAccuracyCase(
        "vanilla-high",
        "vanilla",
        "vanilla",
        matmul_precision="high",
        forward_tolerance=(1e-3, 0.0),
        input_gradient_tolerance=(5e-2, 0.0),
        parameter_gradient_tolerance=(5e-2, 0.0),
    ),
]
ACCURACY_CASES = [pytest.param(case, id=case.name) for case in _FP32_ACCURACY_CASES]

INVARIANCE_CASES = [
    VariantAccuracyCase(
        "vanilla",
        "vanilla",
        "vanilla",
        input_gradient_tolerance=(2e-6, 0.0),
        parameter_gradient_tolerance=(2e-6, 0.0),
    ),
]
INVARIANCE_CASES = [pytest.param(case, id=case.name) for case in INVARIANCE_CASES]


TILED_VARIANT_CASE = VariantAccuracyCase(
    "vanilla-tiled",
    "vanilla",
    "vanilla",
    forward_tolerance=(1e-7, 1e-5),
    input_gradient_tolerance=(1e-5, 1e-5),
    parameter_gradient_tolerance=(1e-5, 1e-5),
)

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", marks=pytest.mark.gpu, id="cuda"),
]


def _reference(num_heads: int = NUM_HEADS, use_efficient_fusion: bool = False) -> torch.nn.Module:
    """Return a vanilla module with active output weights."""
    torch.manual_seed(42)
    ref = AttentionPairBias(
        c_s=C_S,
        c_z=C_Z,
        num_heads=num_heads,
        use_efficient_fusion=use_efficient_fusion,
    ).eval()
    with torch.no_grad():
        for p in ref.parameters():
            if p.abs().sum() == 0:
                p.copy_(torch.randn_like(p) * 0.02)
    return ref


def _make_inputs(
    B: int,
    n: int,
    *,
    multiplicity: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create attention inputs with a nontrivial mask."""
    s = torch.randn(
        B * multiplicity, n, C_S, device=device, dtype=dtype, requires_grad=requires_grad
    )
    z = torch.randn(B, n, n, C_Z, device=device, dtype=dtype, requires_grad=requires_grad)
    mask = torch.ones(B, n, dtype=torch.bool, device=device)
    for b in range(B):
        pad_len = min((b + 1) * 3, n - 4)
        if pad_len > 0:
            mask[b, n - pad_len :] = False
    return s, z, mask


class TestImplementationsMatchReference:
    """Compare each implementation and dtype with its stated FP32 reference."""

    @pytest.mark.parametrize("case", ACCURACY_CASES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_forward(
        self,
        case: VariantAccuracyCase,
        device,
        num_heads,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if case.autocast_dtype is torch.bfloat16:
            if device != "cuda":
                pytest.skip("bfloat16 accuracy is tested on CUDA")
            if not torch.cuda.is_bf16_supported():
                pytest.skip("bfloat16 is not supported on this GPU")
        torch.manual_seed(0)
        s, z, mask = _make_inputs(1, N, device=device)

        ref = _reference(num_heads)
        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            **case.options,
        )
        module.load_state_dict(ref.state_dict())
        ref.eval().to(device)
        module.eval().to(device)

        with torch.no_grad(), set_matmul_precision("highest"):
            ref_out = ref(s, z, mask)
        assert ref_out.abs().max().item() > 0, "reference output is trivially zero"

        with (
            torch.no_grad(),
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type=device,
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            var_out = module(s, z, mask)

        fwd_atol, fwd_rtol = case.forward_tolerance
        torch.testing.assert_close(
            var_out.float(),
            ref_out.float(),
            atol=fwd_atol,
            rtol=fwd_rtol,
        )

    @pytest.mark.parametrize("case", ACCURACY_CASES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_backward(
        self,
        case: VariantAccuracyCase,
        device,
        num_heads,
    ) -> None:
        """Compare gradients with the row's FP32 reference."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if case.autocast_dtype is torch.bfloat16:
            if device != "cuda":
                pytest.skip("bfloat16 accuracy is tested on CUDA")
            if not torch.cuda.is_bf16_supported():
                pytest.skip("bfloat16 is not supported on this GPU")

        ref = _reference(num_heads)
        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            **case.options,
        )
        module.load_state_dict(ref.state_dict())
        ref.train().to(device)
        module.train().to(device)

        torch.manual_seed(0)
        s, z, mask = _make_inputs(1, N, device=device)
        ref_s, ref_z = (
            s.detach().clone().requires_grad_(True),
            z.detach().clone().requires_grad_(True),
        )
        var_s, var_z = (
            s.detach().clone().requires_grad_(True),
            z.detach().clone().requires_grad_(True),
        )

        with set_matmul_precision("highest"):
            ref(ref_s, ref_z, mask).sum().backward()
            ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}
            ref.zero_grad()

        with (
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type=device,
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            module(var_s, var_z, mask).sum().backward()
        grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
        assert grads.keys() == ref_grads.keys()
        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            var_s.grad,
            ref_s.grad,
            atol=input_atol,
            rtol=input_rtol,
            msg="input s grad",
        )
        torch.testing.assert_close(
            var_z.grad,
            ref_z.grad,
            atol=input_atol,
            rtol=input_rtol,
            msg="input z grad",
        )

        param_atol, param_rtol = case.parameter_gradient_tolerance
        for name in ref_grads:
            torch.testing.assert_close(
                ref_grads[name],
                grads[name],
                atol=param_atol,
                rtol=param_rtol,
                msg=f"grad {name}",
            )


FUSION_CASES = [pytest.param(case, id=case.name) for case in _FP32_ACCURACY_CASES]


class TestImplementationsWithFusionMatchReference:
    """Each fused implementation matches the fused vanilla reference."""

    @pytest.mark.parametrize("case", FUSION_CASES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_forward(
        self,
        case: VariantAccuracyCase,
        device,
        num_heads,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(0)
        s, z, mask = _make_inputs(1, N, device=device)

        ref = _reference(num_heads, use_efficient_fusion=True).to(device)

        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            use_efficient_fusion=True,
            **case.options,
        )
        module.load_state_dict(ref.state_dict())
        module.eval().to(device)

        with torch.no_grad(), set_matmul_precision("highest"):
            ref_out = ref(s, z, mask)
        assert ref_out.abs().max().item() > 0, "reference output is trivially zero"

        with torch.no_grad(), set_matmul_precision(case.matmul_precision):
            var_out = module(s, z, mask)

        fwd_atol, fwd_rtol = case.forward_tolerance
        torch.testing.assert_close(
            var_out,
            ref_out,
            atol=fwd_atol,
            rtol=fwd_rtol,
        )

    @pytest.mark.parametrize("case", FUSION_CASES)
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_backward(
        self,
        case: VariantAccuracyCase,
        device,
        num_heads,
    ) -> None:
        """Gradient accuracy of the fused variant vs the fused vanilla reference."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref = _reference(num_heads, use_efficient_fusion=True).to(device)

        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            use_efficient_fusion=True,
            **case.options,
        )
        module.load_state_dict(ref.state_dict())
        module.to(device)

        torch.manual_seed(0)
        s, z, mask = _make_inputs(1, N, device=device)
        ref_s, ref_z = (
            s.detach().clone().requires_grad_(True),
            z.detach().clone().requires_grad_(True),
        )
        var_s, var_z = (
            s.detach().clone().requires_grad_(True),
            z.detach().clone().requires_grad_(True),
        )

        with set_matmul_precision("highest"):
            ref(ref_s, ref_z, mask).sum().backward()
            ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters() if p.grad is not None}
            ref.zero_grad()

        with set_matmul_precision(case.matmul_precision):
            module(var_s, var_z, mask).sum().backward()
        grads = {n: p.grad for n, p in module.named_parameters() if p.grad is not None}
        assert grads.keys() == ref_grads.keys()
        grad_atol, grad_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            var_s.grad,
            ref_s.grad,
            atol=grad_atol,
            rtol=grad_rtol,
            msg="input s grad",
        )
        torch.testing.assert_close(
            var_z.grad,
            ref_z.grad,
            atol=grad_atol,
            rtol=grad_rtol,
            msg="input z grad",
        )

        for name in ref_grads:
            atol, rtol = case.parameter_gradient_tolerance
            if name in _FUSED_PARAM_NAMES:
                atol = max(atol, _FUSED_PARAM_ATOL)
            torch.testing.assert_close(
                ref_grads[name],
                grads[name],
                atol=atol,
                rtol=rtol,
                msg=f"grad {name}",
            )


# Measured worst-case batch-multiplicity drift at B=3, M=3 (A100):
#
#  variant        precision                      param_grad  s_grad   z_grad
#  vanilla        f64  cublas=highest             7e-14      1e-16    1e-17
#  vanilla        f32  cublas=highest  cpu        2.5e-5     3e-8     7.5e-9
#  vanilla        f32  cublas=highest  cuda       4.6e-5     2e-7     1.7e-8
#  vanilla        f32  cublas=high(tf32)          3.5e-3     5.4e-5   2.1e-5
#  vanilla        f32  autocast(bf16)             1.8e-1     2.1e-3   2.0e-4

_BATCH_MULTI_CONFIGS = [
    pytest.param(
        "vanilla",
        {},
        torch.float64,
        "highest",
        "cpu",
        None,
        1e-12,
        1e-12,
        1e-12,
        id="vanilla-f64-cpu",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float64,
        "highest",
        "cuda",
        None,
        1e-12,
        1e-12,
        1e-12,
        id="vanilla-f64-cuda",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float32,
        "highest",
        "cpu",
        None,
        2e-4,
        1e-6,
        5e-7,
        id="vanilla-f32-ieee-cpu",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float32,
        "highest",
        "cuda",
        None,
        1e-4,
        1e-5,
        5e-7,
        id="vanilla-f32-ieee-cuda",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float32,
        "high",
        "cuda",
        None,
        1e-2,
        2e-4,
        1e-4,
        id="vanilla-f32-cublas_tf32-cuda",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float32,
        "highest",
        "cuda",
        torch.bfloat16,
        5e-1,
        5e-3,
        5e-4,
        id="vanilla-autocast_bf16-cuda",
    ),
]


# The efficient conv2d fusion (use_efficient_fusion=True) is NOT batch-invariant on CUDA:
# cuDNN's 1×1 conv2d reduces differently per batch size, unlike the plain matmul used in the
# default path. Measured per-parameter at B=3, M=3 (CUDA fp32), the ONLY gradients that lose
# batch-invariance are the ones produced by the fused z-bias projection:
#   proj_z.weight ~3.5e-3, norm_z.weight ~5.2e-4, and the input-z grad ~2.9e-5.
# Every other parameter gradient (proj_qkv/proj_g/proj_o/norm_s/proj_q_bias) flows through
# batch-invariant matmuls and stays ≤6.1e-5, i.e. within the tight _BATCH_MULTI_CONFIGS
# param_atol. We therefore relax ONLY the fused params (via a dedicated fused_param_atol) and
# the input-z grad; the generic param_atol and s_atol stay tight. f64/CPU are bit-exact and
# bf16-autocast/cublas-TF32 already have ample headroom, so those configs are left unchanged.
_FUSED_PARAM_NAMES = frozenset({"norm_z.weight", "proj_z.weight"})

# Fused projection parameter gradients differ by up to 1.9e-4 on CUDA.
_FUSED_PARAM_ATOL = 5e-4

# Per-config (fused_param_atol, z_atol) overrides for the fusion variant. Configs absent here
# keep their _BATCH_MULTI_CONFIGS values (fused_param_atol falls back to param_atol).
_FUSION_TOL_OVERRIDES: dict[str, tuple[float, float]] = {
    "vanilla-f32-ieee-cuda": (5e-3, 5e-5),
}


def _with_fusion_tols(param: ParameterSet) -> ParameterSet:
    """Derive a fusion config from a _BATCH_MULTI_CONFIGS entry.

    Relaxes the input-z-grad tolerance (z_atol) and appends a trailing ``fused_param_atol``
    column used only for the fused-op params; ``param_atol``/``s_atol`` are left untouched.
    """
    values = list(param.values)  # [..., param_atol, s_atol, z_atol]
    param_atol, z_atol = values[-3], values[-1]
    fused_param_atol, z_override = _FUSION_TOL_OVERRIDES.get(param.id, (param_atol, z_atol))
    values[-1] = z_override
    values.append(fused_param_atol)
    return pytest.param(*values, id=param.id, marks=param.marks)


_BATCH_MULTI_FUSION_CONFIGS = [_with_fusion_tols(p) for p in _BATCH_MULTI_CONFIGS]


def _make_module(
    variant: str,
    variant_kwargs: dict,
    dtype: torch.dtype,
    device: str,
    use_efficient_fusion: bool = False,
):
    """Create a module from the registry with reference weights."""
    ref = _reference()
    module = AttentionPairBias(
        c_s=C_S,
        c_z=C_Z,
        num_heads=NUM_HEADS,
        use_efficient_fusion=use_efficient_fusion,
        **variant_kwargs,
    )
    module.load_state_dict(ref.state_dict())
    return module.to(device=device, dtype=dtype)


class TestBatchedMultiplicityMatchesSeparateRuns:
    """Full-attention APB (default no-fusion path): batched (B, M) forward/gradient matches
    B=1 M=1 ground truth.

    With ``use_efficient_fusion=False`` the pair-bias projection is a plain LayerNorm + matmul,
    which is numerically batch-invariant, so the tight ``_BATCH_MULTI_CONFIGS`` tolerances hold.
    The fusion variant is covered by :class:`TestFusedBatchedMultiplicityMatchesSeparateRuns`.
    """

    @pytest.mark.parametrize(
        "variant, variant_kwargs, dtype, cublas_precision, device, autocast_dtype,"
        " param_atol, s_atol, z_atol",
        _BATCH_MULTI_CONFIGS,
    )
    @pytest.mark.parametrize("B,M", [pytest.param(2, 3, id="batched-multiplicity")])
    def test_forward(
        self,
        variant: str,
        variant_kwargs: dict,
        dtype: torch.dtype,
        cublas_precision: str,
        device: str,
        autocast_dtype: torch.dtype | None,
        param_atol: float,
        s_atol: float,
        z_atol: float,
        B: int,
        M: int,
    ) -> None:
        del param_atol, z_atol
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_module(variant, variant_kwargs, dtype, device)

        s, z, mask = _make_inputs(B, N, multiplicity=M, device=device, dtype=dtype)

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )

        with torch.no_grad(), set_matmul_precision(cublas_precision), amp:
            out_batched = module(s, z, mask, multiplicity=M)
            outs_seq: list[torch.Tensor] = []
            for b in range(B):
                for m in range(M):
                    out_bm = module(
                        s[b * M + m : b * M + m + 1],
                        z[b : b + 1],
                        mask[b : b + 1],
                        multiplicity=1,
                    )
                    outs_seq.append(out_bm)
            out_seq = torch.cat(outs_seq, dim=0)

        assert out_batched.abs().max().item() > 0, "output is trivially zero"
        torch.testing.assert_close(out_batched, out_seq, atol=s_atol, rtol=0.0)

    @pytest.mark.parametrize(
        "variant, variant_kwargs, dtype, cublas_precision, device, autocast_dtype,"
        " param_atol, s_atol, z_atol",
        _BATCH_MULTI_CONFIGS,
    )
    @pytest.mark.parametrize("B,M", [pytest.param(2, 3, id="batched-multiplicity")])
    def test_backward(
        self,
        variant: str,
        variant_kwargs: dict,
        dtype: torch.dtype,
        cublas_precision: str,
        device: str,
        autocast_dtype: torch.dtype | None,
        param_atol: float,
        s_atol: float,
        z_atol: float,
        B: int,
        M: int,
    ) -> None:
        """Parameter gradients from a batched (B, M) call equal the sum of per-(b,m) gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_module(variant, variant_kwargs, dtype, device)

        s_data, z_data, mask = _make_inputs(B, N, multiplicity=M, device=device, dtype=dtype)

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )

        s_bat = s_data.clone().requires_grad_(True)
        z_bat = z_data.clone().requires_grad_(True)
        with set_matmul_precision(cublas_precision), amp:
            module(s_bat, z_bat, mask, multiplicity=M).sum().backward()
        bat_pg = {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
        bat_sg = s_bat.grad.clone()
        bat_zg = z_bat.grad.clone()
        module.zero_grad()

        seq_pg: dict[str, torch.Tensor] = {}
        seq_sg = torch.zeros_like(s_data)
        seq_zg = torch.zeros_like(z_data)
        for b in range(B):
            for m in range(M):
                t_bm = s_data[b * M + m : b * M + m + 1].clone().requires_grad_(True)
                z_b = z_data[b : b + 1].clone().requires_grad_(True)
                with set_matmul_precision(cublas_precision), amp:
                    module(t_bm, z_b, mask[b : b + 1], multiplicity=1).sum().backward()
                for n, p in module.named_parameters():
                    if p.grad is not None:
                        if n not in seq_pg:
                            seq_pg[n] = p.grad.clone()
                        else:
                            seq_pg[n] += p.grad
                seq_sg[b * M + m] = t_bm.grad.squeeze(0)
                seq_zg[b] += z_b.grad.squeeze(0)
                module.zero_grad()

        assert len(bat_pg) > 0, "no parameter gradients"
        for name in bat_pg:
            torch.testing.assert_close(
                bat_pg[name], seq_pg[name], atol=param_atol, rtol=0.0, msg=f"param grad {name}"
            )
        torch.testing.assert_close(bat_sg, seq_sg, atol=s_atol, rtol=0.0, msg="input s grad")
        torch.testing.assert_close(bat_zg, seq_zg, atol=z_atol, rtol=0.0, msg="input z grad")


class TestFusedBatchedMultiplicityMatchesSeparateRuns:
    """Same as :class:`TestFullAttentionBatchMultiplicity` but with the efficient conv2d
    fusion enabled (``use_efficient_fusion=True``).

    The fused ``_proj_z_fused`` folds the LayerNorm gain into a 1×1 ``F.conv2d``, which on
    CUDA is not numerically batch-invariant (cuDNN reduces differently per batch size). Only
    the gradients produced by that fused projection lose batch-invariance, so we relax the
    tolerance ONLY for the fused params (``fused_param_atol`` on ``norm_z``/``proj_z``) and the
    input-z grad; every other param and the input-s grad keep the tight tolerance.
    """

    @pytest.mark.parametrize(
        "variant, variant_kwargs, dtype, cublas_precision, device, autocast_dtype,"
        " param_atol, s_atol, z_atol, fused_param_atol",
        _BATCH_MULTI_FUSION_CONFIGS,
    )
    @pytest.mark.parametrize("B,M", [pytest.param(2, 3, id="batched-multiplicity")])
    def test_forward(
        self,
        variant: str,
        variant_kwargs: dict,
        dtype: torch.dtype,
        cublas_precision: str,
        device: str,
        autocast_dtype: torch.dtype | None,
        param_atol: float,
        s_atol: float,
        z_atol: float,
        fused_param_atol: float,
        B: int,
        M: int,
    ) -> None:
        del param_atol, z_atol, fused_param_atol
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_module(variant, variant_kwargs, dtype, device, use_efficient_fusion=True)

        s, z, mask = _make_inputs(B, N, multiplicity=M, device=device, dtype=dtype)

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )

        with torch.no_grad(), set_matmul_precision(cublas_precision), amp:
            out_batched = module(s, z, mask, multiplicity=M)
            outs_seq: list[torch.Tensor] = []
            for b in range(B):
                for m in range(M):
                    out_bm = module(
                        s[b * M + m : b * M + m + 1],
                        z[b : b + 1],
                        mask[b : b + 1],
                        multiplicity=1,
                    )
                    outs_seq.append(out_bm)
            out_seq = torch.cat(outs_seq, dim=0)

        assert out_batched.abs().max().item() > 0, "output is trivially zero"
        torch.testing.assert_close(out_batched, out_seq, atol=s_atol, rtol=0.0)

    @pytest.mark.parametrize(
        "variant, variant_kwargs, dtype, cublas_precision, device, autocast_dtype,"
        " param_atol, s_atol, z_atol, fused_param_atol",
        _BATCH_MULTI_FUSION_CONFIGS,
    )
    @pytest.mark.parametrize("B,M", [pytest.param(2, 3, id="batched-multiplicity")])
    def test_backward(
        self,
        variant: str,
        variant_kwargs: dict,
        dtype: torch.dtype,
        cublas_precision: str,
        device: str,
        autocast_dtype: torch.dtype | None,
        param_atol: float,
        s_atol: float,
        z_atol: float,
        fused_param_atol: float,
        B: int,
        M: int,
    ) -> None:
        """Parameter gradients from a batched (B, M) call equal the sum of per-(b,m) gradients."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_module(variant, variant_kwargs, dtype, device, use_efficient_fusion=True)

        s_data, z_data, mask = _make_inputs(B, N, multiplicity=M, device=device, dtype=dtype)

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )

        s_bat = s_data.clone().requires_grad_(True)
        z_bat = z_data.clone().requires_grad_(True)
        with set_matmul_precision(cublas_precision), amp:
            module(s_bat, z_bat, mask, multiplicity=M).sum().backward()
        bat_pg = {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
        bat_sg = s_bat.grad.clone()
        bat_zg = z_bat.grad.clone()
        module.zero_grad()

        seq_pg: dict[str, torch.Tensor] = {}
        seq_sg = torch.zeros_like(s_data)
        seq_zg = torch.zeros_like(z_data)
        for b in range(B):
            for m in range(M):
                t_bm = s_data[b * M + m : b * M + m + 1].clone().requires_grad_(True)
                z_b = z_data[b : b + 1].clone().requires_grad_(True)
                with set_matmul_precision(cublas_precision), amp:
                    module(t_bm, z_b, mask[b : b + 1], multiplicity=1).sum().backward()
                for n, p in module.named_parameters():
                    if p.grad is not None:
                        if n not in seq_pg:
                            seq_pg[n] = p.grad.clone()
                        else:
                            seq_pg[n] += p.grad
                seq_sg[b * M + m] = t_bm.grad.squeeze(0)
                seq_zg[b] += z_b.grad.squeeze(0)
                module.zero_grad()

        assert len(bat_pg) > 0, "no parameter gradients"
        for name in bat_pg:
            # Only the fused z-bias projection params (norm_z/proj_z) lose batch-invariance
            # under the conv2d fusion; all other params keep the tight param_atol.
            atol = fused_param_atol if name in _FUSED_PARAM_NAMES else param_atol
            torch.testing.assert_close(
                bat_pg[name], seq_pg[name], atol=atol, rtol=0.0, msg=f"param grad {name}"
            )
        torch.testing.assert_close(bat_sg, seq_sg, atol=s_atol, rtol=0.0, msg="input s grad")
        torch.testing.assert_close(bat_zg, seq_zg, atol=z_atol, rtol=0.0, msg="input z grad")


def _pad_sequence(
    s: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    pad_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append pad_len fully-masked positions to s, z, and mask.

    Args:
        s: [B*M, n, C_S]
        z: [B, n, n, C_Z]
        mask: [B, n] bool

    Returns:
        Padded (s, z, mask) with sequence length increased by pad_len.
    """
    BM, n, c_s = s.shape
    B = mask.shape[0]
    c_z = z.shape[-1]
    device, dtype = s.device, s.dtype

    # Pad s: append zeros along sequence dim
    s_padded = torch.cat([s, torch.zeros(BM, pad_len, c_s, device=device, dtype=dtype)], dim=1)

    # Pad z: expand from [B, n, n, c_z] to [B, n+pad_len, n+pad_len, c_z]
    z_padded = torch.zeros(B, n + pad_len, n + pad_len, c_z, device=device, dtype=dtype)
    z_padded[:, :n, :n, :] = z

    # Pad mask: append False
    mask_padded = torch.cat([mask, torch.zeros(B, pad_len, dtype=torch.bool, device=device)], dim=1)

    return s_padded, z_padded, mask_padded


class TestMaskedPadding:
    """Output for valid positions is unchanged when extra fully-masked positions are appended."""

    PAD_LEN = 7

    @pytest.mark.parametrize("B,M", [pytest.param(2, 2, id="batched-multiplicity")])
    def test_padding_invariance_vanilla_f64(self, B: int, M: int) -> None:
        """Vanilla f64 on CPU: padded output must be bitwise identical for original positions."""
        torch.manual_seed(42)
        module = _make_module("vanilla", {}, torch.float64, "cpu")

        s, z, mask = _make_inputs(B, N, multiplicity=M, device="cpu", dtype=torch.float64)

        # The standard path processes one copy at a time.
        outs_orig: list[torch.Tensor] = []
        outs_padded: list[torch.Tensor] = []
        with torch.no_grad(), set_matmul_precision("highest"):
            for b in range(B):
                for m in range(M):
                    s_bm = s[b * M + m : b * M + m + 1]
                    z_b = z[b : b + 1]
                    mask_b = mask[b : b + 1]
                    outs_orig.append(module(s_bm, z_b, mask_b, multiplicity=1))

                    s_p, z_p, mask_p = _pad_sequence(s_bm, z_b, mask_b, self.PAD_LEN)
                    outs_padded.append(module(s_p, z_p, mask_p, multiplicity=1))

        out_orig = torch.cat(outs_orig, dim=0)
        out_padded = torch.cat(outs_padded, dim=0)[:, :N, :]

        assert out_orig.shape == out_padded.shape
        # f64 with highest precision: expect bitwise equality
        torch.testing.assert_close(out_padded, out_orig, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("B,M", [pytest.param(2, 2, id="batched-multiplicity")])
    def test_padding_invariance_vanilla_f32(self, B: int, M: int) -> None:
        """Vanilla f32 on CPU: padded output matches for original positions."""
        torch.manual_seed(42)
        module = _make_module("vanilla", {}, torch.float32, "cpu")

        s, z, mask = _make_inputs(B, N, multiplicity=M, device="cpu", dtype=torch.float32)

        outs_orig: list[torch.Tensor] = []
        outs_padded: list[torch.Tensor] = []
        with torch.no_grad(), set_matmul_precision("highest"):
            for b in range(B):
                for m in range(M):
                    s_bm = s[b * M + m : b * M + m + 1]
                    z_b = z[b : b + 1]
                    mask_b = mask[b : b + 1]
                    outs_orig.append(module(s_bm, z_b, mask_b, multiplicity=1))

                    s_p, z_p, mask_p = _pad_sequence(s_bm, z_b, mask_b, self.PAD_LEN)
                    outs_padded.append(module(s_p, z_p, mask_p, multiplicity=1))

        out_orig = torch.cat(outs_orig, dim=0)
        out_padded = torch.cat(outs_padded, dim=0)[:, :N, :]

        assert out_orig.shape == out_padded.shape
        # f32 ieee: masking is deterministic, expect exact match
        torch.testing.assert_close(out_padded, out_orig, atol=0.0, rtol=0.0)

    @pytest.mark.gpu
    @pytest.mark.parametrize("case", INVARIANCE_CASES)
    def test_backward(self, gpu_device: torch.device, case: VariantAccuracyCase) -> None:
        """Appending fully masked tokens preserves valid input and parameter gradients."""
        reference = _make_module(
            case.reference_variant,
            case.options,
            torch.float32,
            str(gpu_device),
        )
        padded = _make_module(
            case.variant,
            case.options,
            torch.float32,
            str(gpu_device),
        )
        padded.load_state_dict(reference.state_dict())
        s, z, mask = _make_inputs(1, N, device=gpu_device)
        padded_s, padded_z, padded_mask = _pad_sequence(s, z, mask, self.PAD_LEN)
        reference_s = s.detach().clone().requires_grad_(True)
        reference_z = z.detach().clone().requires_grad_(True)
        padded_s = padded_s.detach().clone().requires_grad_(True)
        padded_z = padded_z.detach().clone().requires_grad_(True)
        output_gradient = torch.randn_like(reference_s)

        reference(reference_s, reference_z, mask).backward(output_gradient)
        padded(padded_s, padded_z, padded_mask)[:, :N].backward(output_gradient)

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            padded_s.grad[:, :N],
            reference_s.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        torch.testing.assert_close(
            padded_z.grad[:, :N, :N],
            reference_z.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        assert torch.count_nonzero(padded_s.grad[:, N:]) == 0
        assert torch.count_nonzero(padded_z.grad[:, N:, :]) == 0
        assert torch.count_nonzero(padded_z.grad[:, :, N:]) == 0
        reference_grads = {name: p.grad.clone() for name, p in reference.named_parameters()}
        parameter_atol, parameter_rtol = case.parameter_gradient_tolerance
        for name, parameter in padded.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference_grads[name],
                atol=parameter_atol,
                rtol=parameter_rtol,
            )


class TestMasking:
    """Masked keys and pairs do not affect valid outputs or gradients."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("case", INVARIANCE_CASES)
    def test_forward(self, gpu_device: torch.device, case: VariantAccuracyCase) -> None:
        module = _make_module(
            case.variant,
            case.options,
            torch.float32,
            str(gpu_device),
        )
        s, z, mask = _make_inputs(1, N, device=gpu_device)
        changed_s, changed_z = s.clone(), z.clone()
        changed_s[:, ~mask[0]] = 1_000.0
        changed_z[:, ~mask[0], :, :] = 1_000.0
        changed_z[:, :, ~mask[0], :] = 1_000.0

        with torch.no_grad():
            expected = module(s, z, mask)
            actual = module(changed_s, changed_z, mask)

        assert mask.any() and (~mask).any()
        forward_atol, forward_rtol = case.forward_tolerance
        torch.testing.assert_close(
            actual[:, mask[0]],
            expected[:, mask[0]],
            atol=forward_atol,
            rtol=forward_rtol,
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        "variant",
        [
            "vanilla",
        ],
    )
    def test_backward(self, gpu_device: torch.device, variant: str) -> None:
        module = _make_module(
            variant,
            {},
            torch.float32,
            str(gpu_device),
        )
        s, z, mask = _make_inputs(1, N, device=gpu_device)
        s.requires_grad_(True)
        z.requires_grad_(True)
        module(s, z, mask)[:, mask[0]].sum().backward()

        assert torch.count_nonzero(s.grad[:, ~mask[0]]) == 0
        assert torch.count_nonzero(z.grad[:, ~mask[0], :, :]) == 0
        assert torch.count_nonzero(z.grad[:, :, ~mask[0], :]) == 0
        assert torch.count_nonzero(s.grad[:, mask[0]]) > 0
        assert any(parameter.grad is not None for parameter in module.parameters())


class TestTiledAttentionMatchesUntiled:
    """Query-tiled vanilla must match untiled for any chunk size (including remainder)."""

    N_TILING = 100

    @pytest.mark.parametrize(
        "tile_size",
        [pytest.param(s, id=f"tile{s}") for s in (1, 16, 33, 64, 83)],
    )
    def test_forward(self, tile_size: int) -> None:
        case = TILED_VARIANT_CASE
        ref = _reference()
        untiled = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=NUM_HEADS, tile_size=None)
        tiled = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=NUM_HEADS, tile_size=tile_size)
        untiled.load_state_dict(ref.state_dict())
        tiled.load_state_dict(ref.state_dict())
        untiled.eval()
        tiled.eval()

        torch.manual_seed(0)
        s, z, mask = _make_inputs(1, self.N_TILING)

        with torch.no_grad(), set_matmul_precision("highest"):
            out_untiled = untiled(s, z, mask)
            out_tiled = tiled(s, z, mask)

        assert out_untiled.abs().max().item() > 0, "untiled output is trivially zero"
        # Tiling changes reduction order → small float32 rounding drift for non-dividing sizes.
        forward_atol, forward_rtol = case.forward_tolerance
        torch.testing.assert_close(
            out_tiled,
            out_untiled,
            atol=forward_atol,
            rtol=forward_rtol,
        )

    @pytest.mark.parametrize(
        "tile_size",
        [pytest.param(s, id=f"tile{s}") for s in (1, 33, 83)],
    )
    def test_backward(self, tile_size: int) -> None:
        case = TILED_VARIANT_CASE
        reference = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=NUM_HEADS, tile_size=None)
        tiled = AttentionPairBias(c_s=C_S, c_z=C_Z, num_heads=NUM_HEADS, tile_size=tile_size)
        tiled.load_state_dict(reference.state_dict())
        s, z, mask = _make_inputs(1, self.N_TILING)
        reference_s = s.detach().clone().requires_grad_(True)
        reference_z = z.detach().clone().requires_grad_(True)
        tiled_s = s.detach().clone().requires_grad_(True)
        tiled_z = z.detach().clone().requires_grad_(True)
        output_gradient = torch.randn_like(reference_s)

        reference(reference_s, reference_z, mask).backward(output_gradient)
        tiled(tiled_s, tiled_z, mask).backward(output_gradient)

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            tiled_s.grad,
            reference_s.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        torch.testing.assert_close(
            tiled_z.grad,
            reference_z.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        reference_grads = {name: p.grad.clone() for name, p in reference.named_parameters()}
        parameter_atol, parameter_rtol = case.parameter_gradient_tolerance
        for name, parameter in tiled.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference_grads[name],
                atol=parameter_atol,
                rtol=parameter_rtol,
            )


class TestFusedProjectionMatchesReference:
    """The bias-free ``_proj_z_fused`` conv2d path must equal LayerNorm(no bias) + Linear.

    When ``use_bias_in_layer_norm=False`` the pair-bias projection folds the LayerNorm
    gain γ into a 1×1 ``F.conv2d`` kernel (see ``_proj_z_fused``). This test guards that
    fusion — forward output and parameter/input gradients — against an explicit, unfused
    reference computed with ``nn.LayerNorm(bias=False)`` followed by the linear projection.

    Note: on CUDA, cuDNN's 1×1 conv2d and matmul use different reduction algorithms, so
    the two agree only to ~3e-4 relative (bit-exact on CPU). This is also why the bias-free
    path is not numerically batch-invariant.
    """

    @pytest.mark.parametrize(
        "variant",
        [
            "vanilla",
        ],
    )
    def test_efficient_fusion_with_layernorm_bias_raises(self, variant: str) -> None:
        # The γ-fold underlying the efficient fusion cannot absorb a LayerNorm bias.
        with pytest.raises(ValueError, match="use_efficient_fusion"):
            AttentionPairBias(
                c_s=C_S,
                c_z=C_Z,
                num_heads=NUM_HEADS,
                use_efficient_fusion=True,
                use_bias_in_layer_norm=True,
            )

    @pytest.mark.parametrize(
        "precision",
        ["highest", "high", "medium"],
        ids=["fp32", "tf32", "tf32-medium"],
    )
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_conv_fusion_matches_layernorm_no_bias(
        self, num_heads: int, device: str, precision: str
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # float32 matmul precision (TF32) only affects CUDA; on CPU non-"highest" is redundant.
        if device == "cpu" and precision != "highest":
            pytest.skip("float32 matmul precision does not change CPU results")

        torch.manual_seed(42)
        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            use_bias_in_layer_norm=False,
            use_efficient_fusion=True,
        )
        # Non-trivial γ (LayerNorm gain) and projection weights so the fold is exercised.
        with torch.no_grad():
            module.norm_z.weight.copy_(torch.randn_like(module.norm_z.weight))
            module.proj_z.weight.copy_(torch.randn_like(module.proj_z.weight) * 0.1)
        module = module.to(device)

        # Unfused reference with independent parameters (so gradients don't collide with the
        # fused module): LayerNorm without bias (γ = norm_z.weight) then the linear proj.
        ref_ln = torch.nn.LayerNorm(C_Z, bias=False).to(device)
        ref_proj = torch.nn.Linear(C_Z, num_heads, bias=False).to(device)
        with torch.no_grad():
            ref_ln.weight.copy_(module.norm_z.weight)
            ref_proj.weight.copy_(module.proj_z.weight)

        torch.manual_seed(0)
        z_base = torch.randn(2, N, N, C_Z, device=device)  # [B, N, N, c_z]

        with set_matmul_precision(precision):
            z_fused = z_base.clone().requires_grad_(True)
            fused = module._proj_z_fused(z_fused)  # [B, H, N, N]

            z_ref = z_base.clone().requires_grad_(True)
            reference = ref_proj(ref_ln(z_ref)).permute(0, 3, 1, 2)  # [B, H, N, N]

            assert fused.shape == reference.shape == (2, num_heads, N, N)
            assert fused.abs().max().item() > 0, "fused projection is trivially zero"
            # Fusion is algebraically exact (see fp64 companion). All drift here is pure
            # rounding: conv2d and matmul use different reduction orders, and the exact orders
            # depend on thread count / kernel dispatch, so tolerances are set generously to
            # cover machine-to-machine variation (observed CI drift ~6e-5 abs / ~2e-4 rel on
            # CPU; CUDA cuDNN 1×1 conv2d up to ~1.4e-3 / ~2.4e-3 TF32).
            fwd_atol, fwd_rtol = (3e-3, 1e-3) if device == "cuda" else (5e-4, 2e-3)
            torch.testing.assert_close(fused, reference, atol=fwd_atol, rtol=fwd_rtol)

            # Gradients: same conv-vs-matmul drift, amplified by reductions. Parameter grads
            # are large-magnitude reductions (~O(100)); CUDA reaches ~5e-2 abs / ~1e-4 rel and
            # CPU drift is thread-dependent (observed ~6e-5 abs / ~2e-4 rel — set generously).
            # Input-z gradients drift by up to 6.7e-4 across supported kernels.
            fused.sum().backward()
            reference.sum().backward()

            param_atol, param_rtol = (6e-2, 1e-3) if device == "cuda" else (1e-3, 5e-3)
            torch.testing.assert_close(
                module.norm_z.weight.grad, ref_ln.weight.grad, atol=param_atol, rtol=param_rtol
            )
            torch.testing.assert_close(
                module.proj_z.weight.grad, ref_proj.weight.grad, atol=param_atol, rtol=param_rtol
            )
            zgrad_atol, zgrad_rtol = (1e-3, 0.0) if device == "cuda" else (1e-5, 1e-4)
            torch.testing.assert_close(z_fused.grad, z_ref.grad, atol=zgrad_atol, rtol=zgrad_rtol)

    @pytest.mark.parametrize("num_heads", NUM_HEADS_VARIANTS, ids=lambda h: f"H{h}")
    def test_conv_fusion_matches_layernorm_no_bias_fp64_cpu(self, num_heads: int) -> None:
        # In float64 on CPU the conv2d fold and the matmul reference agree to machine
        # precision (forward ~1e-16, gradients bit-exact), confirming the fusion is
        # algebraically exact — the fp32/TF32 drift above is purely rounding.
        torch.manual_seed(42)
        module = AttentionPairBias(
            c_s=C_S,
            c_z=C_Z,
            num_heads=num_heads,
            use_bias_in_layer_norm=False,
            use_efficient_fusion=True,
        )
        # Non-trivial γ (LayerNorm gain) and projection weights so the fold is exercised.
        with torch.no_grad():
            module.norm_z.weight.copy_(torch.randn_like(module.norm_z.weight))
            module.proj_z.weight.copy_(torch.randn_like(module.proj_z.weight) * 0.1)
        module = module.double()

        # Unfused reference with independent parameters (so gradients don't collide).
        ref_ln = torch.nn.LayerNorm(C_Z, bias=False).double()
        ref_proj = torch.nn.Linear(C_Z, num_heads, bias=False).double()
        with torch.no_grad():
            ref_ln.weight.copy_(module.norm_z.weight)
            ref_proj.weight.copy_(module.proj_z.weight)

        torch.manual_seed(0)
        z_base = torch.randn(2, N, N, C_Z, dtype=torch.float64)  # [B, N, N, c_z]

        z_fused = z_base.clone().requires_grad_(True)
        fused = module._proj_z_fused(z_fused)  # [B, H, N, N]

        z_ref = z_base.clone().requires_grad_(True)
        reference = ref_proj(ref_ln(z_ref)).permute(0, 3, 1, 2)  # [B, H, N, N]

        assert fused.shape == reference.shape == (2, num_heads, N, N)
        assert fused.abs().max().item() > 0, "fused projection is trivially zero"
        torch.testing.assert_close(fused, reference, atol=1e-12, rtol=1e-10)

        fused.sum().backward()
        reference.sum().backward()
        torch.testing.assert_close(
            module.norm_z.weight.grad, ref_ln.weight.grad, atol=1e-12, rtol=1e-10
        )
        torch.testing.assert_close(
            module.proj_z.weight.grad, ref_proj.weight.grad, atol=1e-12, rtol=1e-10
        )
        torch.testing.assert_close(z_fused.grad, z_ref.grad, atol=1e-12, rtol=1e-10)
