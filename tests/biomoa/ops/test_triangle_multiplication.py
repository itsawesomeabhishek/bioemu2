"""Tests TriangleMultiplication implementations."""

import pytest
import torch

from bioemu2.biomoa.ops.triangle_multiplication import REGISTRY
from bioemu2.biomoa.utils.strategy import make
from tests.conftest import set_matmul_precision

from ._accuracy import VariantAccuracyCase

DIM = 64
N = 16
# cuEquivariance's triangle_multiplicative_update falls back to PyTorch in eager
# mode for N <= CUEQ_TRIMUL_FALLBACK_THRESHOLD (default 100); exceed it to test
# the kernel. cueq_tiled calls the fused ops directly and has no such threshold.
CUEQ_N = 128

VARIANT_CASES = [
    VariantAccuracyCase(
        "cueq-ieee",
        "cueq",
        "full",
        options={"n_tokens": CUEQ_N},
        # Parameter gradients sum over N² pairs, so their scale and absolute
        # error grow with N (max |grad| ~3e2 at N=128).
        forward_tolerance=(2e-6, 0.0),
        input_gradient_tolerance=(2e-5, 0.0),
        parameter_gradient_tolerance=(3e-4, 0.0),
    ),
    VariantAccuracyCase(
        "cueq-tiled-default-ieee",
        "cueq_tiled",
        "full",
        forward_tolerance=(1e-6, 0.0),
        input_gradient_tolerance=(1e-6, 0.0),
        parameter_gradient_tolerance=(3e-5, 0.0),
    ),
    VariantAccuracyCase(
        "cueq-tiled-small-ieee",
        "cueq_tiled",
        "full",
        options={"tile_size": 8},
        forward_tolerance=(1e-6, 0.0),
        input_gradient_tolerance=(1e-6, 0.0),
        parameter_gradient_tolerance=(3e-5, 0.0),
    ),
    # BF16 input quantization and accumulation can shift input gradients by ~0.016
    # relative to the FP32 reference on CUDA.
    VariantAccuracyCase(
        "full-bf16",
        "full",
        "full",
        autocast_dtype=torch.bfloat16,
        forward_tolerance=(2e-2, 2e-2),
        input_gradient_tolerance=(2e-2, 2e-2),
        parameter_gradient_tolerance=(3e-1, 2e-2),
    ),
    VariantAccuracyCase(
        "cueq-bf16",
        "cueq",
        "full",
        dtype=torch.bfloat16,
        options={"n_tokens": CUEQ_N},
        forward_tolerance=(2e-2, 2e-2),
        input_gradient_tolerance=(2e-2, 2e-2),
        parameter_gradient_tolerance=(3.0, 2e-2),
    ),
    VariantAccuracyCase(
        "cueq-tiled-bf16",
        "cueq_tiled",
        "full",
        autocast_dtype=torch.bfloat16,
        forward_tolerance=(2e-2, 2e-2),
        input_gradient_tolerance=(2e-2, 2e-2),
        parameter_gradient_tolerance=(3e-1, 2e-2),
    ),
]
VARIANT_CASES = [pytest.param(case, id=case.name) for case in VARIANT_CASES]

PADDING_INVARIANCE_CASES = [
    VariantAccuracyCase(
        "full-padding",
        "full",
        "full",
        options={"device": "cpu"},
        # Padding changes the order of floating-point reductions, so a few
        # parameter-gradient elements land just outside a tight absolute bound.
        # A small relative tolerance absorbs this without hiding real regressions.
        parameter_gradient_tolerance=(1e-4, 1e-3),
    ),
    VariantAccuracyCase(
        "cueq-padding",
        "cueq",
        "cueq",
        options={"n_tokens": CUEQ_N, "device": "cuda"},
        parameter_gradient_tolerance=(1e-4, 1e-3),
    ),
    VariantAccuracyCase(
        "cueq-tiled-padding",
        "cueq_tiled",
        "cueq_tiled",
        options={"n_tokens": CUEQ_N, "device": "cuda"},
        parameter_gradient_tolerance=(1e-4, 1e-3),
    ),
]
PADDING_INVARIANCE_CASES = [
    pytest.param(
        case,
        id=case.name,
        marks=pytest.mark.gpu if case.options["device"] == "cuda" else (),
    )
    for case in PADDING_INVARIANCE_CASES
]
PADDING = 4


def _make_trimul(
    variant: str = "full",
    *,
    mode: str = "outgoing",
    tile_size: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    torch.manual_seed(42)
    module = make(
        REGISTRY,
        variant,
        dim=DIM,
        mode=mode,
        tile_size=tile_size,
        matmul_precision="ieee",
    ).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.02)
    return module.to(device=device, dtype=dtype)


def _inputs(
    batch_size: int = 1,
    n_tokens: int = N,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(batch_size, n_tokens, n_tokens, DIM, device=device, dtype=dtype)
    mask = torch.rand(batch_size, n_tokens, n_tokens, device=device) > 0.25
    return x, mask


def _parameter_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _pad(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = x.shape[1]
    # Non-zero padding values ensure invariance comes from the mask.
    padded_x = x.new_full((1, n + PADDING, n + PADDING, DIM), 1_000.0)
    padded_x[:, :n, :n] = x
    padded_mask = mask.new_zeros(1, n + PADDING, n + PADDING)
    padded_mask[:, :n, :n] = mask
    return padded_x, padded_mask


@pytest.mark.gpu
class TestOptimizedImplementationMatchesReference:
    """Each optimized implementation matches the full reference."""

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    @pytest.mark.parametrize("case", VARIANT_CASES)
    def test_forward(
        self,
        gpu_device: torch.device,
        mode: str,
        case: VariantAccuracyCase,
    ) -> None:
        if (
            case.dtype is torch.bfloat16 or case.autocast_dtype is torch.bfloat16
        ) and not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 is not supported on this GPU")
        if case.variant != "full" or case.reference_variant != "full":
            pytest.importorskip("cuequivariance_torch")
        tile_size = case.options.get("tile_size")
        reference = _make_trimul(
            case.reference_variant,
            mode=mode,
            tile_size=tile_size,
            device=gpu_device,
            dtype=case.reference_dtype,
        )
        candidate = _make_trimul(
            case.variant,
            mode=mode,
            tile_size=tile_size,
            device=gpu_device,
            dtype=case.dtype,
        )
        candidate.load_state_dict(reference.state_dict())
        reference_x, mask = _inputs(
            n_tokens=case.options.get("n_tokens", N),
            device=gpu_device,
            dtype=case.reference_dtype,
        )
        candidate_x = reference_x.to(case.dtype)

        with torch.no_grad(), set_matmul_precision("highest"):
            expected = reference(reference_x, mask)
        with (
            torch.no_grad(),
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type="cuda",
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            actual = candidate(candidate_x, mask)

        assert expected.abs().max().item() > 0, "zero-initialized output projection was not reset"
        fwd_atol, fwd_rtol = case.forward_tolerance
        torch.testing.assert_close(actual.float(), expected.float(), atol=fwd_atol, rtol=fwd_rtol)

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    @pytest.mark.parametrize("case", VARIANT_CASES)
    def test_backward(
        self,
        gpu_device: torch.device,
        mode: str,
        case: VariantAccuracyCase,
    ) -> None:
        if (
            case.dtype is torch.bfloat16 or case.autocast_dtype is torch.bfloat16
        ) and not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 is not supported on this GPU")
        if case.variant != "full" or case.reference_variant != "full":
            pytest.importorskip("cuequivariance_torch")
        tile_size = case.options.get("tile_size")
        reference = _make_trimul(
            case.reference_variant,
            mode=mode,
            tile_size=tile_size,
            device=gpu_device,
            dtype=case.reference_dtype,
        ).train()
        candidate = _make_trimul(
            case.variant,
            mode=mode,
            tile_size=tile_size,
            device=gpu_device,
            dtype=case.dtype,
        ).train()
        candidate.load_state_dict(reference.state_dict())
        x, mask = _inputs(
            n_tokens=case.options.get("n_tokens", N),
            device=gpu_device,
            dtype=case.reference_dtype,
        )
        output_gradient = torch.randn_like(x)
        reference_x = x.clone().requires_grad_(True)
        candidate_x = x.to(case.dtype).requires_grad_(True)

        with set_matmul_precision("highest"):
            reference(reference_x, mask).backward(output_gradient)
        with (
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type="cuda",
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            candidate(candidate_x, mask).backward(output_gradient.to(case.dtype))

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            candidate_x.grad.float(),
            reference_x.grad.float(),
            atol=input_atol,
            rtol=input_rtol,
        )
        reference_grads = _parameter_grads(reference)
        candidate_grads = _parameter_grads(candidate)
        assert candidate_grads.keys() == reference_grads.keys()
        param_atol, param_rtol = case.parameter_gradient_tolerance
        for name in reference_grads:
            torch.testing.assert_close(
                candidate_grads[name].float(),
                reference_grads[name].float(),
                atol=param_atol,
                rtol=param_rtol,
                msg=f"parameter gradient {name}",
            )


class TestMaskedPadding:
    """Masked padding does not affect valid outputs or gradients."""

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    @pytest.mark.parametrize("case", PADDING_INVARIANCE_CASES)
    def test_forward(self, mode: str, case: VariantAccuracyCase) -> None:
        device = case.options["device"]
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if case.variant != "full":
            pytest.importorskip("cuequivariance_torch")
        n = case.options.get("n_tokens", N)
        module = _make_trimul(case.variant, mode=mode, device=device)
        x, mask = _inputs(n_tokens=n, device=device)
        padded_x, padded_mask = _pad(x, mask)

        with torch.no_grad(), set_matmul_precision("highest"):
            expected = module(x, mask)
            actual = module(padded_x, padded_mask)[:, :n, :n]

        forward_atol, forward_rtol = case.forward_tolerance
        torch.testing.assert_close(
            actual,
            expected,
            atol=forward_atol,
            rtol=forward_rtol,
        )

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    @pytest.mark.parametrize("case", PADDING_INVARIANCE_CASES)
    def test_backward(self, mode: str, case: VariantAccuracyCase) -> None:
        device = case.options["device"]
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if case.variant != "full":
            pytest.importorskip("cuequivariance_torch")
        n = case.options.get("n_tokens", N)
        reference = _make_trimul(case.reference_variant, mode=mode, device=device)
        padded = _make_trimul(case.variant, mode=mode, device=device)
        padded.load_state_dict(reference.state_dict())
        x, mask = _inputs(n_tokens=n, device=device)
        padded_x, padded_mask = _pad(x, mask)
        x = x.requires_grad_(True)
        padded_x = padded_x.requires_grad_(True)
        output_gradient = torch.randn(1, n, n, DIM, device=device)

        with set_matmul_precision("highest"):
            reference(x, mask).backward(output_gradient)
            padded(padded_x, padded_mask)[:, :n, :n].backward(output_gradient)

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            padded_x.grad[:, :n, :n],
            x.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        assert torch.count_nonzero(padded_x.grad[:, n:]) == 0
        assert torch.count_nonzero(padded_x.grad[:, :, n:]) == 0
        parameter_atol, parameter_rtol = case.parameter_gradient_tolerance
        for name, expected in _parameter_grads(reference).items():
            # The longer input changes the order of floating-point sums.
            torch.testing.assert_close(
                _parameter_grads(padded)[name],
                expected,
                atol=parameter_atol,
                rtol=parameter_rtol,
            )


class TestMasking:
    """Masked pairs do not affect valid outputs or gradients."""

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    def test_forward(self, mode: str) -> None:
        module = _make_trimul(mode=mode)
        x, mask = _inputs()
        changed = x.clone()
        changed[~mask] = 1_000.0

        with torch.no_grad():
            expected = module(x, mask)
            actual = module(changed, mask)

        assert mask.any() and (~mask).any()
        torch.testing.assert_close(actual[mask], expected[mask], atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("mode", ["outgoing", "incoming"])
    def test_backward(self, mode: str) -> None:
        module = _make_trimul(mode=mode)
        x, mask = _inputs()
        x.requires_grad_(True)
        module(x, mask)[mask].sum().backward()

        assert torch.count_nonzero(x.grad[~mask]) == 0
        assert torch.count_nonzero(x.grad[mask]) > 0
        assert _parameter_grads(module)
