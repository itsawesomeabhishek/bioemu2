"""Tests OuterProductMean implementations."""

import math

import pytest
import torch

from bioemu2.biomoa.ops.outer_product_mean import REGISTRY
from bioemu2.biomoa.utils.strategy import make
from tests.conftest import set_matmul_precision

from ._accuracy import VariantAccuracyCase

C_IN = 64
C_HIDDEN = 32
C_OUT = 128
N = 37
S = 300
BATCH_SIZE = 2

VARIANT_CASES = [
    # A100 bf16 forward max error against fp32 was 2.1e-2 for these seeds.
    # Gradients allow 1% relative error plus an absolute floor for cancellation near zero.
    # After the relative allowance, max residual errors were 3.86e-3 (input)
    # and 2.02 (parameters).
    VariantAccuracyCase(
        "vanilla-bf16",
        "vanilla",
        "vanilla",
        autocast_dtype=torch.bfloat16,
        options={"tile_size": None},
        forward_tolerance=(2.5e-2, 0.0),  # atl, rtol
        input_gradient_tolerance=(4e-3, 1e-2),
        parameter_gradient_tolerance=(2.1, 1e-2),
        # atol is large because the gradient is large
    ),
    VariantAccuracyCase(
        "triton-default-tile-ieee",
        "triton_memory_saving",
        "vanilla",
        options={"tile_size": None},
        forward_tolerance=(1e-5, 0.0),
        inference_only=True,
    ),
    VariantAccuracyCase(
        "triton-small-tile-odd-channels-ieee",
        "triton_memory_saving",
        "vanilla",
        options={"tile_size": 8, "c_hidden": 33, "c_out": 35},
        forward_tolerance=(1e-5, 0.0),
        inference_only=True,
    ),
    VariantAccuracyCase(
        "triton-default-tile-tf32",
        "triton_memory_saving",
        "vanilla",
        matmul_precision="high",
        options={"tile_size": None, "matmul_precision": "tf32"},
        forward_tolerance=(1e-2, 0.0),
        inference_only=True,
    ),
    VariantAccuracyCase(
        "triton-default-tile-bf16",
        "triton_memory_saving",
        "vanilla",
        autocast_dtype=torch.bfloat16,
        options={"tile_size": None},
        forward_tolerance=(2.5e-2, 0.0),
        inference_only=True,
    ),
]
VARIANT_CASES = [pytest.param(case, id=case.name) for case in VARIANT_CASES]


def _make_opm(
    variant: str = "vanilla",
    *,
    tile_size: int | None = None,
    c_hidden: int = C_HIDDEN,
    c_out: int = C_OUT,
    matmul_precision: str = "ieee",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    torch.manual_seed(42)
    module = make(
        REGISTRY,
        variant,
        c_in=C_IN,
        c_hidden=c_hidden,
        c_out=c_out,
        tile_size=tile_size,
        matmul_precision=matmul_precision,
    ).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            std = 1 / math.sqrt(parameter.shape[-1]) if parameter.ndim == 2 else 1.0
            parameter.normal_(0, std)
    return module.to(device=device, dtype=dtype)


def _inputs(
    batch_size: int = 1,
    n_tokens: int = N,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    msa = torch.randn(batch_size, S, n_tokens, C_IN, device=device, dtype=dtype)
    mask = torch.ones(batch_size, S, n_tokens, device=device, dtype=torch.bool)
    mask[:, 1, -3:] = False
    mask[0, -10:] = False
    return msa, mask


def _parameter_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


@pytest.mark.gpu
class TestOptimizedImplementationMatchesReference:
    """Each optimized implementation matches the PyTorch reference."""

    @pytest.mark.parametrize("case", VARIANT_CASES)
    def test_forward(
        self,
        gpu_device: torch.device,
        case: VariantAccuracyCase,
    ) -> None:
        reference = _make_opm(
            case.reference_variant,
            c_hidden=case.options.get("c_hidden", C_HIDDEN),
            c_out=case.options.get("c_out", C_OUT),
            device=gpu_device,
            dtype=case.reference_dtype,
        )
        candidate = _make_opm(
            case.variant,
            tile_size=case.options["tile_size"],
            c_hidden=case.options.get("c_hidden", C_HIDDEN),
            c_out=case.options.get("c_out", C_OUT),
            matmul_precision=case.options.get("matmul_precision", "ieee"),
            device=gpu_device,
            dtype=case.dtype,
        )
        candidate.load_state_dict(reference.state_dict())
        reference_msa, mask = _inputs(BATCH_SIZE, device=gpu_device, dtype=case.reference_dtype)
        candidate_msa = reference_msa.to(case.dtype)

        with torch.no_grad(), set_matmul_precision("highest"):
            fp32_reference = reference(reference_msa, mask)
        with (
            torch.no_grad(),
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type="cuda",
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            actual = candidate(candidate_msa, mask)

        assert fp32_reference.abs().max().item() > 0, (
            "zero-initialized output projection was not reset"
        )
        fwd_atol, fwd_rtol = case.forward_tolerance
        torch.testing.assert_close(
            actual.float(), fp32_reference.float(), atol=fwd_atol, rtol=fwd_rtol
        )

    @pytest.mark.parametrize("case", VARIANT_CASES)
    def test_backward(
        self,
        gpu_device: torch.device,
        case: VariantAccuracyCase,
    ) -> None:
        if case.inference_only:
            pytest.skip("Skipping backward test for inference-only case.")

        reference = _make_opm(
            case.reference_variant,
            c_hidden=case.options.get("c_hidden", C_HIDDEN),
            c_out=case.options.get("c_out", C_OUT),
            device=gpu_device,
            dtype=case.reference_dtype,
        )
        candidate = _make_opm(
            case.variant,
            tile_size=case.options["tile_size"],
            c_hidden=case.options.get("c_hidden", C_HIDDEN),
            c_out=case.options.get("c_out", C_OUT),
            matmul_precision=case.options.get("matmul_precision", "ieee"),
            device=gpu_device,
            dtype=case.dtype,
        )
        candidate.load_state_dict(reference.state_dict())
        msa, mask = _inputs(BATCH_SIZE, device=gpu_device, dtype=case.reference_dtype)
        output_gradient = torch.randn(
            BATCH_SIZE,
            N,
            N,
            case.options.get("c_out", C_OUT),
            device=gpu_device,
            dtype=case.reference_dtype,
        )

        reference_msa = msa.clone().requires_grad_(True)
        candidate_msa = msa.to(case.dtype).requires_grad_(True)
        with set_matmul_precision("highest"):
            reference(reference_msa, mask).backward(output_gradient)
        with (
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type="cuda",
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            candidate(candidate_msa, mask).backward(output_gradient.to(case.dtype))

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            candidate_msa.grad.float(),
            reference_msa.grad.float(),
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
    """Appending masked tokens preserves valid outputs and gradients."""

    def test_forward(self) -> None:
        reference = _make_opm(dtype=torch.float64)
        msa, mask = _inputs(dtype=torch.float64)
        padded_msa = torch.cat([msa, torch.randn(1, S, 3, C_IN, dtype=torch.float64)], dim=2)
        padded_mask = torch.cat([mask, torch.zeros(1, S, 3, dtype=torch.bool)], dim=2)

        with torch.no_grad():
            expected = reference(msa, mask)
            actual = reference(padded_msa, padded_mask)[:, :N, :N]

        # Padding changes the order of floating-point sums on CPU.
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=0.0)

    def test_backward(self) -> None:
        reference = _make_opm(dtype=torch.float64)
        padded = _make_opm(dtype=torch.float64)
        padded.load_state_dict(reference.state_dict())
        msa, mask = _inputs(dtype=torch.float64)
        padded_msa = torch.cat([msa, torch.randn(1, S, 3, C_IN, dtype=torch.float64)], dim=2)
        padded_mask = torch.cat([mask, torch.zeros(1, S, 3, dtype=torch.bool)], dim=2)
        msa = msa.requires_grad_(True)
        padded_msa = padded_msa.requires_grad_(True)
        output_gradient = torch.randn(1, N, N, C_OUT, dtype=torch.float64)

        reference(msa, mask).backward(output_gradient)
        padded(padded_msa, padded_mask)[:, :N, :N].backward(output_gradient)

        # Padding changes the order of floating-point sums on CPU.
        torch.testing.assert_close(padded_msa.grad[:, :, :N], msa.grad, atol=2e-10, rtol=0.0)
        assert torch.count_nonzero(padded_msa.grad[:, :, N:]) == 0
        for name, expected in _parameter_grads(reference).items():
            torch.testing.assert_close(
                _parameter_grads(padded)[name], expected, atol=1e-8, rtol=0.0
            )


class TestMasking:
    """Masked MSA values do not affect outputs or gradients."""

    def test_forward(self) -> None:
        module = _make_opm(dtype=torch.float64)
        msa, mask = _inputs(dtype=torch.float64)
        changed = msa.clone()
        changed[~mask] = 1_000.0

        with torch.no_grad():
            expected = module(msa, mask)
            actual = module(changed, mask)

        assert mask.any() and (~mask).any()
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_backward(self) -> None:
        module = _make_opm(dtype=torch.float64)
        msa, mask = _inputs(dtype=torch.float64)
        msa.requires_grad_(True)
        module(msa, mask).square().sum().backward()

        assert torch.count_nonzero(msa.grad[~mask]) == 0
        assert torch.count_nonzero(msa.grad[mask]) > 0
        assert _parameter_grads(module)
