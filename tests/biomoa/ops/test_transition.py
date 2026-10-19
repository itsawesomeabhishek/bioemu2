"""Tests Transition implementations."""

import pytest
import torch

from bioemu2.biomoa.ops.transition import Transition
from tests.conftest import set_matmul_precision

from ._accuracy import VariantAccuracyCase

DIM = 128
HIDDEN = 256

FORWARD_CASES = [
    VariantAccuracyCase(
        "vanilla-no-norm-chunk-73",
        "vanilla",
        "vanilla",
        options={"device": "cpu", "use_norm": False, "hidden_chunk": 73},
        forward_tolerance=(2e-6, 0.0),
    ),
]
FORWARD_CASES = [
    pytest.param(
        case,
        marks=pytest.mark.gpu if case.options["device"] == "cuda" else (),
        id=case.name,
    )
    for case in FORWARD_CASES
]

BACKWARD_CASES = [
    VariantAccuracyCase(
        "vanilla-norm-bf16",
        "vanilla",
        "vanilla",
        autocast_dtype=torch.bfloat16,
        options={"use_norm": True, "hidden_chunk": None},
        input_gradient_tolerance=(5e-3, 2e-2),
        parameter_gradient_tolerance=(5e-2, 2e-2),
    ),
]
BACKWARD_CASES = [pytest.param(case, id=case.name) for case in BACKWARD_CASES]


def _make_transition(
    *,
    use_norm: bool = True,
    chunk_size: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    torch.manual_seed(42)
    module = Transition(
        dim=DIM,
        hidden=HIDDEN,
        use_norm=use_norm,
        chunk_size=chunk_size,
    )
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear):
                submodule.weight.normal_(0, 0.02)
            elif isinstance(submodule, torch.nn.LayerNorm):
                submodule.weight.uniform_(0.5, 1.5)
                submodule.bias.normal_(0, 0.1)
    return module.to(device=device, dtype=dtype).eval()


def _parameter_grads(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


class TestOptimizedImplementationMatchesReference:
    """Each implementation matches the PyTorch reference."""

    @pytest.mark.parametrize("case", FORWARD_CASES)
    def test_forward(
        self,
        case: VariantAccuracyCase,
    ) -> None:
        device = case.options["device"]
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if case.dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 is not supported on this GPU")
        reference = _make_transition(
            use_norm=case.options["use_norm"],
            device=device,
            dtype=case.reference_dtype,
        )
        candidate = _make_transition(
            use_norm=case.options["use_norm"],
            chunk_size=case.options["hidden_chunk"],
            device=device,
            dtype=case.dtype,
        )
        candidate.load_state_dict(reference.state_dict())
        torch.manual_seed(0)
        reference_x = torch.randn(2, 11, 13, DIM, device=device, dtype=case.reference_dtype)
        candidate_x = reference_x.to(case.dtype)

        with torch.no_grad(), set_matmul_precision("highest"):
            expected = reference(reference_x)
        with torch.no_grad(), set_matmul_precision(case.matmul_precision):
            actual = candidate(candidate_x)

        assert expected.abs().max().item() > 0, "zero-initialized output projection was not reset"
        atol, rtol = case.forward_tolerance
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("case", BACKWARD_CASES)
    def test_backward(
        self,
        gpu_device: torch.device,
        case: VariantAccuracyCase,
    ) -> None:
        """Training variants and BF16 autocast match their FP32 reference."""
        if case.autocast_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 is not supported on this GPU")
        reference = _make_transition(
            use_norm=case.options["use_norm"],
            device=gpu_device,
        ).train()
        candidate = _make_transition(
            use_norm=case.options["use_norm"],
            chunk_size=case.options["hidden_chunk"],
            device=gpu_device,
        ).train()
        candidate.load_state_dict(reference.state_dict())
        torch.manual_seed(0)
        x = torch.randn(2, 5, 7, DIM, device=gpu_device)
        reference_x = x.clone().requires_grad_(True)
        candidate_x = x.clone().requires_grad_(True)
        output_gradient = torch.randn_like(x)

        with set_matmul_precision("highest"):
            reference(reference_x).backward(output_gradient)
        with (
            set_matmul_precision(case.matmul_precision),
            torch.autocast(
                device_type="cuda",
                dtype=case.autocast_dtype,
                enabled=case.autocast_dtype is not None,
            ),
        ):
            candidate(candidate_x).backward(output_gradient)

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            candidate_x.grad,
            reference_x.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        reference_grads = _parameter_grads(reference)
        candidate_grads = _parameter_grads(candidate)
        assert candidate_grads.keys() == reference_grads.keys()
        param_atol, param_rtol = case.parameter_gradient_tolerance
        for name in reference_grads:
            torch.testing.assert_close(
                candidate_grads[name],
                reference_grads[name],
                atol=param_atol,
                rtol=param_rtol,
                msg=f"parameter gradient {name}",
            )
