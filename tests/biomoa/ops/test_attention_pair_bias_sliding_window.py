"""Tests sliding-window AttentionPairBias implementations."""

from contextlib import AbstractContextManager, nullcontext

import pytest
import torch

from bioemu2.biomoa.ops.attention_pair_bias import AttentionPairBiasSlidingWindow
from tests.conftest import set_matmul_precision

from ._accuracy import VariantAccuracyCase

C_S = 384
C_Z = 128
NUM_HEADS = 16

# Small dims for CPU-friendly tests with NW > 1.
W_WIN = 8
H_KEYS = 32
N_WIN_ATOMS = 32
NW = N_WIN_ATOMS // W_WIN

# Large dims for Triton stress tests.
PROD_W_WIN = 32
PROD_H_KEYS = 128
PROD_C_S = 128
PROD_C_Z = 16
PROD_NUM_HEADS = 8
PROD_N_WIN_ATOMS = 128


def _fold_windows(
    tokens: torch.Tensor,
    pairs: torch.Tensor,
    mask: torch.Tensor,
    w: int = W_WIN,
    h: int = H_KEYS,
    c_s: int = C_S,
    c_z: int = C_Z,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold tokens/pairs/mask into sliding windows (mirrors AtomTransformer)."""
    n_atoms = tokens.shape[1]
    nw = n_atoms // w
    s_win = tokens.reshape(tokens.shape[0] * nw, w, c_s)
    z_win = pairs.reshape(pairs.shape[0], nw, w, h, c_z)
    mask_win = mask.reshape(mask.shape[0] * nw, w)
    return s_win, z_win, mask_win


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


def _build_reference_sw_apb(
    c_s: int = C_S,
    c_z: int = C_Z,
    num_heads: int = NUM_HEADS,
    w: int = W_WIN,
    h: int = H_KEYS,
) -> torch.nn.Module:
    """Vanilla sliding-window APB with non-trivial proj_o weights."""
    torch.manual_seed(0)
    module = AttentionPairBiasSlidingWindow(
        c_s=c_s,
        c_z=c_z,
        num_heads=num_heads,
        window_queries=w,
        window_keys=h,
    ).eval()
    with torch.no_grad():
        for p in module.parameters():
            if p.abs().sum() == 0:
                p.copy_(torch.randn_like(p) * 0.02)
    return module


def _make_sw_inputs(
    B: int,
    n_atoms: int = N_WIN_ATOMS,
    *,
    multiplicity: int = 1,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
    w: int = W_WIN,
    h: int = H_KEYS,
    c_s: int = C_S,
    c_z: int = C_Z,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (tokens [B*M, n, c_s], pairs [B, n, h, c_z], mask [B, n]) for SW tests."""
    tokens = torch.randn(
        B * multiplicity, n_atoms, c_s, device=device, dtype=dtype, requires_grad=requires_grad
    )
    pairs = torch.randn(B, n_atoms, h, c_z, device=device, dtype=dtype, requires_grad=requires_grad)
    mask = torch.ones(B, n_atoms, dtype=torch.bool, device=device)
    # make non-trivial mask
    for b in range(B):
        pad_len = (b + 1) * 2
        # Ensure at least w valid atoms remain (softmax over all-masked row → NaN).
        pad_len = min(pad_len, n_atoms - w)
        if pad_len > 0:
            mask[b, n_atoms - pad_len :] = False
    return tokens, pairs, mask


def _make_sw_module(
    variant: str,
    variant_kwargs: dict,
    dtype: torch.dtype,
    device: str,
    *,
    c_s: int = C_S,
    c_z: int = C_Z,
    num_heads: int = NUM_HEADS,
    w: int = W_WIN,
    h: int = H_KEYS,
) -> torch.nn.Module:
    """Create a sliding-window module from the registry with reference weights."""
    ref = _build_reference_sw_apb(c_s=c_s, c_z=c_z, num_heads=num_heads, w=w, h=h)
    module = AttentionPairBiasSlidingWindow(
        c_s=c_s,
        c_z=c_z,
        num_heads=num_heads,
        window_queries=w,
        window_keys=h,
        **variant_kwargs,
    )
    module.load_state_dict(ref.state_dict())
    return module.eval().to(device=device, dtype=dtype)


def _ground_truth_sequential(
    module: torch.nn.Module,
    tokens: torch.Tensor,  # [B*M, n_atoms, c_s]
    pairs: torch.Tensor,  # [B, n_atoms, h, c_z]
    mask: torch.Tensor,  # [B, n_atoms] bool
    B: int,
    M: int,
    *,
    w: int = W_WIN,
    h: int = H_KEYS,
    c_s: int = C_S,
    c_z: int = C_Z,
    amp_ctx: AbstractContextManager[None] | None = None,
) -> torch.Tensor:
    """Ground truth: loop B=1, M=1 calls and stack results."""
    amp = amp_ctx if amp_ctx is not None else nullcontext()
    outs: list[torch.Tensor] = []
    for b in range(B):
        for m in range(M):
            s_bm, z_b, mask_b = _fold_windows(
                tokens[b * M + m : b * M + m + 1],
                pairs[b : b + 1],
                mask[b : b + 1],
                w=w,
                h=h,
                c_s=c_s,
                c_z=c_z,
            )
            with amp:
                out_bm = module(s_bm, z_b, mask_b, multiplicity=1)  # [1*NW, W, c_s]
            outs.append(out_bm)
    # outs order: (b0,m0), (b0,m1), ..., (b1,m0), ... — same as the folded batch layout
    return torch.cat(outs, dim=0)


# Measured worst-case batch-multiplicity drift at B=3, M=3 (A100):
#
#  variant        precision                      param_grad  s_grad   z_grad
#  vanilla        f64  cublas=highest             4.6e-14    0        3.5e-17
#  vanilla        f32  cublas=highest  cpu        1.5e-5     0        3.7e-8  (CI CPU can exceed 1e-4; tolerance set to 2e-4)
#  vanilla        f32  cublas=highest  cuda       1.7e-5     2.4e-7   3.3e-8
#  vanilla        f32  cublas=high(tf32)          2.2e-3     4.8e-5   3.2e-5
#  vanilla        f32  autocast(bf16)             1.2e-1     1.9e-3   4.3e-4

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
        1e-5,
        1e-6,
        id="vanilla-f32-ieee-cpu",
    ),
    pytest.param(
        "vanilla",
        {},
        torch.float32,
        "highest",
        "cuda",
        None,
        5e-5,
        1e-5,
        1e-6,
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
        2e-3,
        id="vanilla-autocast_bf16-cuda",
    ),
]


class TestBatchedMultiplicityMatchesSeparateRuns:
    """Sliding-window APB: batched (B, M) forward/gradient matches B=1 M=1 ground truth."""

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
        del param_atol, z_atol  # forward only checks s output
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_sw_module(variant, variant_kwargs, dtype, device)

        tokens, pairs, mask = _make_sw_inputs(B, multiplicity=M, device=device, dtype=dtype)

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )
        autocast_before = torch.is_autocast_enabled(device)
        with torch.no_grad(), set_matmul_precision(cublas_precision):
            s_b, z_b, mask_b = _fold_windows(tokens, pairs, mask)
            with amp:
                out_batched = module(s_b, z_b, mask_b, multiplicity=M)
            assert torch.is_autocast_enabled(device) == autocast_before
            out_ref = _ground_truth_sequential(module, tokens, pairs, mask, B, M, amp_ctx=amp)

        assert torch.is_autocast_enabled(device) == autocast_before
        assert out_batched.abs().max().item() > 0, "output is trivially zero"
        torch.testing.assert_close(out_batched, out_ref, atol=s_atol, rtol=0.0)

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
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
        module = _make_sw_module(variant, variant_kwargs, dtype, device)

        tokens_data, pairs_data, mask = _make_sw_inputs(
            B, multiplicity=M, device=device, dtype=dtype
        )

        amp = (
            torch.autocast(device, dtype=autocast_dtype)
            if autocast_dtype is not None
            else torch.autocast(device, enabled=False)
        )

        tokens_bat = tokens_data.clone().requires_grad_(True)
        pairs_bat = pairs_data.clone().requires_grad_(True)
        s_b, z_b, mask_b = _fold_windows(tokens_bat, pairs_bat, mask)
        with set_matmul_precision(cublas_precision), amp:
            module(s_b, z_b, mask_b, multiplicity=M).sum().backward()
        bat_param_grads = {
            n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None
        }
        bat_s_grad = tokens_bat.grad.clone()
        bat_z_grad = pairs_bat.grad.clone()
        module.zero_grad()

        seq_param_grads: dict[str, torch.Tensor] = {}
        seq_s_grad = torch.zeros_like(tokens_data)
        seq_z_grad = torch.zeros_like(pairs_data)

        for b in range(B):
            for m in range(M):
                t_bm = tokens_data[b * M + m : b * M + m + 1].clone().requires_grad_(True)
                p_b = pairs_data[b : b + 1].clone().requires_grad_(True)
                s_bm, z_bm, mask_bm = _fold_windows(t_bm, p_b, mask[b : b + 1])
                with set_matmul_precision(cublas_precision), amp:
                    module(s_bm, z_bm, mask_bm, multiplicity=1).sum().backward()
                for n, p in module.named_parameters():
                    if p.grad is not None:
                        if n not in seq_param_grads:
                            seq_param_grads[n] = p.grad.clone()
                        else:
                            seq_param_grads[n] += p.grad
                seq_s_grad[b * M + m] = t_bm.grad.squeeze(0)
                seq_z_grad[b] += p_b.grad.squeeze(0)
                module.zero_grad()

        assert len(bat_param_grads) > 0, "no parameter gradients"
        for name in bat_param_grads:
            torch.testing.assert_close(
                bat_param_grads[name],
                seq_param_grads[name],
                atol=param_atol,
                rtol=0.0,
                msg=f"param grad {name}",
            )
        torch.testing.assert_close(
            bat_s_grad, seq_s_grad, atol=s_atol, rtol=0.0, msg="input s grad"
        )
        torch.testing.assert_close(
            bat_z_grad, seq_z_grad, atol=z_atol, rtol=0.0, msg="input z grad"
        )


class TestMultiplicityGradientIsolation:
    """Replicate 0's gradient is unaffected by co-batched noise replicates."""

    def test_backward(self) -> None:
        M = 2
        torch.manual_seed(0)
        B = 1
        module = _make_sw_module("vanilla", {}, torch.float32, "cpu")

        torch.manual_seed(1)
        s_data = torch.randn(B * M * NW, W_WIN, C_S)
        z_data = torch.randn(B, NW, W_WIN, H_KEYS, C_Z)
        mask = torch.ones(B * NW, W_WIN, dtype=torch.bool)

        # Solo (M=1) backward
        s_solo = s_data[:NW].clone().requires_grad_(True)
        z_solo = z_data.clone().requires_grad_(True)
        with set_matmul_precision("highest"):
            module(s_solo, z_solo, mask, multiplicity=1).sum().backward()

        # Multi (M>1) backward, loss on replicate 0 only
        s_multi = s_data.clone().detach().requires_grad_(True)
        z_multi = z_data.clone().requires_grad_(True)
        with set_matmul_precision("highest"):
            out_multi = module(s_multi, z_multi, mask, multiplicity=M)
            out_multi[:NW].sum().backward()  # only replicate 0

        torch.testing.assert_close(s_multi.grad[:NW], s_solo.grad, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(z_multi.grad, z_solo.grad, atol=1e-6, rtol=1e-4)
        assert torch.count_nonzero(s_multi.grad[NW:]) == 0


def _pad_one_window(
    tokens: torch.Tensor,
    pairs: torch.Tensor,
    mask: torch.Tensor,
    *,
    w: int = W_WIN,
    h: int = H_KEYS,
    c_s: int = C_S,
    c_z: int = C_Z,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append one fully-masked window to tokens, pairs, and mask.

    Args:
        tokens: [B*M, n_atoms, c_s]
        pairs: [B, n_atoms, h, c_z]
        mask: [B, n_atoms] bool

    Returns:
        Padded (tokens, pairs, mask) with n_atoms increased by w.
    """
    BM, n_atoms, _ = tokens.shape
    B = mask.shape[0]

    # Pad tokens with zeros for one extra window
    pad_tokens = torch.zeros(BM, w, c_s, device=tokens.device, dtype=tokens.dtype)
    tokens_padded = torch.cat([tokens, pad_tokens], dim=1)

    # Pad pairs with zeros for one extra window
    pad_pairs = torch.zeros(B, w, h, c_z, device=pairs.device, dtype=pairs.dtype)
    pairs_padded = torch.cat([pairs, pad_pairs], dim=1)

    # Pad mask with False (fully masked)
    pad_mask = torch.zeros(B, w, dtype=torch.bool, device=mask.device)
    mask_padded = torch.cat([mask, pad_mask], dim=1)

    return tokens_padded, pairs_padded, mask_padded


class TestMaskedPadding:
    """Output for valid windows is unchanged when an extra fully-masked window is appended."""

    @pytest.mark.parametrize("B,M", [pytest.param(2, 2, id="batched-multiplicity")])
    def test_padding_invariance_vanilla_f64(self, B: int, M: int) -> None:
        """Vanilla f64 on CPU: padded output must be bitwise identical for original windows."""
        torch.manual_seed(42)
        module = _make_sw_module("vanilla", {}, torch.float64, "cpu")

        tokens, pairs, mask = _make_sw_inputs(B, multiplicity=M, device="cpu", dtype=torch.float64)
        nw_orig = N_WIN_ATOMS // W_WIN

        # Run without padding
        s_orig, z_orig, mask_orig = _fold_windows(tokens, pairs, mask)
        with torch.no_grad(), set_matmul_precision("highest"):
            out_orig = module(s_orig, z_orig, mask_orig, multiplicity=M)

        # Run with one extra fully-masked window
        tokens_pad, pairs_pad, mask_pad = _pad_one_window(tokens, pairs, mask)
        s_pad, z_pad, mask_pad_folded = _fold_windows(
            tokens_pad, pairs_pad, mask_pad, w=W_WIN, h=H_KEYS, c_s=C_S, c_z=C_Z
        )
        with torch.no_grad(), set_matmul_precision("highest"):
            out_padded = module(s_pad, z_pad, mask_pad_folded, multiplicity=M)

        # Extract the original windows from the padded output.
        # Folded layout: [B*M*NW, W, c_s]. After padding NW becomes nw_orig+1.
        nw_padded = nw_orig + 1
        out_padded_reshaped = out_padded.view(B * M, nw_padded, W_WIN, C_S)
        out_padded_orig = out_padded_reshaped[:, :nw_orig, :, :].reshape(
            B * M * nw_orig, W_WIN, C_S
        )

        assert out_orig.shape == out_padded_orig.shape
        # f64 with highest precision: expect bitwise equality
        torch.testing.assert_close(out_padded_orig, out_orig, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize("B,M", [pytest.param(2, 2, id="batched-multiplicity")])
    def test_padding_invariance_vanilla_f32(self, B: int, M: int) -> None:
        """Vanilla f32 on CPU: padded output matches within float32 tolerance."""
        torch.manual_seed(42)
        module = _make_sw_module("vanilla", {}, torch.float32, "cpu")

        tokens, pairs, mask = _make_sw_inputs(B, multiplicity=M, device="cpu", dtype=torch.float32)
        nw_orig = N_WIN_ATOMS // W_WIN

        s_orig, z_orig, mask_orig = _fold_windows(tokens, pairs, mask)
        with torch.no_grad(), set_matmul_precision("highest"):
            out_orig = module(s_orig, z_orig, mask_orig, multiplicity=M)

        tokens_pad, pairs_pad, mask_pad = _pad_one_window(tokens, pairs, mask)
        s_pad, z_pad, mask_pad_folded = _fold_windows(
            tokens_pad, pairs_pad, mask_pad, w=W_WIN, h=H_KEYS, c_s=C_S, c_z=C_Z
        )
        with torch.no_grad(), set_matmul_precision("highest"):
            out_padded = module(s_pad, z_pad, mask_pad_folded, multiplicity=M)

        nw_padded = nw_orig + 1
        out_padded_reshaped = out_padded.view(B * M, nw_padded, W_WIN, C_S)
        out_padded_orig = out_padded_reshaped[:, :nw_orig, :, :].reshape(
            B * M * nw_orig, W_WIN, C_S
        )

        assert out_orig.shape == out_padded_orig.shape
        # f32: changing the batch dimension (NW vs NW+1) causes BLAS to partition matmuls
        # differently across threads, introducing rounding drift up to ~1 ULP (~1.2e-7).
        torch.testing.assert_close(out_padded_orig, out_orig, atol=2e-7, rtol=0.0)

    @pytest.mark.gpu
    @pytest.mark.parametrize("case", INVARIANCE_CASES)
    def test_backward(self, gpu_device: torch.device, case: VariantAccuracyCase) -> None:
        """An appended masked window preserves valid input and parameter gradients."""
        reference = _make_sw_module(
            case.reference_variant,
            case.options,
            torch.float32,
            str(gpu_device),
        )
        padded = _make_sw_module(case.variant, case.options, torch.float32, str(gpu_device))
        padded.load_state_dict(reference.state_dict())
        tokens, pairs, mask = _make_sw_inputs(1, device=gpu_device)
        padded_tokens, padded_pairs, padded_mask = _pad_one_window(tokens, pairs, mask)
        reference_tokens = tokens.detach().clone().requires_grad_(True)
        reference_pairs = pairs.detach().clone().requires_grad_(True)
        padded_tokens = padded_tokens.detach().clone().requires_grad_(True)
        padded_pairs = padded_pairs.detach().clone().requires_grad_(True)
        s_ref, z_ref, mask_ref = _fold_windows(reference_tokens, reference_pairs, mask)
        s_pad, z_pad, mask_pad = _fold_windows(padded_tokens, padded_pairs, padded_mask)
        output_gradient = torch.randn_like(s_ref)

        # Use full precision to measure differences caused by the window count.
        with set_matmul_precision("highest"):
            reference(s_ref, z_ref, mask_ref).backward(output_gradient)
            padded_output = padded(s_pad, z_pad, mask_pad).view(1, NW + 1, W_WIN, C_S)
            padded_output[:, :NW].reshape_as(s_ref).backward(output_gradient)

        input_atol, input_rtol = case.input_gradient_tolerance
        torch.testing.assert_close(
            padded_tokens.grad[:, :N_WIN_ATOMS],
            reference_tokens.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        torch.testing.assert_close(
            padded_pairs.grad[:, :N_WIN_ATOMS],
            reference_pairs.grad,
            atol=input_atol,
            rtol=input_rtol,
        )
        assert torch.count_nonzero(padded_tokens.grad[:, N_WIN_ATOMS:]) == 0
        assert torch.count_nonzero(padded_pairs.grad[:, N_WIN_ATOMS:]) == 0
        reference_grads = {
            name: parameter.grad.clone() for name, parameter in reference.named_parameters()
        }
        parameter_atol, parameter_rtol = case.parameter_gradient_tolerance
        for name, parameter in padded.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference_grads[name],
                atol=parameter_atol,
                rtol=parameter_rtol,
            )


class TestMasking:
    """Masked atoms do not affect valid outputs or gradients."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("case", INVARIANCE_CASES)
    def test_forward(self, gpu_device: torch.device, case: VariantAccuracyCase) -> None:
        module = _make_sw_module(case.variant, case.options, torch.float32, str(gpu_device))
        tokens, pairs, mask = _make_sw_inputs(1, device=gpu_device)
        changed_tokens, changed_pairs = tokens.clone(), pairs.clone()
        changed_tokens[:, ~mask[0]] = 1_000.0
        changed_pairs[:, ~mask[0]] = 1_000.0
        s, z, folded_mask = _fold_windows(tokens, pairs, mask)
        changed_s, changed_z, _ = _fold_windows(changed_tokens, changed_pairs, mask)

        with torch.no_grad():
            expected = module(s, z, folded_mask)
            actual = module(changed_s, changed_z, folded_mask)

        assert mask.any() and (~mask).any()
        forward_atol, forward_rtol = case.forward_tolerance
        torch.testing.assert_close(
            actual[folded_mask],
            expected[folded_mask],
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
        module = _make_sw_module(variant, {}, torch.float32, str(gpu_device))
        tokens, pairs, mask = _make_sw_inputs(1, device=gpu_device)
        tokens.requires_grad_(True)
        pairs.requires_grad_(True)
        s, z, folded_mask = _fold_windows(tokens, pairs, mask)
        module(s, z, folded_mask)[folded_mask].sum().backward()

        assert torch.count_nonzero(tokens.grad[:, ~mask[0]]) == 0
        assert torch.count_nonzero(tokens.grad[:, mask[0]]) > 0
        assert any(parameter.grad is not None for parameter in module.parameters())
