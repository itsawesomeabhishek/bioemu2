"""Triton kernel for the memory-saving OuterProductMean variant.

The kernel fuses the MSA contraction, normalization, output projection, and bias for
blocks of token pairs, so the O(N²·C_H²) outer-product intermediate stays in registers.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

_BLOCK_I = 32
_BLOCK_J = 2
_BLOCK_S = 32
_BLOCK_C = 2


@triton.jit
def _opm_fused_kernel(
    A,  # [C_H, S, TILE_I] — channel-major copy of projected MSA rows for the i tile
    B,  # [S, N, C_H]      — projected MSA rows for all j positions (any strides)
    W,  # [D_OUT, C_H²]    — proj_o weight (any strides)
    Bias,  # [D_OUT]       — proj_o bias; unused when HAS_BIAS is false
    NumValid,  # [B]       — valid MSA rows per batch element
    Out,  # [TILE_I, N, D_OUT]
    batch_idx,
    tile_i,
    n_tokens,
    msa_depth,
    c_hidden,
    d_out,
    stride_ac,
    stride_as,
    stride_ai,
    stride_bs,
    stride_bj,
    stride_bc,
    stride_wd,
    stride_wc,
    stride_oi,
    stride_oj,
    stride_od,
    C_H_PAD: tl.constexpr,
    D_OUT_PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Write ``proj_o(mean_s a[s, i] (x) b[s, j]) + bias`` for a ``BLOCK_I x BLOCK_J`` tile.

    For each group of ``BLOCK_C`` left channels the kernel contracts the MSA rows into the
    outer-product slice ``outer[(c, i), (j, c')] = sum_s a[s, i, c] * b[s, j, c']`` and
    immediately projects it with the weight columns ``c * C_H + c'``. Both steps are
    ``tl.dot`` calls, so the ``C_H x C_H`` outer product never leaves registers.
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    dtype = A.dtype.element_ty

    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    ci_offsets = tl.arange(0, BLOCK_C * BLOCK_I)
    c_of_ci = ci_offsets // BLOCK_I
    i_of_ci = pid_i * BLOCK_I + ci_offsets % BLOCK_I
    jc_offsets = tl.arange(0, BLOCK_J * C_H_PAD)
    j_of_jc = pid_j * BLOCK_J + jc_offsets // C_H_PAD
    c_of_jc = jc_offsets % C_H_PAD
    jc_mask = (j_of_jc < n_tokens) & (c_of_jc < c_hidden)
    cc_offsets = tl.arange(0, BLOCK_C * C_H_PAD)
    c_of_cc = cc_offsets // C_H_PAD
    cp_of_cc = cc_offsets % C_H_PAD
    d_offsets = tl.arange(0, D_OUT_PAD)
    d_mask = d_offsets < d_out
    s_offsets = tl.arange(0, BLOCK_S)

    inv_count = 1.0 / tl.load(NumValid + batch_idx).to(tl.float32)
    accumulator = tl.zeros((BLOCK_I * BLOCK_J, D_OUT_PAD), dtype=tl.float32)

    for c_start in range(0, c_hidden, BLOCK_C):
        ci_mask = (i_of_ci < tile_i) & (c_start + c_of_ci < c_hidden)
        outer = tl.zeros((BLOCK_C * BLOCK_I, BLOCK_J * C_H_PAD), dtype=tl.float32)
        for msa_start in range(0, msa_depth, BLOCK_S):
            msa_idx = msa_start + s_offsets
            msa_mask = msa_idx < msa_depth
            a_values = tl.load(
                A
                + (c_start + c_of_ci)[:, None] * stride_ac
                + msa_idx[None, :] * stride_as
                + i_of_ci[:, None] * stride_ai,
                mask=ci_mask[:, None] & msa_mask[None, :],
                other=0.0,
            )
            b_values = tl.load(
                B
                + msa_idx[:, None] * stride_bs
                + j_of_jc[None, :] * stride_bj
                + c_of_jc[None, :] * stride_bc,
                mask=msa_mask[:, None] & jc_mask[None, :],
                other=0.0,
            )
            outer = tl.dot(a_values, b_values, outer, input_precision=INPUT_PRECISION)

        weight_mask = (c_start + c_of_cc < c_hidden) & (cp_of_cc < c_hidden)
        weight_slice = tl.load(
            W
            + ((c_start + c_of_cc) * c_hidden + cp_of_cc)[:, None] * stride_wc
            + d_offsets[None, :] * stride_wd,
            mask=weight_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(dtype)
        # Normalize before the cast so low-precision operands hold the mean, not the sum.
        outer = tl.reshape(outer * inv_count, (BLOCK_C, BLOCK_I, BLOCK_J, C_H_PAD))
        outer = tl.permute(outer, (1, 2, 0, 3))
        accumulator = tl.dot(
            tl.reshape(outer, (BLOCK_I * BLOCK_J, BLOCK_C * C_H_PAD)).to(dtype),
            weight_slice,
            accumulator,
            input_precision=INPUT_PRECISION,
        )

    if HAS_BIAS:
        bias = tl.load(Bias + d_offsets, mask=d_mask, other=0.0).to(tl.float32)
        accumulator += bias[None, :]

    out_mask = (
        (i_offsets < tile_i)[:, None, None]
        & (j_offsets < n_tokens)[None, :, None]
        & d_mask[None, None, :]
    )
    tl.store(
        Out
        + i_offsets[:, None, None] * stride_oi
        + j_offsets[None, :, None] * stride_oj
        + d_offsets[None, None, :] * stride_od,
        tl.reshape(accumulator, (BLOCK_I, BLOCK_J, D_OUT_PAD)).to(Out.dtype.element_ty),
        mask=out_mask,
    )


class _TritonOPMFn(torch.autograd.Function):
    """Triton forward, einsum backward — following _FusedFlashAttnFn pattern.

    The backward is rarely triggered (auto-switch uses vanilla during training).
    It exists as a safety net for eval-with-gradients (e.g. gradient steering).
    Backward is O(N²·C_H²) memory/time — same as vanilla, no Triton speedup.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        a: Tensor,
        b: Tensor,
        num_valid: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        tile_size: int,
        matmul_precision: str,
    ) -> Tensor:
        # a: [B, S, N, C_H]  b: [B, S, N, C_H]  num_valid: [B]
        ctx.save_for_backward(a, b, num_valid, weight, bias if bias is not None else torch.empty(0))
        ctx.has_bias = bias is not None
        return _triton_opm_fused(
            a,
            b,
            num_valid,
            weight,
            bias,
            tile_size=tile_size,
            matmul_precision=matmul_precision,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_out: Tensor
    ) -> tuple[Tensor | None, ...]:
        # grad_out: [B, N, N, D_OUT]
        a, b, num_valid, weight, bias_or_empty = ctx.saved_tensors
        bias = bias_or_empty if ctx.has_bias else None
        count = num_valid.float()[:, None, None, None]

        # Recompute normalized outer product via einsum (O(N²·C_H²) — same as full path)
        z = torch.einsum("bsic,bsjd->bijcd", a.float(), b.float())  # [B, N, N, C_H, C_H]
        z_flat = z.reshape(*z.shape[:3], -1)  # [B, N, N, C_H²]
        z_norm = z_flat / count  # [B, N, N, C_H²]

        # grad through proj_o: dL/dz_norm = grad_out @ W   (W: [D_OUT, C_H²])
        grad_z_norm = grad_out.float() @ weight.float()  # [B, N, N, C_H²]

        # grad through normalization: dL/dz_flat = dL/dz_norm / num_valid
        grad_z_flat = grad_z_norm / count  # [B, N, N, C_H²]
        grad_z = grad_z_flat.reshape(*grad_z_flat.shape[:3], a.shape[-1], b.shape[-1])

        # grad through outer product: dL/da, dL/db
        # z[b,s,i,j,c,d] = a[b,s,i,c] * b[b,s,j,d]
        grad_a = torch.einsum("bijcd,bsjd->bsic", grad_z, b.float()).to(a.dtype)
        grad_b = torch.einsum("bijcd,bsic->bsjd", grad_z, a.float()).to(b.dtype)

        # grad through proj_o weight and bias
        grad_weight = torch.einsum("bijd,bijc->dc", grad_out.float(), z_norm).to(weight.dtype)
        grad_bias = grad_out.float().sum(dim=(0, 1, 2)).to(bias.dtype) if bias is not None else None

        return grad_a, grad_b, None, grad_weight, grad_bias, None, None


def _triton_opm_fused(
    a: Tensor,
    b: Tensor,
    num_valid: Tensor,
    proj_o_weight: Tensor,
    proj_o_bias: Tensor | None,
    tile_size: int = 256,
    matmul_precision: str = "ieee",
) -> Tensor:
    """Row-tiled OPM launcher.

    Beyond the ``[B, N, N, D_OUT]`` output, it allocates only one channel-major copy of
    ``a`` for ``tile_size`` token rows; ``b`` and the weight are read through their strides.
    """
    # a: [B, S, N, C_H]  b: [B, S, N, C_H]  num_valid: [B]
    if a.ndim == 3:
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

    batch_size, msa_depth, n_tokens, c_hidden = a.shape
    d_out = proj_o_weight.shape[0]
    out = torch.empty(batch_size, n_tokens, n_tokens, d_out, device=a.device, dtype=a.dtype)
    # ``tl.arange`` needs power-of-two extents; padded lanes are masked in the kernel.
    c_hidden_padded = triton.next_power_of_2(max(c_hidden, 16))
    d_out_padded = triton.next_power_of_2(max(d_out, 16))
    bias = proj_o_bias if proj_o_bias is not None else proj_o_weight

    for batch_idx in range(batch_size):
        b_batch = b[batch_idx]  # [S, N, C_H]
        for i_start in range(0, n_tokens, tile_size):
            i_end = min(i_start + tile_size, n_tokens)
            tile_i = i_end - i_start
            a_tile = a[batch_idx, :, i_start:i_end].permute(2, 0, 1).contiguous()
            out_tile = out[batch_idx, i_start:i_end]  # [tile_i, N, D_OUT]
            grid = (triton.cdiv(tile_i, _BLOCK_I), triton.cdiv(n_tokens, _BLOCK_J))
            _opm_fused_kernel[grid](
                a_tile,
                b_batch,
                proj_o_weight,
                bias,
                num_valid,
                out_tile,
                batch_idx,
                tile_i,
                n_tokens,
                msa_depth,
                c_hidden,
                d_out,
                *a_tile.stride(),
                *b_batch.stride(),
                *proj_o_weight.stride(),
                *out_tile.stride(),
                C_H_PAD=c_hidden_padded,
                D_OUT_PAD=d_out_padded,
                HAS_BIAS=proj_o_bias is not None,
                BLOCK_I=_BLOCK_I,
                BLOCK_J=_BLOCK_J,
                BLOCK_S=_BLOCK_S,
                BLOCK_C=_BLOCK_C,
                INPUT_PRECISION=matmul_precision,
                num_warps=4,
                num_stages=3,
            )
    return out
