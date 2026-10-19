"""Triton flash-attention forward kernel that also returns log-sum-exp."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ],
    key=["seqlen_q", "seqlen_k", "headdim"],
)
@triton.jit
def _fused_fwd_kernel_with_lse(
    Q,
    K,
    V,
    Bias,
    Out,
    LseM,
    Amax,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_ob,
    stride_oh,
    stride_om,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    HAS_BIAS: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    MULTIPLICITY: tl.constexpr,
):
    """Fused attention forward that also writes row log-sum-exp and maxima.

    ``LseM`` and ``Amax`` have shape ``[B * H, seqlen_q]``. ``Bias`` may
    have shape ``[B / MULTIPLICITY, H, N_q, N_k]`` and broadcasts across the
    folded multiplicity axis.
    """
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # Real-batch index for the M-broadcast pair bias (bias axis 0 is B_real,
    # queries are folded (B_real * MULTIPLICITY)).
    b_real = off_b // MULTIPLICITY

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    if HAS_BIAS:
        b_ptrs = (
            Bias
            + b_real * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        if EVEN_N:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        qk *= softmax_scale

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        if HAS_BIAS:
            if EVEN_M & EVEN_N:
                bias = tl.load(b_ptrs + start_n * stride_bn).to(tl.float32)
            else:
                bias = tl.load(
                    b_ptrs + start_n * stride_bn,
                    mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k),
                    other=0.0,
                ).to(tl.float32)
            qk += bias

        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        if EVEN_N:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=offs_d[None, :] < headdim,
                    other=0.0,
                )
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )

        p = p.to(v.dtype)
        acc_o += tl.dot(p, v, input_precision=INPUT_PRECISION)

    # Write lse_m = log(l_i) (== lse - amax) and amax = m_i.
    lse_m_vals = tl.log(l_i)
    amax_vals = m_i
    lse_ptrs = LseM + off_hb * seqlen_q + offs_m
    amax_ptrs = Amax + off_hb * seqlen_q + offs_m
    if EVEN_M:
        tl.store(lse_ptrs, lse_m_vals)
        tl.store(amax_ptrs, amax_vals)
    else:
        tl.store(lse_ptrs, lse_m_vals, mask=offs_m < seqlen_q)
        tl.store(amax_ptrs, amax_vals, mask=offs_m < seqlen_q)

    acc_o = acc_o / l_i[:, None]

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            )


def _fused_flash_attn_with_lse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor | None,
    softmax_scale: float,
    matmul_precision: str = "ieee",
    multiplicity: int = 1,
) -> tuple[Tensor, Tensor, Tensor]:
    """Triton flash-attention forward that also returns ``(lse_m, amax)``.

    Inputs are ``[B, H, S, D]`` (batch-heads-seq-dim layout).

    When ``multiplicity > 1`` the ``bias`` tensor's batch axis is ``B // multiplicity``
    (the "real" batch), and the kernel does the M-broadcast in-place — no host-side
    ``bias.expand().reshape().contiguous()`` copy. When ``multiplicity == 1`` the bias
    batch axis matches Q's batch axis.

    Returns
    -------
    o : Tensor
        Output of shape ``[B, H, Sq, D]``.
    lse_m : Tensor
        ``log(sum(exp(scores - amax)))`` of shape ``[B, H, Sq]``.
    amax : Tensor
        Per-row max of pre-softmax scores, shape ``[B, H, Sq]``.
    """
    assert q.shape[-1] <= 128, f"head_dim {q.shape[-1]} > 128 not supported"

    B, H, seqlen_q, d = q.shape
    _, _, seqlen_k, _ = k.shape

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    o = torch.empty_like(q)
    lse_m = torch.empty((B * H, seqlen_q), dtype=torch.float32, device=q.device)
    amax = torch.empty((B * H, seqlen_q), dtype=torch.float32, device=q.device)

    has_bias = bias is not None
    if has_bias:
        assert bias is not None  # mypy narrowing
        # Bias batch axis is B // multiplicity (kernel M-broadcasts).
        assert bias.shape[0] * multiplicity == B, (
            f"bias batch axis {bias.shape[0]} must equal B / multiplicity ({B} / {multiplicity})"
        )
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)
    assert multiplicity >= 1 and B % multiplicity == 0, (
        f"multiplicity={multiplicity} must divide batch B={B}"
    )

    def grid(META: dict) -> tuple:
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), B * H)

    _fused_fwd_kernel_with_lse[grid](
        q,
        k,
        v,
        bias if has_bias else q,  # dummy when no bias
        o,
        lse_m,
        amax,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        bias_strides[0],
        bias_strides[1],
        bias_strides[2],
        bias_strides[3],
        o.stride(0),
        o.stride(1),
        o.stride(2),
        H,
        seqlen_q,
        seqlen_k,
        d,
        HAS_BIAS=has_bias,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        EVEN_M=seqlen_q % 128 == 0,
        EVEN_N=seqlen_k % 128 == 0,
        EVEN_HEADDIM=d == BLOCK_HEADDIM,
        INPUT_PRECISION=matmul_precision,
        MULTIPLICITY=multiplicity,
    )

    return o, lse_m.view(B, H, seqlen_q), amax.view(B, H, seqlen_q)
