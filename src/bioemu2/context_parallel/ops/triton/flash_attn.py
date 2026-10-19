"""Fused Triton FlashAttention with additive pair-bias.

Forward: online softmax + optional logsumexp output (saved for FA-2 backward).
Backward: explicit FA-2 backward in 3 Triton kernels — no materialised P or dS.

Kernels
-------
- ``_fused_fwd_kernel``: forward; if ``HAS_LSE``, also writes ``m_i + log(l_i)``.
- ``_fused_bwd_preprocess_kernel``: ``D[b,h,m] = sum_d O[b,h,m,d] * dO[b,h,m,d]``.
- ``_fused_bwd_dq_kernel``: grid ``(ceil(N_q/BLOCK_M), B*H)``; iterates K-tiles
  for each Q-tile to accumulate ``dQ``. No atomics.
- ``_fused_bwd_dkv_kernel``: grid ``(ceil(N_k/BLOCK_N), B*H)``; iterates Q-tiles
  to accumulate ``dK``, ``dV``, and (optionally) ``d_bias``. Each (Q-tile, K-tile)
  cell of ``d_bias`` is written by exactly one program, so no atomics are needed.

Memory
------
Backward saves ``q, k, v, bias, o, lse`` only — no [B,H,N,N] intermediates.
Extra over forward: one fp32 ``lse`` of shape ``[B,H,N_q]`` (negligible).

Numerics
--------
All accumulators are fp32 in registers; every ``tl.dot`` uses the same
``INPUT_PRECISION`` as the forward kernel so recomputed scores match. Note
``tl.dot``'s ``input_precision`` only applies to fp32 inputs; for bf16/fp16
tensors (e.g. under ``fabric.precision="bf16-mixed"``) tensor cores use the
reduced-precision path and the kwarg is ignored. Recomputation uses the saved
``lse`` directly so the backward picks the *same* softmax max as the forward.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

# Accepted `matmul_precision` values for `fused_flash_attn`. Must match the
# `tl.dot(input_precision=...)` choices used inside the Triton kernels.
# Note: `tl.dot`'s `input_precision` only applies to fp32 inputs; for bf16/fp16
# tensors (e.g. under `fabric.precision="bf16-mixed"`) this kwarg is ignored.
_SUPPORTED_PRECISIONS = frozenset({"ieee", "tf32", "tf32x3"})


_FWD_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_FWD_CONFIGS, key=["seqlen_q", "seqlen_k", "headdim"])
@triton.jit
def _fused_fwd_kernel(
    Q,
    K,
    V,
    Bias,
    MaskBias,
    Out,
    Lse,
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
    stride_mbb,
    stride_mbn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_lb,
    stride_lh,
    stride_lm,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    HAS_BIAS: tl.constexpr,
    HAS_MASK_BIAS: tl.constexpr,
    HAS_LSE: tl.constexpr,
    M: tl.constexpr,
    NW: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # Pair bias broadcasts over the multiplicity axis M: the folded batch id is
    # off_b = (b*M + m)*NW + k, so the un-M-folded (windowed) bias index is
    # bz = (off_b // (M*NW))*NW + (off_b % NW) = b*NW + k. With M=1 this is the
    # identity (bz == off_b) for any NW, so the non-sliding-window path is unchanged.
    bz = (off_b // (M * NW)) * NW + (off_b % NW)

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
            + bz * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        )

    if HAS_MASK_BIAS:
        # Per-folded key mask additive [folded, seqlen_k]; broadcast over heads and
        # the query rows. Indexed by the folded batch id (varies per window, so it
        # cannot be folded into the M-broadcast pair bias).
        mb_ptrs = MaskBias + off_b * stride_mbb + offs_n[None, :] * stride_mbn

    # Accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q
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

    # Stream over K/V blocks
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

        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        qk *= softmax_scale

        if not EVEN_N:
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))

        # Add bias (pair bias, M-broadcast via bz)
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

        # Add per-folded key-mask bias (broadcast over query rows).
        if HAS_MASK_BIAS:
            if EVEN_N:
                mask_bias = tl.load(mb_ptrs + start_n * stride_mbn).to(tl.float32)
            else:
                mask_bias = tl.load(
                    mb_ptrs + start_n * stride_mbn,
                    mask=(start_n + offs_n)[None, :] < seqlen_k,
                    other=0.0,
                ).to(tl.float32)
            qk += mask_bias

        # Online softmax update
        m_ij = tl.maximum(tl.max(qk, 1), m_i)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.exp(m_i - m_ij)
        acc_o = acc_o * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Accumulate V
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

    # Final normalization
    acc_o = acc_o / l_i[:, None]

    if HAS_LSE:
        # Save lse = m + log(l) for FA-2 backward. Using max(l_i, tiny) so rows
        # with all-masked tiles (l_i=0) get a sentinel; those rows won't be
        # queried by backward because dO is then zero for them.
        lse_val = m_i + tl.log(tl.maximum(l_i, 1e-30))
        lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
        if EVEN_M:
            tl.store(lse_ptrs, lse_val)
        else:
            tl.store(lse_ptrs, lse_val, mask=offs_m < seqlen_q)

    # Write output
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


# Backward preprocess kernel — D[b,h,m] = sum_d O * dO


@triton.jit
def _fused_bwd_preprocess_kernel(
    Out,
    DO,
    Delta,
    nheads,
    seqlen_q,
    headdim,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_db,
    stride_dh,
    stride_dm,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    m_mask = offs_m < seqlen_q
    d_mask = offs_d < headdim

    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta + off_b * stride_db + off_h * stride_dh + offs_m * stride_dm,
        delta,
        mask=m_mask,
    )


# Backward kernels share these configs (autotune key matches forward)

_BWD_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_BWD_CONFIGS, key=["seqlen_q", "seqlen_k", "headdim"])
@triton.jit
def _fused_bwd_dq_kernel(
    Q,
    K,
    V,
    Bias,
    MaskBias,
    DO,
    Lse,
    Delta,
    DQ,
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
    stride_mbb,
    stride_mbn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_delb,
    stride_delh,
    stride_delm,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    HAS_BIAS: tl.constexpr,
    HAS_MASK_BIAS: tl.constexpr,
    M: tl.constexpr,
    NW: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Compute dQ tile for one (b, h, q-block).

    Grid: ``(ceil(seqlen_q / BLOCK_M), B * H)`` — same layout as forward.
    No atomics: each Q-tile owns its slice of ``dq``.
    """
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # M-broadcast pair bias: recover windowed bias index bz = b*NW + k from the
    # folded batch id off_b = (b*M + m)*NW + k. Identity when M=1.
    bz = (off_b // (M * NW)) * NW + (off_b % NW)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    m_mask = offs_m < seqlen_q
    d_mask = offs_d < headdim

    # Load Q tile, dO tile, lse, delta — all keyed on the Q dimension
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    do_ptrs = (
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :]
    )
    do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm
    lse = tl.load(lse_ptrs, mask=m_mask, other=0.0)

    del_ptrs = Delta + off_b * stride_delb + off_h * stride_delh + offs_m * stride_delm
    delta = tl.load(del_ptrs, mask=m_mask, other=0.0)

    if HAS_BIAS:
        b_ptrs_base = (
            Bias
            + bz * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
        )

    dq_acc = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_mask = (start_n + offs_n) < seqlen_k

        k_ptrs = (
            K
            + off_b * stride_kb
            + off_h * stride_kh
            + (start_n + offs_n)[:, None] * stride_kn
            + offs_d[None, :]
        )
        k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V
            + off_b * stride_vb
            + off_h * stride_vh
            + (start_n + offs_n)[:, None] * stride_vn
            + offs_d[None, :]
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        # Recompute s = softmax_scale * (q @ k^T) + bias (matches forward order)
        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        s = qk * softmax_scale

        if HAS_BIAS:
            bias = tl.load(
                b_ptrs_base + start_n * stride_bn,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            s += bias

        if HAS_MASK_BIAS:
            mask_bias = tl.load(
                MaskBias + off_b * stride_mbb + (start_n + offs_n) * stride_mbn,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            s += mask_bias[None, :]

        # Mask out invalid keys before exp so they contribute zero P.
        s = tl.where(n_mask[None, :], s, -float("inf"))

        # P = exp(s - lse) — no recompute of max because we saved lse.
        p = tl.exp(s - lse[:, None])
        p = tl.where(n_mask[None, :], p, 0.0)

        # dP = dO @ V^T : [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)

        # dS = P * (dP - delta)  (wrt the softmax input s)
        ds = p * (dp - delta[:, None])
        ds = tl.where(n_mask[None, :], ds, 0.0)

        # dQ += (dS @ K) * scale (scale because s = scale * qk + bias)
        dq_acc += tl.dot(ds, k, input_precision=INPUT_PRECISION) * softmax_scale

    dq_ptrs = (
        DQ
        + off_b * stride_dqb
        + off_h * stride_dqh
        + offs_m[:, None] * stride_dqm
        + offs_d[None, :]
    )
    tl.store(dq_ptrs, dq_acc, mask=m_mask[:, None] & d_mask[None, :])


@triton.autotune(configs=_BWD_CONFIGS, key=["seqlen_q", "seqlen_k", "headdim"])
@triton.jit
def _fused_bwd_dkv_kernel(
    Q,
    K,
    V,
    Bias,
    MaskBias,
    DO,
    Lse,
    Delta,
    DK,
    DV,
    DBias,
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
    stride_mbb,
    stride_mbn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_lb,
    stride_lh,
    stride_lm,
    stride_delb,
    stride_delh,
    stride_delm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_dbb,
    stride_dbh,
    stride_dbm,
    stride_dbn,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    HAS_BIAS: tl.constexpr,
    HAS_MASK_BIAS: tl.constexpr,
    M: tl.constexpr,
    NW: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Compute dK, dV, and (optionally) d_bias tiles for one (b, h, k-block).

    Grid: ``(ceil(seqlen_k / BLOCK_N), B * H)``.
    Each program owns a K-tile and accumulates ``dk``, ``dv`` without atomics.
    ``d_bias[Q, K]`` is unique per (Q-tile, K-tile) pair within this program,
    so we store it non-atomically too (each (m_block, n_block) cell is written
    by exactly one program — the one owning n_block). ``d_bias`` stays at the
    folded batch size ``[folded, ...]`` (indexed by ``off_b``); the M-reduction
    to the windowed ``[B*NW, ...]`` bias grad happens in the autograd ``Function``.
    """
    start_n = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # M-broadcast pair bias: recover windowed bias index bz = b*NW + k from the
    # folded batch id off_b = (b*M + m)*NW + k. Identity when M=1.
    bz = (off_b // (M * NW)) * NW + (off_b % NW)

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    n_mask = offs_n < seqlen_k
    d_mask = offs_d < headdim

    # Load K, V tiles (owned by this program)
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :]
    )
    k = tl.load(k_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]
    )
    v = tl.load(v_ptrs, mask=n_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    dk_acc = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    for start_m in range(0, seqlen_q, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        m_mask = (start_m + offs_m) < seqlen_q

        q_ptrs = (
            Q
            + off_b * stride_qb
            + off_h * stride_qh
            + (start_m + offs_m)[:, None] * stride_qm
            + offs_d[None, :]
        )
        q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        do_ptrs = (
            DO
            + off_b * stride_dob
            + off_h * stride_doh
            + (start_m + offs_m)[:, None] * stride_dom
            + offs_d[None, :]
        )
        do = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        lse_ptrs = Lse + off_b * stride_lb + off_h * stride_lh + (start_m + offs_m) * stride_lm
        lse = tl.load(lse_ptrs, mask=m_mask, other=0.0)

        del_ptrs = (
            Delta + off_b * stride_delb + off_h * stride_delh + (start_m + offs_m) * stride_delm
        )
        delta = tl.load(del_ptrs, mask=m_mask, other=0.0)

        # Recompute s = scale * (q @ k^T) + bias
        qk = tl.dot(q, tl.trans(k), input_precision=INPUT_PRECISION)
        s = qk * softmax_scale

        if HAS_BIAS:
            bias_ptrs = (
                Bias
                + bz * stride_bb
                + off_h * stride_bh
                + (start_m + offs_m)[:, None] * stride_bm
                + offs_n[None, :] * stride_bn
            )
            s += tl.load(
                bias_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            ).to(tl.float32)

        if HAS_MASK_BIAS:
            mask_bias = tl.load(
                MaskBias + off_b * stride_mbb + offs_n * stride_mbn,
                mask=n_mask,
                other=0.0,
            ).to(tl.float32)
            s += mask_bias[None, :]

        s = tl.where(n_mask[None, :], s, -float("inf"))

        # P = exp(s - lse)
        p = tl.exp(s - lse[:, None])
        p = tl.where(m_mask[:, None] & n_mask[None, :], p, 0.0)

        # dV += P^T @ dO : [BLOCK_N, BLOCK_HEADDIM]
        dv_acc += tl.dot(tl.trans(p), do, input_precision=INPUT_PRECISION)

        # dP = dO @ V^T : [BLOCK_M, BLOCK_N]
        dp = tl.dot(do, tl.trans(v), input_precision=INPUT_PRECISION)

        # dS = P * (dP - delta)
        ds = p * (dp - delta[:, None])
        ds = tl.where(m_mask[:, None] & n_mask[None, :], ds, 0.0)

        # dK += (dS^T @ Q) * scale
        dk_acc += tl.dot(tl.trans(ds), q, input_precision=INPUT_PRECISION) * softmax_scale

        if HAS_BIAS:
            # d_bias[b, h, m, n] = ds[m, n]. Each (m_block, n_block) cell is
            # written by exactly ONE program (the one owning this n_block and
            # iterating to this m_block), so no atomics needed — and we can
            # store directly in the user's bias dtype.
            dbias_ptrs = (
                DBias
                + off_b * stride_dbb
                + off_h * stride_dbh
                + (start_m + offs_m)[:, None] * stride_dbm
                + offs_n[None, :] * stride_dbn
            )
            tl.store(
                dbias_ptrs,
                ds.to(DBias.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )

    dk_ptrs = (
        DK
        + off_b * stride_dkb
        + off_h * stride_dkh
        + offs_n[:, None] * stride_dkn
        + offs_d[None, :]
    )
    tl.store(dk_ptrs, dk_acc, mask=n_mask[:, None] & d_mask[None, :])

    dv_ptrs = (
        DV
        + off_b * stride_dvb
        + off_h * stride_dvh
        + offs_n[:, None] * stride_dvn
        + offs_d[None, :]
    )
    tl.store(dv_ptrs, dv_acc, mask=n_mask[:, None] & d_mask[None, :])


class _FusedFlashAttnFn(torch.autograd.Function):
    """Triton forward + Triton FA-2 backward for fused flash attention.

    Forward emits ``lse`` so the backward can recompute attention probabilities
    tile-by-tile without materialising any [B, H, N, N] tensor. Backward runs
    three Triton kernels (preprocess, dQ, dKV); dQ/dK/dV and d_bias have no
    atomics — each program owns a disjoint output tile.
    """

    @staticmethod
    def forward(ctx, q, k, v, bias, softmax_scale, matmul_precision, mask_bias, M, NW):
        B, H, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape

        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
        o = torch.empty_like(q)

        needs_grad = any(ctx.needs_input_grad[:4])  # q, k, v, bias
        # Allocate lse only when backward will run, to keep inference memory unchanged.
        if needs_grad:
            lse = torch.empty(B, H, seqlen_q, device=q.device, dtype=torch.float32)
            lse_strides = (lse.stride(0), lse.stride(1), lse.stride(2))
        else:
            lse = torch.empty(0, device=q.device, dtype=torch.float32)
            lse_strides = (0, 0, 0)

        def grid(META: dict) -> tuple:
            return (triton.cdiv(seqlen_q, META["BLOCK_M"]), B * H)

        has_bias = bias is not None
        if has_bias:
            bias_strides = (
                bias.stride(0),
                bias.stride(1),
                bias.stride(2),
                bias.stride(3),
            )
        else:
            bias_strides = (0, 0, 0, 0)

        has_mask_bias = mask_bias is not None
        mask_bias_strides = (mask_bias.stride(0), mask_bias.stride(1)) if has_mask_bias else (0, 0)

        _fused_fwd_kernel[grid](
            q,
            k,
            v,
            bias if has_bias else q,  # dummy when no bias
            mask_bias if has_mask_bias else q,  # dummy when no mask bias
            o,
            lse,
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
            mask_bias_strides[0],
            mask_bias_strides[1],
            o.stride(0),
            o.stride(1),
            o.stride(2),
            lse_strides[0],
            lse_strides[1],
            lse_strides[2],
            H,
            seqlen_q,
            seqlen_k,
            d,
            HAS_BIAS=has_bias,
            HAS_MASK_BIAS=has_mask_bias,
            HAS_LSE=needs_grad,
            M=M,
            NW=NW,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=seqlen_q % 128 == 0,
            EVEN_N=seqlen_k % 128 == 0,
            EVEN_HEADDIM=d == BLOCK_HEADDIM,
            INPUT_PRECISION=matmul_precision,
        )

        # Save tensors for backward. We always save bias / mask_bias (or an empty
        # placeholder) so the saved-tensor list length is constant — needed
        # because PyTorch checks that backward returns the same number of grads
        # as forward inputs.
        bias_saved = bias if has_bias else torch.empty(0, device=q.device, dtype=q.dtype)
        mask_bias_saved = (
            mask_bias if has_mask_bias else torch.empty(0, device=q.device, dtype=q.dtype)
        )
        ctx.save_for_backward(q, k, v, bias_saved, mask_bias_saved, o, lse)
        ctx.has_bias = has_bias
        ctx.has_mask_bias = has_mask_bias
        ctx.has_lse = needs_grad
        ctx.softmax_scale = softmax_scale
        ctx.matmul_precision = matmul_precision
        ctx.M = M
        ctx.NW = NW
        return o

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.has_lse:
            # forward was run under no_grad; should not get here, but be safe.
            return None, None, None, None, None, None, None, None, None

        q, k, v, bias_saved, mask_bias_saved, o, lse = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        mask_bias = mask_bias_saved if ctx.has_mask_bias else None
        softmax_scale = ctx.softmax_scale
        matmul_precision = ctx.matmul_precision
        M, NW = ctx.M, ctx.NW

        B, H, seqlen_q, d = q.shape
        _, _, seqlen_k, _ = k.shape
        BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)

        # Contiguous dO for stride simplicity
        do = grad_output.contiguous()

        # Allocate outputs in the input dtype (gradients ultimately go back to
        # the input dtype slots in the autograd graph).
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        if ctx.has_bias:
            # d_bias is written by direct store (no atomics) at the folded batch
            # size [folded, H, W, H_keys]; we reduce over M below to match the
            # windowed z-bias input shape. Allocate in the user's bias dtype —
            # no fp32 [B,H,N,N] accumulator.
            d_bias = torch.empty(B, H, seqlen_q, seqlen_k, device=q.device, dtype=bias.dtype)
        else:
            d_bias = torch.empty(0, device=q.device, dtype=q.dtype)

        mask_bias_strides = (
            (mask_bias.stride(0), mask_bias.stride(1)) if ctx.has_mask_bias else (0, 0)
        )

        # 1) Preprocess: delta[b,h,m] = sum_d O[b,h,m,d] * dO[b,h,m,d]
        delta = torch.empty(B, H, seqlen_q, device=q.device, dtype=torch.float32)
        BLOCK_M_PRE = 128
        pre_grid = (triton.cdiv(seqlen_q, BLOCK_M_PRE), B * H)
        _fused_bwd_preprocess_kernel[pre_grid](
            o,
            do,
            delta,
            H,
            seqlen_q,
            d,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BLOCK_M=BLOCK_M_PRE,
        )

        bias_strides = (
            (
                bias.stride(0),
                bias.stride(1),
                bias.stride(2),
                bias.stride(3),
            )
            if ctx.has_bias
            else (0, 0, 0, 0)
        )
        dbias_strides = (
            (
                d_bias.stride(0),
                d_bias.stride(1),
                d_bias.stride(2),
                d_bias.stride(3),
            )
            if ctx.has_bias
            else (0, 0, 0, 0)
        )

        # 2) dQ kernel: grid (Q-blocks, B*H)
        def dq_grid(META: dict) -> tuple:
            return (triton.cdiv(seqlen_q, META["BLOCK_M"]), B * H)

        _fused_bwd_dq_kernel[dq_grid](
            q,
            k,
            v,
            bias if ctx.has_bias else q,
            mask_bias if ctx.has_mask_bias else q,
            do,
            lse,
            delta,
            dq,
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
            mask_bias_strides[0],
            mask_bias_strides[1],
            do.stride(0),
            do.stride(1),
            do.stride(2),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            H,
            seqlen_q,
            seqlen_k,
            d,
            HAS_BIAS=ctx.has_bias,
            HAS_MASK_BIAS=ctx.has_mask_bias,
            M=M,
            NW=NW,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            INPUT_PRECISION=matmul_precision,
        )

        # 3) dKV kernel: grid (K-blocks, B*H)
        def dkv_grid(META: dict) -> tuple:
            return (triton.cdiv(seqlen_k, META["BLOCK_N"]), B * H)

        _fused_bwd_dkv_kernel[dkv_grid](
            q,
            k,
            v,
            bias if ctx.has_bias else q,
            mask_bias if ctx.has_mask_bias else q,
            do,
            lse,
            delta,
            dk,
            dv,
            d_bias if ctx.has_bias else q,
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
            mask_bias_strides[0],
            mask_bias_strides[1],
            do.stride(0),
            do.stride(1),
            do.stride(2),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dbias_strides[0],
            dbias_strides[1],
            dbias_strides[2],
            dbias_strides[3],
            H,
            seqlen_q,
            seqlen_k,
            d,
            HAS_BIAS=ctx.has_bias,
            HAS_MASK_BIAS=ctx.has_mask_bias,
            M=M,
            NW=NW,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            INPUT_PRECISION=matmul_precision,
        )

        # d_bias is computed at the folded batch size [folded, H, W, H_keys].
        # Reduce over the multiplicity axis M to match the windowed z-bias input
        # shape [B*NW, H, W, H_keys]. The fold is off_b = (b*M + m)*NW + k, so
        # view(B, M, NW, ...) splits (b, m, k) and summing dim 1 collapses M to
        # [B, NW, ...] = [b*NW + k, ...], exactly the windowed bias layout.
        # With M=1 this is a no-op reshape (identity), so the full-attention path
        # is unchanged.
        if ctx.has_bias:
            n_windowed = bias.shape[0]  # B * NW
            B_real = n_windowed // NW
            d_bias_out = d_bias.view(B_real, M, NW, H, seqlen_q, seqlen_k).sum(1)
            d_bias_out = d_bias_out.reshape(n_windowed, H, seqlen_q, seqlen_k)
        else:
            d_bias_out = None

        return dq, dk, dv, d_bias_out, None, None, None, None, None


def fused_flash_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    bias: Tensor | None = None,
    softmax_scale: float | None = None,
    matmul_precision: str = "ieee",
    mask_bias: Tensor | None = None,
    M: int = 1,
    NW: int = 1,
) -> Tensor:
    """Autotuned FlashAttention forward with optional additive pair bias.

    Forward is online-softmax FlashAttention; backward is the explicit FA-2 path
    (3 Triton kernels) — no materialised [B, H, N, N] intermediates.

    Args:
        q, k, v: ``[folded, num_heads, seqlen, head_dim]``, where ``folded`` is
            the batch axis the kernel iterates over.
        bias: optional additive pair bias. Its leading dim is the *un-M-folded*
            (windowed) batch ``[folded // M, num_heads, seqlen_q, seqlen_k]``; the
            kernel broadcasts it over the multiplicity axis ``M`` via strides
            (``bz = (off_b // (M*NW))*NW + off_b % NW``) instead of densifying.
            With the defaults ``M=1, NW=1`` the bias is indexed by ``off_b`` (the
            plain dense/full-attention path, unchanged).
        softmax_scale: attention scale; defaults to ``head_dim ** -0.5``.
        matmul_precision: one of ``{"ieee", "tf32", "tf32x3"}``.
        mask_bias: optional per-folded key-mask additive ``[folded, seqlen_k]``,
            broadcast over heads and query rows inside the kernel. Kept separate
            from ``bias`` because the key mask varies per folded window and so
            cannot be M-broadcast. Has no gradient.
        M: multiplicity broadcast factor for ``bias`` (folded = B*M*NW).
        NW: number of windows per (batch, multiplicity) group.
    """
    assert q.shape[-1] <= 128, f"head_dim {q.shape[-1]} > 128 not supported"
    if matmul_precision not in _SUPPORTED_PRECISIONS:
        raise ValueError(
            f"fused_flash_attn matmul_precision must be one of {sorted(_SUPPORTED_PRECISIONS)}; "
            f"got {matmul_precision!r}."
        )

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    return _FusedFlashAttnFn.apply(q, k, v, bias, softmax_scale, matmul_precision, mask_bias, M, NW)


# Public alias used by AttentionPairBiasTritonFlash — the autograd.Function that wraps
# the Triton fwd/bwd kernels for full attention with pair bias.
_TritonFlashAPBFn = _FusedFlashAttnFn
