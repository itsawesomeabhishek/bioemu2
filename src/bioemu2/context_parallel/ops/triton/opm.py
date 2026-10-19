import triton
import triton.language as tl


@triton.jit
def _opm_ring_accum_kernel(
    a,
    b,
    weight,
    output,
    tile_i: tl.constexpr,
    n_tokens: tl.constexpr,
    msa_depth: tl.constexpr,
    hidden_dim: tl.constexpr,
    output_dim: tl.constexpr,
    output_dim_padded: tl.constexpr,
    stride_as: tl.constexpr,
    stride_ai: tl.constexpr,
    stride_ac: tl.constexpr,
    stride_bs: tl.constexpr,
    stride_bj: tl.constexpr,
    stride_bc: tl.constexpr,
    stride_wc: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_oi: tl.constexpr,
    stride_oj: tl.constexpr,
    stride_od: tl.constexpr,
    block_i: tl.constexpr,
    block_j: tl.constexpr,
    block_s: tl.constexpr,
    block_c: tl.constexpr,
    input_precision: tl.constexpr,
):
    """Add ``proj_o(sum_s a[s, i] (x) b[s, j])`` to a ``block_i x block_j`` output tile.

    For each group of ``block_c`` left channels the kernel contracts the MSA rows into the
    outer-product slice ``outer[(c, i), (j, c')] = sum_s a[s, i, c] * b[s, j, c']`` and
    immediately projects it with the weight rows ``c * hidden_dim + c'``. Both steps are
    ``tl.dot`` calls, so the full ``C x C`` outer product never leaves registers.
    """
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    i_offsets = pid_i * block_i + tl.arange(0, block_i)
    i_mask = i_offsets < tile_i
    jc_offsets = tl.arange(0, block_j * hidden_dim)
    j_of_jc = pid_j * block_j + jc_offsets // hidden_dim
    c_of_jc = jc_offsets % hidden_dim
    jc_mask = j_of_jc < n_tokens
    ci_offsets = tl.arange(0, block_c * block_i)
    c_of_ci = ci_offsets // block_i
    i_of_ci = pid_i * block_i + ci_offsets % block_i
    ci_mask = i_of_ci < tile_i
    cc_offsets = tl.arange(0, block_c * hidden_dim)
    d_offsets = tl.arange(0, output_dim_padded)
    s_offsets = tl.arange(0, block_s)
    accumulator = tl.zeros((block_i * block_j, output_dim_padded), dtype=tl.float32)

    for c_start in tl.range(0, hidden_dim, block_c, num_stages=1):
        outer = tl.zeros((block_c * block_i, block_j * hidden_dim), dtype=tl.float32)
        for msa_start in range(0, msa_depth, block_s):
            msa_idx = msa_start + s_offsets
            msa_mask = msa_idx < msa_depth
            a_values = tl.load(
                a
                + msa_idx[None, :] * stride_as
                + i_of_ci[:, None] * stride_ai
                + (c_start + c_of_ci)[:, None] * stride_ac,
                mask=ci_mask[:, None] & msa_mask[None, :],
                other=0.0,
            )
            b_values = tl.load(
                b
                + msa_idx[:, None] * stride_bs
                + j_of_jc[None, :] * stride_bj
                + c_of_jc[None, :] * stride_bc,
                mask=msa_mask[:, None] & jc_mask[None, :],
                other=0.0,
            )
            outer += tl.dot(
                a_values,
                b_values,
                input_precision=input_precision,
            )
        weight_slice = tl.load(
            weight
            + (c_start * hidden_dim + cc_offsets)[:, None] * stride_wc
            + d_offsets[None, :] * stride_wd,
        ).to(tl.float32)
        outer = tl.reshape(outer, (block_c, block_i, block_j, hidden_dim))
        outer = tl.permute(outer, (1, 2, 0, 3))
        accumulator += tl.dot(
            tl.reshape(outer, (block_i * block_j, block_c * hidden_dim)),
            weight_slice,
            input_precision=input_precision,
        )

    j_offsets = pid_j * block_j + tl.arange(0, block_j)
    out_mask = (
        i_mask[:, None, None]
        & (j_offsets < n_tokens)[None, :, None]
        & (d_offsets < output_dim)[None, None, :]
    )
    output_ptrs = (
        output
        + i_offsets[:, None, None] * stride_oi
        + j_offsets[None, :, None] * stride_oj
        + d_offsets[None, None, :] * stride_od
    )
    existing = tl.load(output_ptrs, mask=out_mask, other=0.0)
    update = tl.reshape(accumulator, (block_i, block_j, output_dim_padded))
    tl.store(output_ptrs, (existing + update).to(existing.dtype), mask=out_mask)
