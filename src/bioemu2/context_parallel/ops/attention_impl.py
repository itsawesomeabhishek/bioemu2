"""Context-parallel AttentionPairBias autograd functions.

No serial counterpart — distributed-only. Ported from Fold-CP attention_impl.py,
trimmed to the shardwise and context-vec impls used by bioemu2.
"""

from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import FunctionCtx
from torch.distributed.tensor import DTensor, Replicate, Shard

from bioemu2 import HAS_TRITON
from bioemu2.context_parallel.distributed.comm import AttentionPairBiasComm
from bioemu2.context_parallel.distributed.utils import (
    tiled_softmax_attention_update,
    update_exhaustive_strides,
)
from bioemu2.context_parallel.ops.distributed_primitives.dtensor_metadata_tools import (
    raise_if_incorrect_dtensor_metadata_args,
)

if HAS_TRITON:
    from bioemu2.context_parallel.ops.triton.flash_attn import fused_flash_attn
    from bioemu2.context_parallel.ops.triton.flash_attn_impl import _fused_flash_attn_with_lse


@contextmanager
def _disable_tf32():
    """Temporarily disable TF32 matmul for FP32 precision (no env-var mutation)."""
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


class _AttentionPairBiasContextVecParams(NamedTuple):
    """NamedTuple for attention pair bias context vector parameters."""

    ring_comm: AttentionPairBiasComm
    multiplicity: int
    num_heads: int
    head_dim: int
    inf: float
    use_window_batching: bool


class _AttentionPairBiasShardwiseImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: DTensor,
        k: DTensor,
        v: DTensor,
        z: DTensor,
        d_bias: DTensor | None,
        mask_key: DTensor | None,
        num_heads: int,
        head_dim: int,
        inf: float,
    ) -> DTensor:
        if q.ndim != 4:  # (B * M, K, W, D)
            raise ValueError(f"Input q must have 4 dimensions. Got {q.ndim}.")

        if k.ndim != 4:  # (B * M, K, H, D)
            raise ValueError(f"Input k must have 4 dimensions. Got {k.ndim}.")

        if v.ndim != 4:  # (B * M, K, H, D)
            raise ValueError(f"Input v must have 4 dimensions. Got {v.ndim}.")

        if z.ndim != 5:  # (B, K, W, H, num_heads) with potentially no multiplicity
            raise ValueError(f"Input z must have 5 dimensions. Got {z.ndim}.")

        if mask_key is not None:
            if mask_key.ndim != 3:  # (B, K, H) with potentially no multiplicity
                raise ValueError(f"Input mask_key must have 3 dimensions. Got {mask_key.ndim}.")

            if mask_key.shape != z.shape[:2] + (z.shape[3],):
                raise ValueError(
                    f"Input mask_key must have the same shape as z.shape[:3]. Got {mask_key.shape} and {z.shape[:3]}."
                )

        # Shape checks on the input
        if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:  # B, K
            raise ValueError(
                f"Input q, k, v must have the same leading two B and K dimensions. Got {q.shape[:2]} and {k.shape[:2]} and {v.shape[:2]}"
            )

        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:  # D
            raise ValueError(
                f"Input q, k, v must have the same last dimension. Got {q.shape[-1]} and {k.shape[-1]} and {v.shape[-1]}."
            )

        if q.shape[-1] != num_heads * head_dim:
            raise ValueError(
                f"Input q.shape[-1] and num_heads * head_dim must have the same shape. Got {q.shape[-1]} and {num_heads * head_dim}."
            )

        if k.shape != v.shape:
            raise ValueError(
                f"Input k and v must have the same shape. Got {k.shape} and {v.shape}."
            )

        if q.shape[0] % z.shape[0] != 0:  # B * M % B == 0
            raise ValueError(
                f"Input q.shape[0] must be a multiple of z.shape[0]. Got q.shape[0]={q.shape[0]} and z.shape[0]={z.shape[0]}."
            )

        if z.shape[1:3] != q.shape[1:3]:  # K, W
            raise ValueError(
                f"Input q.shape[1:3] and z.shape[1:3] must have the same shape. Got {q.shape[1:3]} and {z.shape[1:3]}."
            )

        if z.shape[3] != k.shape[2]:  # H
            raise ValueError(
                f"Input z.shape[3] and k.shape[2] must have the same shape. Got {z.shape[3]} and {k.shape[2]}."
            )

        if z.shape[-1] != num_heads:
            raise ValueError(
                f"Input z.shape[-1] and num_heads must have the same shape. Got {z.shape[-1]} and {num_heads}."
            )

        # placements and device mesh checks
        if (
            q.placements != k.placements
            or q.placements != v.placements
            or q.placements != z.placements
            or q.placements != mask_key.placements  # type: ignore[union-attr]
        ):
            raise ValueError(
                f"Input q, k, v, z, and mask must have the same placements. Got {q.placements} and {k.placements} and {v.placements} and {z.placements} and {mask_key.placements}."  # type: ignore[union-attr]
            )
        if (
            q.device_mesh != k.device_mesh
            or q.device_mesh != v.device_mesh
            or q.device_mesh != z.device_mesh
            or q.device_mesh != mask_key.device_mesh  # type: ignore[union-attr]
        ):
            raise ValueError(
                f"Input q, k, v, z, and mask must be on the same device mesh. Got {q.device_mesh} and {k.device_mesh} and {v.device_mesh} and {z.device_mesh} and {mask_key.device_mesh}."  # type: ignore[union-attr]
            )

        multiplicity = q.shape[0] // z.shape[0]

        q_local_orig = q.to_local().detach().requires_grad_(q.requires_grad)
        k_local_orig = k.to_local().detach().requires_grad_(k.requires_grad)
        v_local_orig = v.to_local().detach().requires_grad_(v.requires_grad)
        z_local_orig = (
            z.to_local().detach().requires_grad_(z.requires_grad)
        )  # (B, K, W, H, num_heads)
        if d_bias is not None:
            # Unlike z, d_bias carries the multiplicity axis (it is computed from s,
            # which has shape [B*M, K, W, c_s]). It is full [B*M, K, W, H, num_heads]
            # — no broadcast over M.
            d_bias_local_orig = d_bias.to_local().detach().requires_grad_(d_bias.requires_grad)
        else:
            d_bias_local_orig = None
        if mask_key is not None:
            mask_key_bool_local = (
                mask_key.to_local().detach().bool()
            )  # (B, K, H) — True = valid, False = pad
        else:
            mask_key_bool_local = None

        with torch.enable_grad():
            q_local = q_local_orig.unflatten(
                -1, (num_heads, head_dim)
            )  # (B * M, K, W, num_heads, head_dim)
            k_local = k_local_orig.unflatten(
                -1, (num_heads, head_dim)
            )  # (B * M, K, H, num_heads, head_dim)
            v_local = v_local_orig.unflatten(
                -1, (num_heads, head_dim)
            )  # (B * M, K, H, num_heads, head_dim)
            z_local = z_local_orig

            # Mask shape for masked_fill_: (B, K, 1, H, 1) broadcasts over M, W, heads
            if mask_key_bool_local is not None:
                mask_key_view = mask_key_bool_local[:, :, None, :, None]  # (B, K, 1, H, 1)
            else:
                mask_key_view = None

            if multiplicity > 1:
                # unflatten the multiplicity axis so that mask and z are broadcasted along it
                # This has to be (B * multiplicity, ...) -> (B, multiplicity, ...)
                # but never (B * multiplicity, ...) -> (multiplicity, B, ...) due to the upstream
                # order of multiplicity application
                q_local = q_local.unflatten(0, (-1, multiplicity))
                k_local = k_local.unflatten(0, (-1, multiplicity))
                v_local = v_local.unflatten(0, (-1, multiplicity))
                z_local = z_local.unsqueeze(1)
                if mask_key_view is not None:
                    mask_key_view = mask_key_view.unsqueeze(1)
                if d_bias_local_orig is not None:
                    # d_bias has full M, so unflatten its leading dim (no broadcast).
                    d_bias_local = d_bias_local_orig.unflatten(0, (-1, multiplicity))
                else:
                    d_bias_local = None
            else:
                d_bias_local = d_bias_local_orig

            # use at least FP32 for AttnPairBias
            dtype_compute = torch.promote_types(q_local.dtype, torch.float32)
            q_local = q_local.to(dtype_compute)
            k_local = k_local.to(dtype_compute)
            v_local = v_local.to(dtype_compute)
            z_local = z_local.to(dtype_compute)
            if d_bias_local is not None:
                d_bias_local = d_bias_local.to(dtype_compute)

            with _disable_tf32(), torch.amp.autocast("cuda", enabled=False):
                attn = torch.einsum("...wnd,...hnd->...whn", q_local, k_local)
                attn = attn / head_dim**0.5
                attn = attn + z_local
                if d_bias_local is not None:
                    attn = attn + d_bias_local
                if mask_key_view is not None:
                    attn.masked_fill_(~mask_key_view, -inf)
                attn = attn.softmax(dim=-2)  # axis = -2 is the key dimension
                o_local = torch.einsum(
                    "...whn,...hnd->...wnd", attn, v_local
                )  # (B, K, W, num_heads, head_dim)
            # (..., W, num_heads, head_dim) -> (..., W, num_heads * head_dim)
            o_local = o_local.flatten(-2, -1).to(q.dtype)  # (B, K, W, c_s)

            if multiplicity > 1:
                # (B, multiplicity, ...) -> (B * multiplicity, ...)
                o_local = o_local.flatten(0, 1)

        # save tensors for backward pass
        ctx.save_for_backward(
            q_local_orig,
            k_local_orig,
            v_local_orig,
            z_local_orig,
            d_bias_local_orig,
            mask_key_bool_local,
            o_local,
        )
        ctx.device_mesh = q.device_mesh
        ctx.placements = q.placements
        ctx.q_shape = q.shape
        ctx.q_stride = q.stride()
        ctx.k_shape = k.shape
        ctx.k_stride = k.stride()
        ctx.v_shape = v.shape
        ctx.v_stride = v.stride()
        ctx.z_shape = z.shape
        ctx.z_stride = z.stride()
        ctx.d_bias_shape = d_bias.shape if d_bias is not None else None
        ctx.d_bias_stride = d_bias.stride() if d_bias is not None else None

        o_dtensor = DTensor.from_local(
            o_local.detach(),
            device_mesh=q.device_mesh,
            placements=q.placements,
            shape=q.shape,
            stride=q.stride(),
        )

        return o_dtensor

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: DTensor,
    ) -> tuple[
        DTensor | None,
        DTensor | None,
        DTensor | None,
        DTensor | None,
        DTensor | None,
        None,
        None,
        None,
        None,
    ]:
        # retrieve the leaf nodes for the local graph
        (
            q_local,
            k_local,
            v_local,
            z_local,
            d_bias_local,
            _,
            o_local,
        ) = ctx.saved_tensors
        leaves = (q_local, k_local, v_local, z_local, d_bias_local)
        inputs_needing_grad = [t for t in leaves if t is not None and t.requires_grad]
        if not inputs_needing_grad:
            return None, None, None, None, None, None, None, None, None
        grad_output_local = grad_output.to_local()
        with _disable_tf32(), torch.amp.autocast("cuda", enabled=False):
            grads_local = torch.autograd.grad(
                outputs=[o_local],
                inputs=inputs_needing_grad,
                grad_outputs=[grad_output_local],
                retain_graph=False,
            )

        iter_grads_local = iter(grads_local)
        dq_local = next(iter_grads_local) if q_local.requires_grad else None
        dk_local = next(iter_grads_local) if k_local.requires_grad else None
        dv_local = next(iter_grads_local) if v_local.requires_grad else None
        dz_local = next(iter_grads_local) if z_local.requires_grad else None
        if d_bias_local is not None and d_bias_local.requires_grad:
            dd_bias_local = next(iter_grads_local)
        else:
            dd_bias_local = None

        def _to_dtensor(t, shape, stride):
            if t is None:
                return None
            return DTensor.from_local(
                t,
                device_mesh=ctx.device_mesh,
                placements=ctx.placements,
                shape=shape,
                stride=stride,
            )

        dq = _to_dtensor(dq_local, ctx.q_shape, ctx.q_stride)
        dk = _to_dtensor(dk_local, ctx.k_shape, ctx.k_stride)
        dv = _to_dtensor(dv_local, ctx.v_shape, ctx.v_stride)
        dz = _to_dtensor(dz_local, ctx.z_shape, ctx.z_stride)
        dd_bias = _to_dtensor(dd_bias_local, ctx.d_bias_shape, ctx.d_bias_stride)

        return dq, dk, dv, dz, dd_bias, None, None, None, None


class _AttentionPairBiasShardwiseTritonFlashImpl(torch.autograd.Function):
    """Triton fused-flash shardwise APB.

    Same DTensor-wrapping pattern as :class:`_AttentionPairBiasShardwiseImpl`:
    saves leaf locals with requires_grad, builds the autograd graph by calling
    ``fused_flash_attn`` on a flattened ``[B*M*K, num_heads, W, head_dim]`` view,
    and lets ``torch.autograd.grad`` produce dq/dk/dv/d_bias_eff in backward.

    The triton kernel needs a fully-materialised rank-4 bias of shape
    ``[B*M*K, num_heads, W, H]``; we build it by adding z (broadcast over M),
    optional d_bias (already has M), and mask_bias (broadcast over M).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: DTensor,
        k: DTensor,
        v: DTensor,
        z: DTensor,
        d_bias: DTensor | None,
        mask_key: DTensor | None,
        num_heads: int,
        head_dim: int,
        inf: float,
        matmul_precision: str,
    ) -> DTensor:
        if not HAS_TRITON:
            raise RuntimeError("Context-parallel FlashAttention requires Triton on Linux.")
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"q/k/v must be 4D (B*M, K, W/H, c_s); got {q.ndim}/{k.ndim}/{v.ndim}")
        if z.ndim != 5:
            raise ValueError(f"z must be 5D (B, K, W, H, num_heads); got {z.ndim}")
        if q.shape[0] % z.shape[0] != 0:
            raise ValueError(
                f"q.shape[0]={q.shape[0]} must be divisible by z.shape[0]={z.shape[0]}"
            )

        multiplicity = q.shape[0] // z.shape[0]

        q_local_orig = q.to_local().detach().requires_grad_(q.requires_grad)
        k_local_orig = k.to_local().detach().requires_grad_(k.requires_grad)
        v_local_orig = v.to_local().detach().requires_grad_(v.requires_grad)
        z_local_orig = z.to_local().detach().requires_grad_(z.requires_grad)
        if d_bias is not None:
            d_bias_local_orig = d_bias.to_local().detach().requires_grad_(d_bias.requires_grad)
        else:
            d_bias_local_orig = None
        if mask_key is not None:
            mask_key_local_orig = mask_key.to_local().detach().bool()
        else:
            mask_key_local_orig = None

        with torch.enable_grad():
            B_local = z_local_orig.shape[0]
            K_local, W = q_local_orig.shape[1], q_local_orig.shape[2]
            H_keys = k_local_orig.shape[2]
            dtype_compute = torch.promote_types(q_local_orig.dtype, torch.float32)

            def _to_kernel_layout(t, seq_len):
                return (
                    t.unflatten(0, (B_local, multiplicity))
                    .unflatten(-1, (num_heads, head_dim))
                    .permute(0, 1, 2, 4, 3, 5)
                    .reshape(B_local * multiplicity * K_local, num_heads, seq_len, head_dim)
                    .to(dtype_compute)
                    .contiguous()
                )

            q_bh = _to_kernel_layout(q_local_orig, W)
            k_bh = _to_kernel_layout(k_local_orig, H_keys)
            v_bh = _to_kernel_layout(v_local_orig, H_keys)

            bias = (
                z_local_orig.unsqueeze(1)
                .expand(B_local, multiplicity, K_local, W, H_keys, num_heads)
                .permute(0, 1, 2, 5, 3, 4)  # (B, M, K, num_heads, W, H)
                .to(dtype_compute)
                .contiguous()
            )

            if d_bias_local_orig is not None:
                # d_bias: (B*M, K, W, H, num_heads) — already has M.
                d_bias_local = (
                    d_bias_local_orig.unflatten(0, (B_local, multiplicity))
                    .permute(0, 1, 2, 5, 3, 4)  # (B, M, K, num_heads, W, H)
                    .to(dtype_compute)
                )
                bias = bias + d_bias_local

            if mask_key_local_orig is not None:
                # mask_key: (B, K, H) bool — True=valid, False=pad. Broadcast over M, num_heads, W.
                mask_expand = ~mask_key_local_orig[:, None, :, None, None, :]  # (B, 1, K, 1, 1, H)
                bias.masked_fill_(mask_expand, -inf)

            bias_flat = bias.reshape(B_local * multiplicity * K_local, num_heads, W, H_keys)

            with _disable_tf32(), torch.amp.autocast("cuda", enabled=False):
                o_bh = fused_flash_attn(
                    q_bh,
                    k_bh,
                    v_bh,
                    bias=bias_flat,
                    softmax_scale=head_dim**-0.5,
                    matmul_precision=matmul_precision,
                )  # (B*M*K, num_heads, W, head_dim)

            # Inverse of _to_kernel_layout, then flatten num_heads*head_dim back to c_s.
            o_local = (
                o_bh.reshape(B_local, multiplicity, K_local, num_heads, W, head_dim)
                .permute(0, 1, 2, 4, 3, 5)  # (B, M, K, W, num_heads, head_dim)
                .reshape(B_local * multiplicity, K_local, W, num_heads * head_dim)
                .to(q.dtype)
            )

        ctx.save_for_backward(
            q_local_orig,
            k_local_orig,
            v_local_orig,
            z_local_orig,
            d_bias_local_orig,
            mask_key_local_orig,
            o_local,
        )
        ctx.device_mesh = q.device_mesh
        ctx.placements = q.placements
        ctx.q_shape = q.shape
        ctx.q_stride = q.stride()
        ctx.k_shape = k.shape
        ctx.k_stride = k.stride()
        ctx.v_shape = v.shape
        ctx.v_stride = v.stride()
        ctx.z_shape = z.shape
        ctx.z_stride = z.stride()
        ctx.d_bias_shape = d_bias.shape if d_bias is not None else None
        ctx.d_bias_stride = d_bias.stride() if d_bias is not None else None

        return DTensor.from_local(
            o_local.detach(),
            device_mesh=q.device_mesh,
            placements=q.placements,
            shape=q.shape,
            stride=q.stride(),
        )

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: DTensor,
    ) -> tuple[
        DTensor | None,
        DTensor | None,
        DTensor | None,
        DTensor | None,
        DTensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            q_local,
            k_local,
            v_local,
            z_local,
            d_bias_local,
            _,
            o_local,
        ) = ctx.saved_tensors
        leaves = (q_local, k_local, v_local, z_local, d_bias_local)
        inputs_needing_grad = [t for t in leaves if t is not None and t.requires_grad]
        if not inputs_needing_grad:
            return None, None, None, None, None, None, None, None, None, None
        grad_output_local = grad_output.to_local()
        with _disable_tf32(), torch.amp.autocast("cuda", enabled=False):
            grads_local = torch.autograd.grad(
                outputs=[o_local],
                inputs=inputs_needing_grad,
                grad_outputs=[grad_output_local],
                retain_graph=False,
            )

        iter_grads_local = iter(grads_local)
        dq_local = next(iter_grads_local) if q_local.requires_grad else None
        dk_local = next(iter_grads_local) if k_local.requires_grad else None
        dv_local = next(iter_grads_local) if v_local.requires_grad else None
        dz_local = next(iter_grads_local) if z_local.requires_grad else None
        if d_bias_local is not None and d_bias_local.requires_grad:
            dd_bias_local = next(iter_grads_local)
        else:
            dd_bias_local = None

        def _to_dtensor(t, shape, stride):
            if t is None:
                return None
            return DTensor.from_local(
                t,
                device_mesh=ctx.device_mesh,
                placements=ctx.placements,
                shape=shape,
                stride=stride,
            )

        return (
            _to_dtensor(dq_local, ctx.q_shape, ctx.q_stride),
            _to_dtensor(dk_local, ctx.k_shape, ctx.k_stride),
            _to_dtensor(dv_local, ctx.v_shape, ctx.v_stride),
            _to_dtensor(dz_local, ctx.z_shape, ctx.z_stride),
            _to_dtensor(dd_bias_local, ctx.d_bias_shape, ctx.d_bias_stride),
            None,  # mask_key
            None,  # num_heads
            None,  # head_dim
            None,  # inf
            None,  # matmul_precision
        )


class _AttentionPairBiasContexVecImpl(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        q: DTensor,
        k: DTensor,
        v: DTensor,
        z: DTensor,
        mask: DTensor,
        pair_mask: DTensor | None,
        apb_context_vec_params: _AttentionPairBiasContextVecParams,
    ) -> DTensor:
        """

        c_s = num_heads * head_dim checked in vanilla AttentionPairBias.__init__

        Below, N is the global number of tokens, H is the number of heads.

        Parameters
        ----------
        ctx: FunctionCtx
            The context object.
        q : DTensor
            query vectors computed by projection, (B, N, c_s)
        k : DTensor
            key vectors computed by projection, (B, N, c_s)
        v : DTensor
            value vectors computed by projection, (B, N, c_s)
        z : DTensor
            z, (B, H, N, N)
        mask : torch.Tensor
            The pairwise mask tensor (B, N)
        multiplicity : int, optional
            The diffusion batch size, by default 1
        pair_mask: Union[DTensor, None]
            The pairwise mask tensor.
        apb_context_vec_params: tuple
            The parameters for the attention pair bias context vector.

        Key features:
            - Distributed computation across device meshes with various sharding strategies
            - Memory-efficient implementation that operates on local tensor chunks
            - Supports gradient computation through custom backward pass
            - Validates tensor compatibility (type, device mesh, placements, shapes)`

        Raises
        ------
        TypeError
            If dtensor_instance is not a DTensor.
        ValueError
            If the DTensor metadata is incorrect.
        """
        # Check input metadata
        _AttentionPairBiasContexVecImpl.check_forward_input_metadata_and_store(
            ctx, q, k, v, z, mask, pair_mask, apb_context_vec_params
        )
        # Check implementation scope
        _AttentionPairBiasContexVecImpl.check_forward_input_for_impl_state(apb_context_vec_params)

        # Setup inputs to ring_attention_simple_forward()
        (
            ring_comm,
            multiplicity,
            _,
            _,
            inf,
            use_window_batching,
        ) = apb_context_vec_params

        requires_grad = any(p.requires_grad for p in (q, k, v, z))

        ctx.mark_non_differentiable(mask)
        mask_local: Tensor = mask.to_local()

        # overlay mask comm with qkv projection, pair_mask_ij <- pair_mask_ij + mask_j
        mask_recv: Tensor = ring_comm.comm_transpose_mask.enqueue_to_dispatch(
            mask_local.contiguous()
        )

        dtype_input = q.dtype
        dtype_compute = torch.promote_types(dtype_input, torch.float32)
        ctx.dtype_compute = dtype_compute
        ctx.dtype_input = dtype_input

        q_local: Tensor = q.to_local().to(dtype=dtype_compute)
        k_local: Tensor = k.to_local().to(dtype=dtype_compute)
        v_local: Tensor = v.to_local().to(dtype=dtype_compute)
        z_local: Tensor = z.to_local().to(dtype=dtype_compute)

        ctx.B_each_chunk = q_local.shape[0]
        ctx.N_each_chunk = q_local.shape[1]

        single_rep_view_shape = ctx.B_each_chunk, ctx.N_each_chunk, ctx.H, ctx.head_dim
        q_local = q_local.view(single_rep_view_shape).requires_grad_(q.requires_grad)
        k_local = k_local.view(single_rep_view_shape).requires_grad_(k.requires_grad)
        v_local = v_local.view(single_rep_view_shape).requires_grad_(v.requires_grad)
        z_local = z_local.permute(0, 3, 1, 2).requires_grad_(z.requires_grad)

        if requires_grad:
            ctx.multiplicity = multiplicity

        ring_comm.comm_transpose_mask.wait_until_finished()

        if use_window_batching or pair_mask is None:  # mask-only attention path
            pair_mask_local = mask_recv[:, None, None, :]
        else:  # only atom-level has pair_mask
            ctx.mark_non_differentiable(pair_mask)
            pair_mask_local = pair_mask.to_local()  # shape = (B, I, J)
            pair_mask_local = pair_mask_local[:, None, :, :] * mask_recv[:, None, None, :]

        pair_mask_local = pair_mask_local.to(dtype=dtype_compute)

        with torch.autocast("cuda", enabled=False):
            o_local, ring_attention_simple_data_for_bw = ring_attention_simple_forward(
                q_local,
                k_local,
                v_local,
                z_local,
                pair_mask_local,
                ring_comm,
                inf,
            )
            if requires_grad:
                # Unpack tensors for save_for_backward to enable automatic memory management and hook support
                ctx.save_for_backward(
                    ring_attention_simple_data_for_bw.q_store,
                    ring_attention_simple_data_for_bw.k_t_store,
                    ring_attention_simple_data_for_bw.v_t_store,
                    ring_attention_simple_data_for_bw.z_store,
                    ring_attention_simple_data_for_bw.lse_m,
                    ring_attention_simple_data_for_bw.o_store,
                )
                ctx.ring_comm = ring_attention_simple_data_for_bw.ring_comm

        o_local = o_local.reshape(ctx.B_each_chunk, ctx.N_each_chunk, ctx.c_s)  # o_local_b

        # Compute output shape and stride
        shape_output = v.shape[:-1] + (o_local.shape[-1],)

        strides_output = update_exhaustive_strides(o_local.shape, o_local.stride(), shape_output)

        o = DTensor.from_local(
            o_local.to(dtype=dtype_input),
            device_mesh=ctx.device_mesh,
            placements=ctx.single_rep_placements,
            shape=shape_output,
            stride=strides_output,
        )
        return o

    @staticmethod
    def backward(
        ctx: FunctionCtx,
        grad_output: DTensor,
    ) -> tuple[DTensor, DTensor, DTensor, DTensor, None, None, None]:
        """Backward pass implementation.

        Parameters
        ----------
        ctx: FunctionCtx
            The context object.
        grad_output: DTensor
            The gradient of the output tensor.

        Raises
        ------
        TypeError
            If dtensor_instance is not a DTensor.
        ValueError
            If the DTensor metadata is incorrect.
        """
        _AttentionPairBiasContexVecImpl.check_backward_input_metadata(ctx, grad_output)

        # Get ref to local tensor, and reshape to (B, I, H, D)
        grad_output_local: Tensor = grad_output.to_local().reshape(
            ctx.B_each_chunk, ctx.N_each_chunk, ctx.H, ctx.head_dim
        )
        grad_output_local = grad_output_local.to(dtype=ctx.dtype_compute)

        # Run distributed attention backward.
        q_store, k_t_store, v_t_store, z_store, lse_m, o_store = ctx.saved_tensors
        data_for_backward = RingAttentionSimpleDataForBackward(
            q_store=q_store,
            k_t_store=k_t_store,
            v_t_store=v_t_store,
            z_store=z_store,
            lse_m=lse_m,
            ring_comm=ctx.ring_comm,
            o_store=o_store,
            multiplicity=ctx.multiplicity,
        )
        del ctx.ring_comm

        grad_q, grad_k, grad_v, grad_z = ring_attention_simple_backward(
            data_for_backward=data_for_backward,
            do=grad_output_local,
        )
        del data_for_backward  # free up q_store, k_t_store, v_t_store, z_store, lse_m immediately
        grad_q = grad_q.to(dtype=ctx.dtype_input)  # (B_each_chunk, N_each_chunk, H, D)
        grad_k = grad_k.to(dtype=ctx.dtype_input)  # (B_each_chunk, N_each_chunk, H, D)
        grad_v = grad_v.to(dtype=ctx.dtype_input)  # (B_each_chunk, N_each_chunk, H, D)
        grad_z = grad_z.to(dtype=ctx.dtype_input)  # (B_each_chunk, N_each_chunk, H, D)

        # Reshape, allocate new memory
        single_rep_target_shape = (ctx.B_each_chunk, ctx.N_each_chunk, ctx.c_s)
        grad_q_flat: Tensor = grad_q.reshape(single_rep_target_shape)
        grad_k_flat: Tensor = grad_k.reshape(single_rep_target_shape)
        grad_v_flat: Tensor = grad_v.reshape(single_rep_target_shape)

        grad_z_flat: Tensor = grad_z.permute((0, 2, 3, 1))

        grad_q_dtensor = DTensor.from_local(
            grad_q_flat,
            device_mesh=ctx.device_mesh,
            placements=ctx.single_rep_placements,
            shape=ctx.shape_q,
            stride=ctx.stride_q,
        )
        grad_k_dtensor = DTensor.from_local(
            grad_k_flat,
            device_mesh=ctx.device_mesh,
            placements=ctx.single_rep_placements,
            shape=ctx.shape_k,
            stride=ctx.stride_k,
        )
        grad_v_dtensor = DTensor.from_local(
            grad_v_flat,
            device_mesh=ctx.device_mesh,
            placements=ctx.single_rep_placements,
            shape=ctx.shape_v,
            stride=ctx.stride_v,
        )
        grad_z_dtensor = DTensor.from_local(
            grad_z_flat,
            device_mesh=ctx.device_mesh,
            placements=ctx.pair_rep_placements,
            shape=ctx.shape_z,
            stride=ctx.stride_z,
        )
        _AttentionPairBiasContexVecImpl.check_backward_output_metadata(
            ctx,
            grad_q_dtensor,
            grad_k_dtensor,
            grad_v_dtensor,
        )
        return (
            grad_q_dtensor,
            grad_k_dtensor,
            grad_v_dtensor,
            grad_z_dtensor,
            None,
            None,
            None,
        )

    @staticmethod
    def check_forward_input_metadata_and_store(
        ctx: FunctionCtx,
        q: DTensor,
        k: DTensor,
        v: DTensor,
        z: DTensor,
        mask: DTensor,
        pair_mask: DTensor | None,
        apb_context_vec_params: _AttentionPairBiasContextVecParams,
    ) -> None:
        (
            _,
            _,
            num_heads,
            head_dim,
            _,
            _,
        ) = apb_context_vec_params

        ctx.H = num_heads
        ctx.head_dim = head_dim

        ctx.B = q.shape[0]
        ctx.N = q.shape[1]
        ctx.c_s = q.shape[-1]

        placements_single_expected = (Shard(0), Shard(1), Replicate())
        placements_pair_expected = (Shard(0), Shard(1), Shard(2))
        ctx.single_rep_placements = q.placements
        ctx.pair_rep_placements = z.placements
        ctx.device_mesh = q.device_mesh
        ctx.shape_q = q.shape
        ctx.stride_q = q.stride()
        ctx.shape_k = k.shape
        ctx.stride_k = k.stride()
        ctx.shape_v = v.shape
        ctx.stride_v = v.stride()
        ctx.shape_z = z.shape
        ctx.stride_z = z.stride()

        check_metadata = raise_if_incorrect_dtensor_metadata_args

        check_metadata(
            q,
            "q",
            check_for_partial_placements=True,
            expected_placements=placements_single_expected,  # type: ignore[arg-type]
        )
        check_metadata(
            k,
            "k",
            (ctx.B, ctx.N, ctx.c_s),
            expected_device_mesh=ctx.device_mesh,
            expected_placements=ctx.single_rep_placements,
        )
        check_metadata(
            v,
            "v",
            (ctx.B, ctx.N, ctx.c_s),
            expected_device_mesh=ctx.device_mesh,
            expected_placements=ctx.single_rep_placements,
        )
        check_metadata(
            z,
            "z",
            None,  # shape can be different from single representation(s) due to multiplicity
            expected_device_mesh=ctx.device_mesh,
            check_for_partial_placements=True,
            expected_placements=placements_pair_expected,  # type: ignore[arg-type]
        )
        check_metadata(
            mask,
            "mask",
            None,
            expected_device_mesh=ctx.device_mesh,
            expected_placements=ctx.single_rep_placements,
        )
        if pair_mask is not None:
            check_metadata(
                pair_mask,
                "pair_mask",
                None,  # shape can be different from single representation(s) due to multiplicity
                expected_device_mesh=ctx.device_mesh,
                expected_placements=ctx.pair_rep_placements,
            )

    @staticmethod
    def check_backward_input_metadata(ctx: FunctionCtx, grad_output: DTensor) -> None:
        raise_if_incorrect_dtensor_metadata_args(
            grad_output,
            "grad_output",
            expected_shape=(ctx.B, ctx.N, ctx.c_s),
            expected_device_mesh=ctx.device_mesh,
            expected_placements=ctx.single_rep_placements,
        )

    @staticmethod
    def check_backward_output_metadata(
        ctx: FunctionCtx,
        grad_q_dtensor: DTensor,
        grad_k_dtensor: DTensor,
        grad_v_dtensor: DTensor,
    ) -> None:
        """DTensor.from_local(..) requires the specification of stride if
        shape is specified, so the specification of stride is side-stepped
        in this usage by checking the shape determined by DTensor library."""
        metadata_tuple = (
            (grad_q_dtensor, "grad_q", (ctx.B, ctx.N, ctx.c_s)),
            (grad_k_dtensor, "grad_k", (ctx.B, ctx.N, ctx.c_s)),
            (grad_v_dtensor, "grad_v", (ctx.B, ctx.N, ctx.c_s)),
        )
        for dtensor_instance, dtensor_name, expected_shape in metadata_tuple:
            if not dtensor_instance.shape == expected_shape:
                raise ValueError(
                    ", ".join(
                        [
                            f"dtensor '{dtensor_name}' should have shape {expected_shape}",
                            f"but instead has shape {dtensor_instance.shape}.",
                        ]
                    )
                )

    @staticmethod
    def check_forward_input_for_impl_state(
        apb_context_vec_params: _AttentionPairBiasContextVecParams,
    ) -> None:
        """
        Check the implementation scope of the forward pass.

        Parameters
        ----------
        apb_context_vec_params: AttentionPairBiasContextVecParams
            The parameters for the attention pair bias context vector.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
        """
        (
            _,
            _,
            _,
            _,
            _,
            use_window_batching,
        ) = apb_context_vec_params

        if use_window_batching:
            raise NotImplementedError(
                f"use_window_batching={use_window_batching} is not implemented"
            )


class RingAttentionSimpleDataForBackward(NamedTuple):
    """Data for backward pass of ring attention."""

    q_store: Tensor
    k_t_store: Tensor
    v_t_store: Tensor
    z_store: Tensor
    lse_m: Tensor
    ring_comm: AttentionPairBiasComm
    o_store: Tensor | None = None
    multiplicity: int = 1


def ring_attention_simple_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    z: Tensor,
    pair_mask: Tensor,
    ring_comm: AttentionPairBiasComm,
    inf: float = 1e6,
) -> tuple[Tensor, RingAttentionSimpleDataForBackward]:
    """Forward pass of ring attention.

    example sharding strategy on N_tokens=2, world_size=4
    device mesh = [[  0,   1],
                   [  2,   3]]
    q/k/v, z, m = [[ q0,  q0],   [[ k0,  k0],   [[z00, z01],   [[m00, m01],
                   [ q1,  q1]]    [ k1,  k1]]     z10, z11]]     m10, m11]]

    step = 0
    k_t, z      = [[ k0,  k1],   [[z00, z01],
                   [ k0,  k1]]     z10, z11]]

    step = 1 (roll to the left)
    k_t, z      = [[ k1, k0],   [[z01, z00],
                   [ k1, k0]]     z11, z10]]

    now we roll left k, v, z as the ring attention outer loop.

    o           = [[ (q0k0+z00+m00)v0 + (q0k1+z01+m01)v1, (q0k1+z01+m01)v1 + (q0k0+z00+m00)v0 ],
                   [ (q1k0+z10+m10)v0 + (q1k1+z11+m11)v1, (q1k1+z11+m11)v1 + (q1k0+z10+m10)v0 ]]

    Parameters
    ----------
    q : Tensor
        Query tensor (B, I, H, D)
    k : Tensor
        Key tensor (B, J, H, D)
    v : Tensor
        Value tensor (B, J, H, D)
    z : Tensor
        Pair bias projection (B, H, I, J)
    pair_mask : Tensor
        Pair bias mask (B, 1, I, J)
    ring_comm : AttentionPairBiasComm
        Ring communication for async operation
    inf : float
        Infinity value for masking, by default 1e6

    Returns
    -------
    Tensor
        Output tensor (B, I, H, D)
    RingAttentionSimpleDataForBackward
        Data for backward pass
    """
    B_q, S, H, D = q.shape
    B_z_orig = z.shape[0]
    multiplicity = B_q // B_z_orig

    embed_dim = q.size(-1)

    # save input for backward
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad or z.requires_grad
    if q.is_cuda and not requires_grad and not HAS_TRITON:
        raise RuntimeError("Context-parallel ring attention requires Triton on Linux.")
    if requires_grad:
        q_store = q.detach()
    else:
        q_store = None

    # Overlap k, v comm with mask addition. The additive form (rather than
    # ``masked_fill_``) preserves broadcast semantics: when ``multiplicity > 1``,
    # ``pair_mask`` has the folded (B*M) batch axis while ``z`` has only B, and the
    # sum expands z along M. An in-place ``z.add_`` cannot broadcast the LHS, so the
    # out-of-place ``+`` is required whenever M > 1; we keep the out-of-place form
    # unconditionally so the ring APB stays correct across all M values.
    k_recv = ring_comm.comm_transpose_k.enqueue_to_dispatch(k.contiguous())
    v_recv = ring_comm.comm_transpose_v.enqueue_to_dispatch(v.contiguous())
    z = z.contiguous() + (1 - pair_mask) * -inf
    ring_comm.comm_transpose_k.wait_until_finished()
    ring_comm.comm_transpose_v.wait_until_finished()
    k = k_recv
    v = v_recv

    if requires_grad:
        k_t_store = k.detach()
        v_t_store = v.detach()
    else:
        k_t_store, v_t_store = None, None

    # save input for backward
    if requires_grad:
        z_store = z.detach()
    else:
        z_store = None

    # Ring attention
    o: Tensor | None = None
    lse_m: Tensor | None = None
    amax: Tensor | None = None

    use_triton = q.is_cuda and not requires_grad

    size_device_grid0 = ring_comm.group_layout.shape[0]
    for step in range(size_device_grid0):
        # Overlap k, v, z+m roll-left comm
        if step + 1 != size_device_grid0:
            next_k = ring_comm.comm_k.enqueue_to_dispatch(k)
            next_v = ring_comm.comm_v.enqueue_to_dispatch(v)
            next_z = ring_comm.comm_z.enqueue_to_dispatch(z)

        B_z = z.shape[0]
        B_q_local = q.shape[0]

        if use_triton:
            # Triton flash attention: never materializes the [B, H, I, J]
            # attention score tensor. The kernel does the M-broadcast internally
            # (via ``MULTIPLICITY`` constexpr), so we pass ``z`` at its real
            # batch axis without ``bias.expand().reshape().contiguous()``.
            multiplicity_kernel = B_q_local // B_z

            # [B, I, H, D] -> [B, H, I, D] for the kernel layout. The kernel
            # consumes arbitrary strides, so no `.contiguous()` is required.
            q_bh = q.transpose(1, 2)
            k_bh = k.transpose(1, 2)
            v_bh = v.transpose(1, 2)

            block_o_bh, block_lse_m_bh, block_amax_bh = _fused_flash_attn_with_lse(
                q_bh,
                k_bh,
                v_bh,
                bias=z,
                softmax_scale=embed_dim**-0.5,
                matmul_precision="ieee",
                multiplicity=multiplicity_kernel,
            )
            # Back to ring-loop layout [B, I, H, D].
            block_o = block_o_bh.transpose(1, 2)
            # block_lse_m_bh / block_amax_bh: [B, H, I] -> [B, I, H, 1]
            block_lse_m = block_lse_m_bh.transpose(-1, -2).unsqueeze(-1)
            block_amax = block_amax_bh.transpose(-1, -2).unsqueeze(-1)
        else:
            # CPU / requires-grad fallback: reference einsum attention.
            attn = torch.einsum("bihd,bjhd->bhij", q, k)
            attn = attn / (embed_dim**0.5)

            if B_q_local != B_z:
                attn = (attn.view(B_z, -1, *attn.shape[1:]) + z.unsqueeze(1)).view_as(attn)
            else:
                attn = attn + z

            block_o = torch.einsum("bhij,bjhd->bihd", torch.softmax(attn, dim=-1), v)
            block_amax = attn.amax(dim=-1, keepdim=True)
            block_lse_m = torch.logsumexp(attn - block_amax, dim=-1, keepdim=True)

            block_lse_m = block_lse_m.transpose(-2, -3)
            if block_amax is not None:
                block_amax = block_amax.transpose(-2, -3)

        o, lse_m, amax = tiled_softmax_attention_update(
            block_o, block_lse_m, block_amax, o, lse_m, amax
        )

        # Get input for next round
        if step + 1 != size_device_grid0:
            ring_comm.comm_k.wait_until_finished()
            ring_comm.comm_v.wait_until_finished()
            ring_comm.comm_z.wait_until_finished()
            k, v, z = next_k, next_v, next_z

    if requires_grad:
        assert amax is not None and lse_m is not None  # populated by the per-step update above
        if multiplicity > 1:
            # Reduce amax across multiplicity
            amax_t = amax.transpose(-2, -3)
            amax_t_view = amax_t.view(B_z_orig, multiplicity, *amax_t.shape[1:])
            amax_reduced = amax_t_view.amax(dim=1)
            z_store -= amax_reduced

            # Adjust lse_m to be relative to amax_reduced
            # We use broadcasting via views to avoid memory-heavy repeat_interleave
            amax_reduced_t = amax_reduced.transpose(-2, -3)  # (B, I, H, 1)
            lse_m_view = lse_m.view(B_z_orig, multiplicity, *lse_m.shape[1:])
            amax_view = amax.view(B_z_orig, multiplicity, *amax.shape[1:])

            lse_m_view += amax_view - amax_reduced_t.unsqueeze(1)  # modify lse_m inplace via view
        else:
            z_store -= amax.transpose(-2, -3)

    data_for_backward = RingAttentionSimpleDataForBackward(
        q_store=q_store,
        k_t_store=k_t_store,
        v_t_store=v_t_store,
        z_store=z_store,
        lse_m=lse_m,
        ring_comm=ring_comm,
        o_store=o.detach() if requires_grad and o is not None else None,
        multiplicity=multiplicity,
    )

    return o, data_for_backward


def ring_attention_simple_backward(
    data_for_backward: RingAttentionSimpleDataForBackward,
    do: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Backward pass of ring attention.

    example sharding strategy on N_tokens=2, world_size=4
    device mesh = [[  0,   1],
                   [  2,   3]]

    do          = [[ do0, do0],
                   [ do1, do1]]

    ctx should have saved:
    q    , z    = [[ q0,  q0],   [[z00, z01],
                   [ q1,  q1]]    [z10, z11]]
    k_t, v_t    = [[ k0,  k1],   [[v0,  v1],
                   [ k0,  k1]]    [v0,  v1]]
    lse         = [[ lse0, lse0],
                   [ lse1, lse1]]

    where z has already accounted for the pair mask

    The output gradient should be sharded as follows:
    dq          = [[ dq0, dq0],
                   [ dq1, dq1]]
    dk          = [[ dk0, dk0],
                   [ dk1, dk1]]
    dv          = [[ dv0, dv0],
                   [ dv1, dv1]]
    dz          = [[ dz00, dz01],
                   [ dz10, dz11]]

    Parameters
    ----------
    data_for_backward : RingAttentionSimpleDataForBackward
        Data for backward pass
    do : Tensor
        Gradient of output tensor

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Gradients for q, k, v, z
    """
    (
        q,
        k_t,
        v_t,
        z,
        lse_m,
        ring_comm,
        out_global,
        multiplicity,
    ) = data_for_backward

    embed_dim = q.size(-1)

    # Pre-calculate out_global * do for softmax gradient
    # do: (B, S, H, D)
    # out_global: (B, S, H, D)
    # do_o: (B, S, H, 1) -> will be reshaped to (B, H, S, 1) for the reference path
    do_o = torch.sum(do * out_global, dim=-1, keepdim=True)

    lse_m_t = lse_m.transpose(1, 2)  # (B, S, H, 1) -> (B, H, S, 1)

    # Compute S_ij and A_ij
    s = torch.einsum("bihd,bjhd->bhij", q, k_t)
    s /= embed_dim**0.5

    # Memory efficient in-place softmax reconstruction
    s.sub_(lse_m_t)
    if multiplicity > 1:
        B_z_orig = q.shape[0] // multiplicity
        if z.shape[0] == q.shape[0]:
            s.view(B_z_orig, multiplicity, *s.shape[1:]).add_(
                z.view(B_z_orig, multiplicity, *z.shape[1:])
            )
        else:
            s.view(B_z_orig, multiplicity, *s.shape[1:]).add_(z.unsqueeze(1))
    else:
        s.add_(z)

    a = s.exp_()

    # Compute gradient of v and c
    # dV_j = \Sum_{i} A^T_{ij} dO_i
    # c_i = \Sum_{k} v^T_k A_{ik}
    dv = torch.einsum("bihd,bhij->bjhd", do, a)

    # Compute gradient of S (dS = dz)
    # dS_{ij} = A_{ij} * (dO_i v^T_j - (dO_i * out_global_i))
    # do_o has shape (B, S, H, 1), we need (B, H, S, 1) to match a
    do_o_step = do_o.transpose(1, 2)

    # In-place compute dS to save memory
    tmp = torch.einsum("bihd,bjhd->bhij", do, v_t)
    tmp.sub_(do_o_step)
    a.mul_(tmp)
    del tmp

    dS = a
    dz = a
    if multiplicity > 1 and dz.shape[0] > (q.shape[0] // multiplicity):
        B_z_orig = q.shape[0] // multiplicity
        dz = dz.view(B_z_orig, multiplicity, *dz.shape[1:]).sum(dim=1)

    # Compute gradient of q, k
    # dq_i = \Sum_{j} dS_{ij} k^T_j
    # dk_j = \Sum_{i} dS^T_{ij} q_i
    dq = torch.einsum("bhij,bjhd->bihd", dS, k_t) / (
        embed_dim**0.5
    )  # _t here refers to transposition across device mesh
    dk = torch.einsum("bhij,bihd->bjhd", dS, q) / (embed_dim**0.5)

    dv_recv = ring_comm.comm_transpose_v.enqueue_to_dispatch(dv.contiguous())

    # collect and complete dv reduction
    ring_comm.comm_transpose_v.wait_until_finished()
    dv_work = dist.all_reduce(
        dv_recv, op=dist.ReduceOp.SUM, group=ring_comm.cp_axis_1_group, async_op=True
    )

    dq = dq.contiguous()
    dq_work = dist.all_reduce(
        dq, op=dist.ReduceOp.SUM, group=ring_comm.cp_axis_1_group, async_op=True
    )

    dk_recv = ring_comm.comm_transpose_k.enqueue_to_dispatch(dk.contiguous())
    ring_comm.comm_transpose_k.wait_until_finished()
    dk_work = dist.all_reduce(
        dk_recv, op=dist.ReduceOp.SUM, group=ring_comm.cp_axis_1_group, async_op=True
    )

    # Collect all async works
    dq_work.wait()
    dk_work.wait()
    dv_work.wait()

    return dq, dk_recv, dv_recv, dz
