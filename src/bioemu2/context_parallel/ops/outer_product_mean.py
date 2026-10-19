"""Canonical context-parallel outer-product mean for BioMoa."""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2 import HAS_TRITON
from bioemu2.biomoa.ops.outer_product_mean import (
    OuterProductMeanTritonMemorySaving,
    OuterProductMeanVanilla,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.utils import LayoutRightMap
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import (
    LayerNormParamsReplicated,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated

if HAS_TRITON:
    import triton

    from bioemu2.context_parallel.ops.triton.opm import _opm_ring_accum_kernel

# ``tl.dot`` needs every operand dimension to be at least 16: the MSA contraction uses
# ``_OPM_BLOCK_I x _OPM_BLOCK_S`` and the projection uses ``_OPM_BLOCK_I * _OPM_BLOCK_J`` rows.
_OPM_BLOCK_I = 32
_OPM_BLOCK_J = 2
_OPM_BLOCK_S = 32
_OPM_BLOCK_C = 2


def _opm_ring_step_accum(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    proj_o_weight: torch.Tensor,
    tile_size: int,
    matmul_precision: str,
) -> None:
    """Accumulate one OPM ring step into the output tensor.

    Each kernel program owns a ``_OPM_BLOCK_I x _OPM_BLOCK_J`` output tile and keeps its
    outer products in registers, so the step allocates nothing beyond padded inputs.
    """
    if a.ndim == 3:
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

    batch_size, msa_depth, n_tokens, hidden_dim = a.shape
    output_dim = proj_o_weight.shape[0]
    hidden_padded = triton.next_power_of_2(max(hidden_dim, 16))
    output_padded = triton.next_power_of_2(max(output_dim, 16))

    if hidden_padded != hidden_dim:
        pad = hidden_padded - hidden_dim
        a = torch.nn.functional.pad(a, (0, pad))
        b = torch.nn.functional.pad(b, (0, pad))

    weight = proj_o_weight.t().contiguous()
    if hidden_padded != hidden_dim or output_padded != output_dim:
        padded_weight = torch.zeros(
            hidden_padded * hidden_padded,
            output_padded,
            device=weight.device,
            dtype=weight.dtype,
        )
        for hidden_idx in range(hidden_dim):
            src_start = hidden_idx * hidden_dim
            dst_start = hidden_idx * hidden_padded
            padded_weight[dst_start : dst_start + hidden_dim, :output_dim] = weight[
                src_start : src_start + hidden_dim
            ]
        weight = padded_weight

    for batch_idx in range(batch_size):
        b_all = b[batch_idx].contiguous()
        for start in range(0, n_tokens, tile_size):
            end = min(start + tile_size, n_tokens)
            tile_i = end - start
            # Channel-major copy so each program's per-channel ``a`` loads are contiguous.
            a_tile = a[batch_idx, :, start:end].permute(2, 0, 1).contiguous()
            output_tile = output[batch_idx, start:end]
            grid = (triton.cdiv(tile_i, _OPM_BLOCK_I), triton.cdiv(n_tokens, _OPM_BLOCK_J))
            _opm_ring_accum_kernel[grid](
                a_tile,
                b_all,
                weight,
                output_tile,
                tile_i,
                n_tokens,
                msa_depth,
                hidden_padded,
                output_dim,
                output_padded,
                a_tile.stride(1),
                a_tile.stride(2),
                a_tile.stride(0),
                b_all.stride(0),
                b_all.stride(1),
                b_all.stride(2),
                weight.stride(0),
                weight.stride(1),
                output_tile.stride(0),
                output_tile.stride(1),
                output_tile.stride(2),
                block_i=_OPM_BLOCK_I,
                block_j=_OPM_BLOCK_J,
                block_s=_OPM_BLOCK_S,
                block_c=_OPM_BLOCK_C,
                num_warps=4,
                num_stages=3,
                input_precision=matmul_precision,
            )


class _BioMoaOPMRingImpl(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        a: DTensor,
        b: DTensor,
        mask: DTensor,
        row_count: torch.Tensor,
        ring_comm: Ring2DComm,
        proj_o_weight_local: torch.Tensor,
        proj_o_bias_local: torch.Tensor | None,
        tile_size: int,
        matmul_precision: str,
    ) -> DTensor:
        del ctx
        mask_local = mask.to_local().unsqueeze(-1)
        a_local = a.to_local() * mask_local
        b_local = b.to_local() * mask_local

        batch, _msa_local, n_tokens, c_hidden = a_local.shape
        output_dim = proj_o_weight_local.shape[0]

        a_recv = ring_comm.comm_transpose_row_init.enqueue_to_dispatch(a_local)
        b_recv = ring_comm.comm_col_init.enqueue_to_dispatch(b_local)
        # Each ring step adds a partial sum over one MSA shard; float32 keeps those partial
        # sums and the mean unrounded until the single cast back to the operand dtype.
        output = torch.zeros(
            (batch, n_tokens, n_tokens, output_dim),
            dtype=torch.float32,
            device=a_local.device,
        )
        ring_comm.comm_col_init.wait_until_finished()
        ring_comm.comm_transpose_row_init.wait_until_finished()

        a_buffer = [a_recv, a_local]
        b_buffer = [b_recv, b_local]
        ready = 0
        recv = 1
        n_steps = ring_comm.group_layout.shape[1]
        for step in range(n_steps):
            a_ready = a_buffer[ready]
            b_ready = b_buffer[ready]
            if step < n_steps - 1:
                a_buffer[recv] = ring_comm.comm_row.enqueue_to_dispatch(a_ready, a_buffer[recv])
                b_buffer[recv] = ring_comm.comm_col.enqueue_to_dispatch(b_ready, b_buffer[recv])

            _opm_ring_step_accum(
                a_ready,
                b_ready,
                output,
                proj_o_weight_local,
                tile_size,
                matmul_precision,
            )

            if step < n_steps - 1:
                ring_comm.comm_row.wait_until_finished()
                ring_comm.comm_col.wait_until_finished()
                ready ^= 1
                recv ^= 1

        output.div_(row_count[:, None, None, None].to(output.dtype))

        if proj_o_bias_local is not None:
            output.add_(proj_o_bias_local.to(output.dtype))
        output = output.to(a_local.dtype)
        shape = (a.shape[0], a.shape[2], a.shape[2], output_dim)
        return DTensor.from_local(
            output,
            device_mesh=a.device_mesh,
            placements=a.placements,
            shape=shape,
            stride=LayoutRightMap(shape).strides,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):  # pragma: no cover
        del ctx, grad_output
        raise NotImplementedError("Distributed BioMoa OPM is inference-only")


class DistributedOuterProductMean(torch.nn.Module):
    def __init__(
        self,
        serial: OuterProductMeanVanilla | OuterProductMeanTritonMemorySaving,
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
    ) -> None:
        super().__init__()
        self.ring_comm = comm
        self.tile_size = serial.tile_size
        self.matmul_precision = getattr(serial, "_matmul_precision", "ieee")
        self.norm = LayerNormParamsReplicated(serial.norm, device_mesh)
        self.proj_a = LinearParamsReplicated(
            self._projection_slice(serial.proj_ab, 0, serial.c_hidden),
            device_mesh,
        )
        self.proj_b = LinearParamsReplicated(
            self._projection_slice(serial.proj_ab, serial.c_hidden, 2 * serial.c_hidden),
            device_mesh,
        )
        self.proj_o = LinearParamsReplicated(serial.proj_o, device_mesh)

    @staticmethod
    def _projection_slice(
        projection: torch.nn.Linear,
        start: int,
        end: int,
    ) -> torch.nn.Linear:
        sliced = torch.nn.Linear(projection.in_features, end - start, bias=False)
        sliced.weight = torch.nn.Parameter(projection.weight[start:end])
        return sliced

    def forward(
        self,
        m: DTensor,
        mask: DTensor,
        row_count: torch.Tensor,  # Plain local DP batch; (B / DP,).
    ) -> DTensor:
        if not HAS_TRITON:
            raise RuntimeError("Context-parallel outer-product mean requires Triton on Linux.")
        assert not self.training, "DistributedOuterProductMean is inference-only"
        input_dtype = m.to_local().dtype
        m = self.norm(m)
        if m.to_local().dtype != input_dtype:
            m = DTensor.from_local(
                m.to_local().to(input_dtype),
                device_mesh=m.device_mesh,
                placements=m.placements,
                shape=m.shape,
                stride=m.stride(),
            )
        a = self.proj_a(m)
        b = self.proj_b(m)
        return _BioMoaOPMRingImpl.apply(
            a,
            b,
            mask,
            row_count,
            self.ring_comm,
            self.proj_o.weight.to_local(),
            self.proj_o.bias.to_local() if self.proj_o.bias is not None else None,
            self.tile_size,
            self.matmul_precision,
        )


__all__ = ["DistributedOuterProductMean"]
