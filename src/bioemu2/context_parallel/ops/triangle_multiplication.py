"""Context-parallel triangle multiplication with an in-place residual update for BioMoa."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.ops.triangle_multiplication import (
    TriangleMultiplicationCueq,
    TriangleMultiplicationCueqTiled,
    TriangleMultiplicationFull,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.ops.distributed_primitives.elementwise_op import (
    ElementwiseOp,
    elementwise_op,
)
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import (
    LayerNormParamsReplicated,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.sigmoid_gate import sigmoid_gate
from bioemu2.context_parallel.ops.triangular_mult_impl import (
    _DEFAULT_BMM_CHUNK,
    _DEFAULT_TILE_SIZE,
    _Direction,
    _distributed_bmm,
    _TriangleMultiplicationImpl,
    _XposeArgs,
    cueq_fused_prologue,
)

BioMoaTriangleMultiplication = (
    TriangleMultiplicationFull
    | TriangleMultiplicationCueq
    | TriangleMultiplicationCueqTiled
    | TriangleMultiplicationIncoming
    | TriangleMultiplicationOutgoing
)


class TriangleMultiplicationAddRes(nn.Module):
    def __init__(
        self,
        direction: _Direction,
        layer: nn.Module,  # any TriMul-shaped serial layer: needs .norm_in/.p_in/.g_in/.norm_out/.p_out/.g_out
        device_mesh: DeviceMesh,
        comm: Ring2DComm,
        bmm_chunk: int = _DEFAULT_BMM_CHUNK,
        tile_size: int = _DEFAULT_TILE_SIZE,
    ) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.ring_comm = comm
        self._bmm_chunk = bmm_chunk
        self._tile_size = tile_size

        self.norm_in = LayerNormParamsReplicated(layer.norm_in, self.device_mesh)
        self.p_in = LinearParamsReplicated(layer.p_in, self.device_mesh)
        self.g_in = LinearParamsReplicated(layer.g_in, self.device_mesh)

        self.norm_out = LayerNormParamsReplicated(layer.norm_out, self.device_mesh)
        self.p_out = LinearParamsReplicated(layer.p_out, self.device_mesh)
        self.g_out = LinearParamsReplicated(layer.g_out, self.device_mesh)

        self._direction = direction

    def forward(
        self,
        x: DTensor,  # DTensor, Shard(0), Shard(1), Shard(2); (B, N, N, D)
        mask: DTensor,  # DTensor, Shard(0), Shard(1), Shard(2); (B, N, N)
    ) -> DTensor:  # x, updated in place; Shard(0), Shard(1), Shard(2); (B, N, N, D)
        """Add the triangle-multiplication update to ``x`` in place and return ``x``."""
        x_local_in = x.to_local()  # plain; (B, Ni, Nj, D)

        if x_local_in.is_cuda and not torch.is_grad_enabled():
            mask_local = mask.to_local()  # plain; (B, Ni, Nj)

            norm_in_weight = self.norm_in.weight.to_local()
            norm_in_bias = self.norm_in.bias.to_local()
            p_in_weight = self.p_in.weight.to_local()
            g_in_weight = self.g_in.weight.to_local()
            g_out_weight = self.g_out.weight.to_local()
            g_out_bias = self.g_out.bias.to_local() if self.g_out.bias is not None else None

            B, Ni, Nj, D = x_local_in.shape

            # Same direction-dependent permutations as
            # _TriangleMultiplicationImpl.forward.
            if self._direction == _Direction.Outgoing:
                permute_lhs = (0, 3, 1, 2)
                permute_rhs = (0, 3, 2, 1)
                permute_out = (0, 2, 3, 1)
                xpose_args = _XposeArgs.rhs
            elif self._direction == _Direction.Incoming:
                permute_lhs = (0, 3, 2, 1)
                permute_rhs = (0, 3, 1, 2)
                permute_out = (0, 2, 3, 1)
                xpose_args = _XposeArgs.lhs
            else:
                raise ValueError(f"Invalid direction: {self._direction}")

            operand_dtype = (
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled("cuda")
                else x_local_in.dtype
            )

            def permuted_buffer(permute: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
                shape = (B, Ni, Nj, D)
                buffer = torch.empty(
                    [shape[k] for k in permute], device=x_local_in.device, dtype=operand_dtype
                )
                inverse = sorted(range(4), key=lambda k: permute[k])
                return buffer, buffer.permute(inverse)  # view in (B, Ni, Nj, D) order

            lhs, a_view = permuted_buffer(permute_lhs)
            rhs, b_view = permuted_buffer(permute_rhs)
            tile = self._tile_size if self._tile_size > 0 else Ni
            for i in range(0, Ni, tile):
                fused = cueq_fused_prologue(
                    x_local_in[:, i : i + tile],
                    mask_local[:, i : i + tile],
                    norm_in_weight,
                    norm_in_bias,
                    p_in_weight,
                    g_in_weight,
                    eps=self.norm_in.eps,
                )  # [2D, B, tile, Nj]
                a_raw, b_raw = fused.chunk(2, dim=0)
                a_view[:, i : i + tile] = a_raw.permute(1, 2, 3, 0)
                b_view[:, i : i + tile] = b_raw.permute(1, 2, 3, 0)
                del fused, a_raw, b_raw
            del a_view, b_view

            out = _distributed_bmm(
                lhs,
                rhs,
                self.ring_comm,
                permute_out=permute_out,
                xpose_args=xpose_args,
                bmm_chunk=self._bmm_chunk,
                copy_inputs=False,
                # The ring sums one partial product per step; float32 rounds that sum
                # once, like the serial kernel, while operands and ``x`` stay bfloat16.
                out_dtype=torch.float32,
            )  # (B, Ni, Nj, D) view of a (B, D, Ni, Nj) buffer
            del lhs, rhs

            p_out_bias = self.p_out.bias.to_local() if self.p_out.bias is not None else None
            for i in range(0, Ni, tile):
                x_tile = x_local_in[:, i : i + tile]
                gate = F.linear(
                    F.layer_norm(x_tile, [D], norm_in_weight, norm_in_bias, self.norm_in.eps),
                    g_out_weight,
                    g_out_bias,
                )
                update = F.linear(
                    F.layer_norm(
                        out[:, i : i + tile],
                        [D],
                        self.norm_out.weight.to_local(),
                        self.norm_out.bias.to_local(),
                        self.norm_out.eps,
                    ),
                    self.p_out.weight.to_local(),
                    p_out_bias,
                )
                x_tile += update * gate.to(x_local_in.dtype).sigmoid()
            return x

        # DTensor path for CPU and gradient-enabled calls.
        h = self.norm_in(x)
        g_out = self.g_out(h)
        update = _TriangleMultiplicationImpl.apply(
            self.p_in(h), mask, self.g_in(h), self.ring_comm, self._direction
        )
        update = sigmoid_gate(self.p_out(self.norm_out(update)), g_out)
        return elementwise_op(x, update, ElementwiseOp.SUM, in_place=True)


__all__ = [
    "BioMoaTriangleMultiplication",
    "TriangleMultiplicationAddRes",
    "_DEFAULT_BMM_CHUNK",
    "_DEFAULT_TILE_SIZE",
    "_Direction",
]
