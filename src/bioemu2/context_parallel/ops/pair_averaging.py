# Canonical BioMoa pair-averaging communication, vendored from NVIDIA Fold-CP.
# Source: src/boltz/distributed/model/layers/pair_averaging.py
# BioMoa-isolated copy of the communication and autograd implementation.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from copy import deepcopy

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from bioemu2.context_parallel.distributed.comm import (
    One2OneComm,
    TransposeComm,
    ternary_parity,
)
from bioemu2.context_parallel.distributed.utils import (
    LayoutMap,
    get_group_rank_from_axial_shift,
    tiled_softmax_attention_update,
)


class Ring2DCommPairAveraging:
    """Communication primitives required by pair-weighted averaging."""

    def __init__(
        self,
        group_2d: dist.ProcessGroup,
        group_col: dist.ProcessGroup,
        group_layout: LayoutMap,
    ) -> None:
        self.group_2d = group_2d
        self.group_col = group_col
        self.group_layout = group_layout
        ranks_group_2d = set(dist.get_process_group_ranks(self.group_2d))
        ranks_group_col = set(dist.get_process_group_ranks(self.group_col))
        if not ranks_group_col.issubset(ranks_group_2d):
            raise ValueError("The col ranks are not a subset of ranks_group_2d")

        self.size_2d = dist.get_world_size(self.group_2d)
        if self.size_2d != self.group_layout.numel:
            raise ValueError(
                f"size of group_2d {self.size_2d} differs from the number of "
                f"elements in group_layout {self.group_layout.numel}"
            )
        if self.group_layout.shape[0] != self.group_layout.shape[1]:
            raise ValueError(f"group_layout.shape {self.group_layout.shape} is not square")

        self.rank_2d = dist.get_rank(self.group_2d)
        self.coord_2d = self.group_layout.unravel(self.rank_2d)
        self.comm_2d_trans = TransposeComm(self.group_2d, self.group_layout)

        self.send_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -self.coord_2d[0], self.group_layout
        )
        self.recv_rank_row_init = get_group_rank_from_axial_shift(
            self.coord_2d, 1, self.coord_2d[0], self.group_layout
        )
        self.comm_row_init = One2OneComm(
            self.group_2d,
            self.send_rank_row_init,
            self.recv_rank_row_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_row_init, self.recv_rank_row_init),
        )
        self.send_rank_row = get_group_rank_from_axial_shift(
            self.coord_2d, 1, -1, self.group_layout
        )
        self.recv_rank_row = get_group_rank_from_axial_shift(self.coord_2d, 1, 1, self.group_layout)
        self.comm_row = One2OneComm(
            self.group_2d,
            self.send_rank_row,
            self.recv_rank_row,
            parity=ternary_parity(self.rank_2d, self.send_rank_row, self.recv_rank_row),
        )

        self.send_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -self.coord_2d[1], self.group_layout
        )
        self.recv_rank_col_init = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )
        self.comm_col_init = One2OneComm(
            self.group_2d,
            self.send_rank_col_init,
            self.recv_rank_col_init,
            parity=ternary_parity(self.rank_2d, self.send_rank_col_init, self.recv_rank_col_init),
        )
        self.send_rank_col = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -1, self.group_layout
        )
        self.recv_rank_col = get_group_rank_from_axial_shift(self.coord_2d, 0, 1, self.group_layout)
        self.comm_col = One2OneComm(
            self.group_2d,
            self.send_rank_col,
            self.recv_rank_col,
            parity=ternary_parity(self.rank_2d, self.send_rank_col, self.recv_rank_col),
        )

        self.comm_d_init = deepcopy(self.comm_row_init)
        self.comm_d = deepcopy(self.comm_row)
        self.comm_db = deepcopy(self.comm_col)

        self.send_rank_db_final = get_group_rank_from_axial_shift(
            self.coord_2d, 0, self.coord_2d[1], self.group_layout
        )
        self.recv_rank_db_final = get_group_rank_from_axial_shift(
            self.coord_2d, 0, -self.coord_2d[1], self.group_layout
        )
        self.comm_db_final = One2OneComm(
            self.group_2d,
            self.send_rank_db_final,
            self.recv_rank_db_final,
            parity=ternary_parity(self.rank_2d, self.send_rank_db_final, self.recv_rank_db_final),
        )
        self.comm_2d_trans_lse_m = deepcopy(self.comm_2d_trans)
        self.comm_2d_trans_amax = deepcopy(self.comm_2d_trans)


class _PairWeightedAveragingImpl(torch.autograd.Function):
    """Distributed pair-weighted averaging using 2D ring communication."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        v: DTensor,
        b: DTensor,
        mask: DTensor,
        g: DTensor,
        comm: Ring2DCommPairAveraging,
        n_heads: int,
        inf: float,
    ) -> DTensor:
        if not isinstance(comm, Ring2DCommPairAveraging):
            raise ValueError(
                f"Input comm must be of type Ring2DCommPairAveraging. Got type {type(comm)}."
            )
        if not all(isinstance(tensor, DTensor) for tensor in (v, b, mask, g)):
            raise TypeError(
                f"Inputs 'v', 'b', 'mask' and 'g' must be of type DTensor. Got types "
                f"{type(v)}, {type(b)}, {type(mask)}, and {type(g)}."
            )
        if any(tensor.device_mesh != v.device_mesh for tensor in (b, mask, g)):
            raise ValueError("Inputs must have identical device meshes.")
        if any(tensor.placements != v.placements for tensor in (b, mask, g)):
            raise ValueError("Inputs must have identical placements.")
        if v.placements != (Shard(0), Shard(1), Shard(2)):
            raise ValueError(
                f"Input placements must be (Shard(0), Shard(1), Shard(2)). Got {v.placements}."
            )
        if (v.ndim, b.ndim, mask.ndim, g.ndim) != (4, 4, 3, 4):
            raise ValueError("Inputs v, b, mask, and g must have dimensions 4, 4, 3, and 4.")

        v_local = v.to_local()
        b_local = b.to_local()
        mask_local = mask.to_local()
        g_local = g.to_local()
        batch, msa, tokens, heads_channels = v_local.shape
        if heads_channels % n_heads:
            raise ValueError(f"Input tensor 'v' must have a last dimension divisible by {n_heads}.")
        c_h = heads_channels // n_heads
        if b_local.shape != (batch, tokens, tokens, n_heads):
            raise ValueError("Input tensor 'b' has an incompatible shape.")
        if mask_local.shape != (batch, tokens, tokens):
            raise ValueError("Input tensor 'mask' has an incompatible shape.")
        if g_local.shape != v_local.shape:
            raise ValueError("Input tensor 'g' has an incompatible shape.")

        tensor_to_mesh_axis = [-1] * v.ndim
        for mesh_axis, placement in enumerate(v.placements):
            if isinstance(placement, Shard):
                tensor_to_mesh_axis[placement.dim] = mesh_axis
        if tensor_to_mesh_axis[1] == -1 or tensor_to_mesh_axis[2] == -1:
            raise ValueError("Input dimensions 1 and 2 must be sharded.")
        if comm.group_col != v.device_mesh.get_group(tensor_to_mesh_axis[1]):
            raise ValueError("Input comm group_col does not match the input device mesh.")
        coordinate = v.device_mesh.get_coordinate()
        if coordinate is None:
            raise ValueError("The current rank is not on the input device mesh.")
        if comm.coord_2d != (
            coordinate[tensor_to_mesh_axis[1]],
            coordinate[tensor_to_mesh_axis[2]],
        ):
            raise ValueError("Input comm coordinates do not match the input device mesh.")

        requires_grad = v.requires_grad or b.requires_grad
        ctx.mark_non_differentiable(mask)
        ctx.device_mesh_input = v.device_mesh
        ctx.placements_input = v.placements
        ctx.input_shape = v.shape
        ctx.input_stride = v.stride()
        ctx.g_shape = g.shape
        ctx.g_stride = g.stride()
        ctx.b_shape = b.shape
        ctx.b_stride = b.stride()
        ctx.comm = comm
        ctx.n_heads = n_heads
        ctx.c_h = c_h

        v_local = v_local.unflatten(-1, (n_heads, c_h)).permute(0, 3, 1, 2, 4).contiguous()
        v_saved = v_local.detach().clone() if requires_grad else None
        v_recv = comm.comm_row_init.enqueue_to_dispatch(v_local)

        mask_bias = (1 - mask_local) * -inf
        b_local = b_local + mask_bias.unsqueeze(-1)
        b_transposed = b_local.permute(0, 3, 2, 1).contiguous()
        b_recv = comm.comm_2d_trans.enqueue_to_dispatch(b_transposed)
        g_local = g_local.sigmoid()
        comm.comm_2d_trans.wait_until_finished()
        b_transposed = comm.comm_col_init.enqueue_to_dispatch(b_recv, b_transposed)
        comm.comm_col_init.wait_until_finished()
        comm.comm_row_init.wait_until_finished()

        b_buffers = [b_transposed, b_recv]
        v_buffers = [v_recv, v_local]
        ready, recv = 0, 1
        output = lse = amax = None
        for step in range(comm.group_layout.shape[0]):
            another_step = step + 1 < comm.group_layout.shape[0]
            if another_step:
                v_buffers[recv] = comm.comm_row.enqueue_to_dispatch(
                    v_buffers[ready], v_buffers[recv]
                )
                b_buffers[recv] = comm.comm_col.enqueue_to_dispatch(
                    b_buffers[ready], b_buffers[recv]
                )
            block_amax = b_buffers[ready].amax(dim=-2, keepdim=True)
            block_lse = torch.logsumexp(b_buffers[ready] - block_amax, dim=-2, keepdim=True)
            probabilities = b_buffers[ready].softmax(dim=-2)
            block_output = torch.einsum("bhsjd,bhji->bhsid", v_buffers[ready], probabilities)
            output, lse, amax = tiled_softmax_attention_update(
                block_output.transpose(-3, -2).flatten(start_dim=-2),
                block_lse.transpose(-2, -1),
                block_amax.transpose(-2, -1),
                output,
                lse,
                amax,
            )
            if another_step:
                comm.comm_row.wait_until_finished()
                comm.comm_col.wait_until_finished()
                ready ^= 1
                recv ^= 1

        assert output is not None and lse is not None and amax is not None
        output = output.unflatten(-1, (msa, c_h)).permute(0, 3, 2, 1, 4).flatten(-2)
        output_saved = output.detach().clone(memory_format=torch.contiguous_format)
        if requires_grad:
            amax_recv = comm.comm_2d_trans_amax.enqueue_to_dispatch(amax)
            b_local = b_local.permute(0, 3, 1, 2).contiguous()
            lse_recv = comm.comm_2d_trans_lse_m.enqueue_to_dispatch(lse)
            comm.comm_2d_trans_amax.wait_until_finished()
            b_local -= amax_recv
            comm.comm_2d_trans_lse_m.wait_until_finished()
            b_local -= lse_recv
            b_local.exp_()
            ctx.save_for_backward(v_saved, b_local, g_local, output_saved)
        output *= g_local
        return DTensor.from_local(
            output,
            device_mesh=v.device_mesh,
            placements=v.placements,
            shape=ctx.input_shape,
            stride=ctx.input_stride,
        )

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(
        ctx,
        do: DTensor,
    ) -> tuple[DTensor, DTensor, None, DTensor, None, None, None]:
        if not isinstance(do, DTensor):
            raise TypeError(f"Input 'do' must be of type DTensor. Got type {type(do)}.")
        if do.device_mesh != ctx.device_mesh_input:
            raise ValueError("Input 'do' must have the same device mesh as the inputs.")
        if do.placements != ctx.placements_input:
            raise ValueError("Input 'do' must have the same placements as the inputs.")

        v_local, p_local, g_local, output_local = ctx.saved_tensors
        comm = ctx.comm
        msa = do.to_local().shape[1]
        p_recv = comm.comm_col_init.enqueue_to_dispatch(p_local)
        do_local = do.to_local().to(g_local.dtype) * g_local
        dg_local = (1 - g_local) * do_local * output_local
        dg = DTensor.from_local(
            dg_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.g_shape,
            stride=ctx.g_stride,
        )
        do_local = (
            do_local.unflatten(-1, (ctx.n_heads, ctx.c_h))
            .permute(0, 3, 1, 4, 2)
            .flatten(-3, -2)
            .contiguous()
        )
        do_recv = comm.comm_row_init.enqueue_to_dispatch(do_local)
        output_local = (
            output_local.unflatten(-1, (ctx.n_heads, ctx.c_h))
            .permute(0, 3, 1, 4, 2)
            .flatten(-3, -2)
            .contiguous()
        )
        d_local = torch.einsum("bhti,bhti->bhi", do_local, output_local).contiguous()
        d_recv = comm.comm_d_init.enqueue_to_dispatch(d_local)
        v_local = v_local.transpose(-2, -1).flatten(-3, -2).contiguous()
        comm.comm_col_init.wait_until_finished()
        comm.comm_row_init.wait_until_finished()
        comm.comm_d_init.wait_until_finished()

        ready, recv = 0, 1
        p_buffers = [p_recv, p_local]
        do_buffers = [do_recv, do_local]
        d_buffers = [d_recv, d_local]
        db_buffers = [torch.zeros_like(p_local), torch.zeros_like(p_local)]
        dv_local = torch.zeros_like(output_local)
        for step in range(comm.group_layout.shape[0]):
            another_step = step < comm.group_layout.shape[0] - 1
            if another_step:
                p_buffers[recv] = comm.comm_col.enqueue_to_dispatch(
                    p_buffers[ready], p_buffers[recv]
                )
                do_buffers[recv] = comm.comm_row.enqueue_to_dispatch(
                    do_buffers[ready], do_buffers[recv]
                )
                d_buffers[recv] = comm.comm_d.enqueue_to_dispatch(d_buffers[ready], d_buffers[recv])
            dv_local += torch.einsum("bhti,bhij->bhtj", do_buffers[ready], p_buffers[ready])
            db_block = torch.einsum("bhti,bhtj->bhij", do_buffers[ready], v_local).contiguous()
            db_block -= d_buffers[ready].unsqueeze(-1)
            db_block *= p_buffers[ready]
            if step:
                comm.comm_db.wait_until_finished()
            db_buffers[ready] += db_block
            db_buffers[recv] = comm.comm_db.enqueue_to_dispatch(
                db_buffers[ready].to(db_buffers[recv].dtype), db_buffers[recv]
            )
            if another_step:
                comm.comm_col.wait_until_finished()
                comm.comm_row.wait_until_finished()
                comm.comm_d.wait_until_finished()
                ready ^= 1
                recv ^= 1

        dv_local = dv_local.unflatten(-2, (msa, ctx.c_h)).permute(0, 2, 4, 1, 3).flatten(-2)
        dv = DTensor.from_local(
            dv_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.input_shape,
            stride=ctx.input_stride,
        )
        comm.comm_db.wait_until_finished()
        ready ^= 1
        recv ^= 1
        db_buffers[recv] = comm.comm_db_final.enqueue_to_dispatch(
            db_buffers[ready], db_buffers[recv]
        )
        comm.comm_db_final.wait_until_finished()
        ready ^= 1
        db_local = db_buffers[ready].permute(0, 2, 3, 1)
        db = DTensor.from_local(
            db_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.b_shape,
            stride=ctx.b_stride,
        )
        return dv, db, None, dg, None, None, None


__all__ = ["Ring2DCommPairAveraging"]
