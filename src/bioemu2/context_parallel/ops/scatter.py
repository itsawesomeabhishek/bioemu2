"""Context-parallel 1-D scatter-reduce with eps_for_mean.

No serial counterpart — distributed-only collective.
Ported from Fold-CP scatter.py; diverges by adding eps_for_mean to the mean denominator.
"""

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.tensor import Partial

from bioemu2.context_parallel.distributed.utils import update_exhaustive_strides
from bioemu2.context_parallel.ops.distributed_primitives.outer_gather import (
    get_overlap_from_peers,
)


class DistributedScatterReduce(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        output_size_per_rank: int,
        axis: int,
        idx_dtensor: DTensor,
        src_dtensor: DTensor,
        reduce: str,
        idx_mask: DTensor | None,
        are_ids_contiguous: bool,
        eps_for_mean: float = 0.0,
    ) -> DTensor:
        """Distributed scatter reduce.

        Scatters values from src into an output tensor at positions specified by idx,
        applying a reduction operation for duplicate indices. The output is initialized
        to zeros (not from a dst tensor).
        """
        if not are_ids_contiguous:
            raise NotImplementedError(
                "DistributedScatterReduce currently only supports are_ids_contiguous=True"
            )

        if reduce not in ("sum", "mean"):
            raise ValueError(
                f"reduce must be 'sum' or 'mean', got '{reduce}'. "
                "Other reductions are not supported due to complex backward requirements."
            )

        if not isinstance(idx_dtensor, DTensor) or not isinstance(src_dtensor, DTensor):
            raise TypeError("idx_dtensor and src_dtensor must be DTensors")

        batch_dims = idx_dtensor.shape[:axis]
        feature_shape = src_dtensor.shape[axis + 1 :]

        if idx_dtensor.ndim != axis + 1:
            raise ValueError(
                f"idx_dtensor should have ndim={axis + 1} for axis={axis}, got ndim={idx_dtensor.ndim}"
            )

        N_src = idx_dtensor.shape[axis]

        if src_dtensor.shape[:axis] != batch_dims:
            raise ValueError(
                f"Batch dimensions must match: idx {batch_dims} vs src {src_dtensor.shape[:axis]}"
            )
        if src_dtensor.shape[axis] != N_src:
            raise ValueError(
                f"src axis {axis} size {src_dtensor.shape[axis]} must match idx axis size {N_src}"
            )

        mesh = idx_dtensor.device_mesh
        if src_dtensor.device_mesh != mesh:
            raise ValueError("idx and src must be on the same DeviceMesh")
        if src_dtensor.placements != idx_dtensor.placements:
            raise ValueError("idx and src must have identical placements")

        if idx_mask is not None:
            if not isinstance(idx_mask, DTensor):
                raise TypeError("idx_mask must be a DTensor")
            if idx_mask.shape != idx_dtensor.shape:
                raise ValueError(
                    f"idx_mask shape {idx_mask.shape} must match idx_dtensor shape {idx_dtensor.shape}"
                )
            if idx_mask.device_mesh != idx_dtensor.device_mesh:
                raise ValueError("idx_mask must have the same device_mesh as idx_dtensor")
            if idx_mask.placements != idx_dtensor.placements:
                raise ValueError("idx_mask must have the same placements as idx_dtensor")
            if idx_mask.dtype != torch.bool:
                raise TypeError(
                    f"idx_mask must have dtype torch.bool, got {idx_mask.dtype}. "
                    "Use mask.bool() to convert."
                )

        placements = idx_dtensor.placements

        ndim_src = src_dtensor.ndim
        if axis < 0:
            axis += ndim_src
        if axis < 0 or axis >= ndim_src:
            raise ValueError(f"axis {axis} out of range for src.ndim={ndim_src}")

        mesh_dim_axis = None
        for i_mesh_dim, p in enumerate(placements):
            if isinstance(p, Partial):
                raise ValueError("Partial placements are not supported")
            if isinstance(p, Shard):
                if p.dim == axis:
                    mesh_dim_axis = i_mesh_dim
                if idx_dtensor.shape[p.dim] % mesh.size(i_mesh_dim) != 0:
                    raise ValueError(
                        f"idx_dtensor axis {p.dim} size {idx_dtensor.shape[p.dim]} "
                        f"not evenly divisible by mesh dim {i_mesh_dim}"
                    )
                if src_dtensor.shape[p.dim] % mesh.size(i_mesh_dim) != 0:
                    raise ValueError(
                        f"src_dtensor axis {p.dim} size {src_dtensor.shape[p.dim]} "
                        f"not evenly divisible by mesh dim {i_mesh_dim}"
                    )
                if idx_mask is not None and idx_mask.shape[p.dim] % mesh.size(i_mesh_dim) != 0:
                    raise ValueError(
                        f"idx_mask axis {p.dim} size {idx_mask.shape[p.dim]} "
                        f"not evenly divisible by mesh dim {i_mesh_dim}"
                    )

        if mesh_dim_axis is None:
            raise ValueError(f"Tensors must be sharded along axis {axis}")

        size_group = mesh.size(mesh_dim_axis)
        output_global_size = output_size_per_rank * size_group
        output_global_shape = batch_dims + (output_global_size,) + feature_shape

        idx_local = idx_dtensor.to_local()
        src_local = src_dtensor.to_local()
        idx_mask_local = idx_mask.to_local() if idx_mask is not None else None
        device = idx_local.device
        cpu_device = torch.device("cpu")

        batch_dims_local = src_local.shape[:axis]
        feature_shape_local = src_local.shape[axis + 1 :]
        output_local_shape = batch_dims_local + (output_size_per_rank,) + feature_shape_local

        # Write interval: where our local idx values want to write to.
        if idx_local.numel() > 0:
            if idx_mask_local is not None:
                if idx_mask_local.any():
                    valid_idx = idx_local[idx_mask_local]
                    write_interval = (
                        torch.stack(valid_idx.aminmax()).to(dtype=torch.long).unsqueeze(0)
                    )
                    write_interval[:, -1] += 1
                else:
                    write_interval = torch.tensor([[0, 0]], device=device, dtype=torch.long)
            else:
                write_interval = torch.stack(idx_local.aminmax()).to(dtype=torch.long).unsqueeze(0)
                write_interval[:, -1] += 1
        else:
            write_interval = torch.tensor([[0, 0]], device=device, dtype=torch.long)

        write_start = write_interval[0, 0]
        write_end = write_interval[0, 1]

        coord_axis = mesh.get_local_rank(mesh_dim_axis)
        own_start = torch.tensor(
            coord_axis * output_size_per_rank, device=cpu_device, dtype=torch.long
        )
        own_end = own_start + output_size_per_rank
        own_interval = torch.stack([own_start, own_end]).unsqueeze(0)

        group_axis = mesh.get_group(mesh_dim_axis)
        write_range = [torch.zeros_like(write_interval) for _ in range(size_group)]
        dist.all_gather(write_range, write_interval, group=group_axis)
        write_range = torch.stack(write_range)
        write_range_cpu = write_range.cpu()

        ranks_global_on_mesh = mesh.mesh
        my_coords = mesh.get_coordinate()
        index_list_submesh = []
        for dim in range(ranks_global_on_mesh.ndim):
            if dim == mesh_dim_axis:
                index_list_submesh.append(slice(None))
            else:
                index_list_submesh.append(torch.tensor(my_coords[dim], device=cpu_device))
        ranks_global_on_submesh = ranks_global_on_mesh[tuple(index_list_submesh)]

        my_rank = mesh.mesh[tuple(my_coords)].item()

        start_peers_own = (
            torch.arange(size_group, device=cpu_device, dtype=torch.long) * output_size_per_rank
        )
        end_peers_own = start_peers_own + output_size_per_rank
        interval_peers_own = torch.stack([start_peers_own, end_peers_own], dim=-1).unsqueeze(1)

        N_src_local = idx_local.shape[axis]
        shape_leading_flat = torch.Size(batch_dims_local).numel() if batch_dims_local else 1
        shape_trailing_flat = torch.Size(feature_shape_local).numel() if feature_shape_local else 1

        src_flat = src_local.reshape(shape_leading_flat, N_src_local, shape_trailing_flat)
        idx_flat = idx_local.reshape(shape_leading_flat, N_src_local)
        mask_flat = (
            idx_mask_local.reshape(shape_leading_flat, N_src_local)
            if idx_mask_local is not None
            else None
        )

        if write_start >= write_end:
            send_plan: list[dict] = []
        else:
            send_plan = get_overlap_from_peers(
                ranks_global_on_submesh,
                interval_peers_own,
                write_interval.to(cpu_device),
            )

        recv_plan = get_overlap_from_peers(
            ranks_global_on_submesh,
            write_range_cpu.view(size_group, 1, 2),
            own_interval.view(1, 1, 2),
        )

        batch_indices = (
            torch.arange(shape_leading_flat, device=device).unsqueeze(1).expand(-1, N_src_local)
        )

        send_counts = torch.zeros(size_group, device=device, dtype=torch.long)
        send_data_per_peer: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for item in send_plan:
            peer = item["peer"]
            interval = item["interval"]
            peer_start = interval[0, 0].item()
            peer_end = interval[0, 1].item()

            in_peer_interval = (idx_flat >= peer_start) & (idx_flat < peer_end)
            if mask_flat is not None:
                in_peer_interval = in_peer_interval & mask_flat

            count = in_peer_interval.sum().item()
            peer_coord = dist.get_group_rank(group_axis, peer)
            send_counts[peer_coord] = count

            if count > 0:
                peer_batch_idx = batch_indices[in_peer_interval]
                peer_owned_start = peer_coord * output_size_per_rank
                peer_idx = idx_flat[in_peer_interval] - peer_owned_start
                peer_src = src_flat[in_peer_interval]
                send_data_per_peer[peer] = (peer_batch_idx, peer_idx, peer_src)

        all_counts = [torch.zeros_like(send_counts) for _ in range(size_group)]
        dist.all_gather(all_counts, send_counts, group=group_axis)
        all_counts = torch.stack(all_counts)

        my_coord_in_group = coord_axis
        recv_counts = all_counts[:, my_coord_in_group]

        ops = []
        recv_bufs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        for item in recv_plan:
            peer = item["peer"]
            if peer == my_rank:
                if my_rank in send_data_per_peer:
                    recv_bufs[my_rank] = send_data_per_peer[my_rank]
                continue
            peer_coord = dist.get_group_rank(group_axis, peer)
            count = recv_counts[peer_coord].item()
            if count > 0:
                recv_batch_buf = torch.empty(count, device=device, dtype=torch.long)
                recv_idx_buf = torch.empty(count, device=device, dtype=idx_local.dtype)
                recv_src_buf = torch.empty(
                    count, shape_trailing_flat, device=device, dtype=src_local.dtype
                )
                ops.append(dist.P2POp(dist.irecv, recv_batch_buf, peer))
                ops.append(dist.P2POp(dist.irecv, recv_idx_buf, peer))
                ops.append(dist.P2POp(dist.irecv, recv_src_buf, peer))
                recv_bufs[peer] = (recv_batch_buf, recv_idx_buf, recv_src_buf)

        for item in send_plan:
            peer = item["peer"]
            if peer == my_rank:
                continue
            if peer in send_data_per_peer:
                peer_batch_idx, peer_idx, peer_src = send_data_per_peer[peer]
                ops.append(dist.P2POp(dist.isend, peer_batch_idx.contiguous(), peer))
                ops.append(dist.P2POp(dist.isend, peer_idx.contiguous(), peer))
                ops.append(dist.P2POp(dist.isend, peer_src.contiguous(), peer))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        out_local = torch.zeros(output_local_shape, dtype=src_local.dtype, device=device)
        out_linear = out_local.reshape(
            shape_leading_flat * output_size_per_rank, shape_trailing_flat
        )

        if reduce == "mean":
            count_linear = torch.zeros(
                shape_leading_flat * output_size_per_rank,
                1,
                dtype=out_local.dtype,
                device=device,
            )

        for peer, (recv_batch_idx, recv_idx, recv_src) in recv_bufs.items():
            recv_idx_long = recv_idx.to(torch.long)
            linear_idx = recv_batch_idx * output_size_per_rank + recv_idx_long
            linear_idx_expanded = linear_idx.unsqueeze(-1).expand(-1, shape_trailing_flat)
            out_linear.scatter_add_(0, linear_idx_expanded, recv_src)
            if reduce == "mean":
                ones = torch.ones(linear_idx.shape[0], 1, dtype=out_local.dtype, device=device)
                count_linear.scatter_add_(0, linear_idx.unsqueeze(-1), ones)

        count_local = None
        if reduce == "mean":
            if eps_for_mean > 0.0:
                # Bioemu-serial compat: serial computes
                # ``atom_to_token / (sum + 1e-6)`` so we mirror the biased
                # denominator here. Equivalent to vendor's
                # ``count.clamp(min=1)`` numerically everywhere count>=1 but
                # bit-identical to the serial reference up to fp rounding.
                denom = count_linear + eps_for_mean
            else:
                denom = count_linear.clamp(min=1)
            out_linear = out_linear / denom.expand(-1, shape_trailing_flat)
            count_local = count_linear.reshape(shape_leading_flat, output_size_per_rank, 1)
            count_local = count_local.expand(-1, -1, shape_trailing_flat).reshape(
                output_local_shape
            )

        out_local = out_linear.reshape(output_local_shape)

        strides_out = update_exhaustive_strides(
            out_local.shape, out_local.stride(), output_global_shape
        )
        out_dtensor = DTensor.from_local(
            out_local, mesh, placements, shape=output_global_shape, stride=strides_out
        )

        tensors_to_save = [idx_local, src_local]
        if idx_mask_local is not None:
            tensors_to_save.append(idx_mask_local)
        if count_local is not None:
            tensors_to_save.append(count_local)
        ctx.save_for_backward(*tensors_to_save)
        ctx.has_mask = idx_mask_local is not None
        ctx.has_count = count_local is not None
        ctx.reduce = reduce
        ctx.eps_for_mean = eps_for_mean
        ctx.comm_meta = {
            "axis": axis,
            "placements": placements,
            "output_global_shape": output_global_shape,
            "output_size_per_rank": output_size_per_rank,
            "src_global_shape": src_dtensor.shape,
            "output_global_stride": strides_out,
            "src_global_stride": src_dtensor.stride(),
            "own_interval": own_interval,
            "device_mesh": mesh,
            "mesh_dim_axis": mesh_dim_axis,
        }

        return out_dtensor

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        saved = ctx.saved_tensors
        idx_local = saved[0]
        src_local = saved[1]
        tensor_idx = 2

        idx_mask_local = None
        if ctx.has_mask:
            idx_mask_local = saved[tensor_idx]
            tensor_idx += 1

        count_local = None
        if ctx.has_count:
            count_local = saved[tensor_idx]

        reduce = ctx.reduce
        meta = ctx.comm_meta
        axis = meta["axis"]
        placements = meta["placements"]
        output_size_per_rank = meta["output_size_per_rank"]
        src_global_shape = meta["src_global_shape"]
        src_global_stride = meta["src_global_stride"]
        own_interval = meta["own_interval"]
        mesh = meta["device_mesh"]
        mesh_dim_axis = meta["mesh_dim_axis"]

        grad_local = grad_output.to_local().contiguous()
        device = grad_local.device
        cpu_device = torch.device("cpu")

        if reduce == "mean" and count_local is not None:
            eps = getattr(ctx, "eps_for_mean", 0.0)
            if eps > 0.0:
                count_clamped = count_local + eps
            else:
                count_clamped = count_local.clamp(min=1)
            grad_gather_source = grad_local / count_clamped
        else:
            grad_gather_source = grad_local

        if idx_local.numel() > 0:
            if idx_mask_local is not None:
                if idx_mask_local.any():
                    valid_idx = idx_local[idx_mask_local]
                    need_interval = (
                        torch.stack(valid_idx.aminmax()).to(dtype=torch.long).unsqueeze(0)
                    )
                    need_interval[:, -1] += 1
                else:
                    need_interval = torch.tensor([[0, 0]], device=device, dtype=torch.long)
            else:
                need_interval = torch.stack(idx_local.aminmax()).to(dtype=torch.long).unsqueeze(0)
                need_interval[:, -1] += 1
        else:
            need_interval = torch.tensor([[0, 0]], device=device, dtype=torch.long)

        need_start = need_interval[0, 0]
        need_end = need_interval[0, 1]
        need_start_cpu = need_start.to(cpu_device)

        own_start = own_interval[0, 0]

        group_axis = mesh.get_group(mesh_dim_axis)
        size_group = mesh.size(mesh_dim_axis)
        need_range = [torch.zeros_like(need_interval) for _ in range(size_group)]
        dist.all_gather(need_range, need_interval, group=group_axis)
        need_range = torch.stack(need_range)
        need_range_cpu = need_range.cpu()

        ranks_global_on_mesh = mesh.mesh
        my_coords = mesh.get_coordinate()
        index_list_submesh = []
        for dim in range(ranks_global_on_mesh.ndim):
            if dim == mesh_dim_axis:
                index_list_submesh.append(slice(None))
            else:
                index_list_submesh.append(torch.tensor(my_coords[dim], device=cpu_device))
        ranks_global_on_submesh = ranks_global_on_mesh[tuple(index_list_submesh)]

        my_rank = mesh.mesh[tuple(my_coords)].item()

        if need_start >= need_end:
            needed_chunks: list[dict] = []
        else:
            start_peers_own = (
                torch.arange(size_group, device=cpu_device, dtype=torch.long) * output_size_per_rank
            )
            end_peers_own = start_peers_own + output_size_per_rank
            interval_peers_own = torch.stack([start_peers_own, end_peers_own], dim=-1).unsqueeze(1)

            needed_chunks = get_overlap_from_peers(
                ranks_global_on_submesh,
                interval_peers_own,
                need_interval.to(cpu_device),
            )

        ops = []
        recv_bufs = {}

        for item in needed_chunks:
            peer = item["peer"]
            interval = item["interval"]
            start_global = interval[0, 0]
            length = interval[0, 1] - interval[0, 0]

            shape = list(grad_gather_source.shape)
            shape[axis] = length.item()
            buf = torch.empty(shape, dtype=grad_gather_source.dtype, device=device)

            if peer == my_rank:
                start_local = start_global - own_start
                buf.copy_(grad_gather_source.narrow(axis, start_local.item(), length.item()))
                recv_bufs[peer] = buf
            else:
                ops.append(dist.P2POp(dist.irecv, buf, peer))
                recv_bufs[peer] = buf

        send_chunks = get_overlap_from_peers(
            ranks_global_on_submesh,
            need_range_cpu.view(size_group, 1, 2),
            own_interval.view(1, 1, 2),
        )

        for item in send_chunks:
            peer = item["peer"]
            if peer == my_rank:
                continue
            interval = item["interval"]
            start_global = interval[0, 0]
            length = interval[0, 1] - interval[0, 0]
            start_local = start_global - own_start
            chunk = grad_gather_source.narrow(axis, start_local.item(), length.item()).contiguous()
            ops.append(dist.P2POp(dist.isend, chunk, peer))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        if need_start >= need_end:
            buffer_shape = list(grad_gather_source.shape)
            buffer_shape[axis] = 0
            grad_buffer = torch.empty(buffer_shape, dtype=grad_gather_source.dtype, device=device)
        else:
            buffer_shape = list(grad_gather_source.shape)
            buffer_shape[axis] = (need_end - need_start).item()
            grad_buffer = torch.zeros(buffer_shape, dtype=grad_gather_source.dtype, device=device)

            for item in needed_chunks:
                peer = item["peer"]
                interval = item["interval"]
                buf = recv_bufs.get(peer)
                if buf is None:
                    raise RuntimeError(f"Missing recv buffer for peer {peer}")
                start_global = interval[0, 0]
                length = interval[0, 1] - interval[0, 0]
                start_local_buf = start_global - need_start_cpu
                target = grad_buffer.narrow(axis, start_local_buf.item(), length.item())
                target.copy_(buf)

        idx_local_buffer = idx_local - need_start.item()

        shape_trailing = grad_buffer.shape[axis + 1 :]
        shape_trailing_flat = torch.Size(shape_trailing).numel() if shape_trailing else 1

        shape_leading = grad_buffer.shape[:axis]
        shape_leading_flat = torch.Size(shape_leading).numel()
        L = grad_buffer.shape[axis]
        N_src_local = idx_local.shape[axis]

        if L == 0:
            grad_src_local = torch.zeros_like(src_local)
        else:
            if idx_mask_local is not None:
                idx_local_buffer = torch.where(
                    idx_mask_local, idx_local_buffer, torch.zeros_like(idx_local_buffer)
                )

            grad_flat = grad_buffer.reshape(shape_leading_flat, L, shape_trailing_flat)
            idx_flat = idx_local_buffer.reshape(shape_leading_flat, N_src_local)

            batch_idx = torch.arange(shape_leading_flat, device=device).reshape(
                shape_leading_flat, 1
            )
            grad_src_flat = grad_flat[batch_idx, idx_flat, :]

            if idx_mask_local is not None:
                mask_flat = idx_mask_local.reshape(shape_leading_flat, N_src_local, 1).to(
                    grad_src_flat.dtype
                )
                grad_src_flat = grad_src_flat * mask_flat

            if shape_trailing:
                grad_src_local = grad_src_flat.reshape(*shape_leading, N_src_local, *shape_trailing)
            else:
                grad_src_local = grad_src_flat.reshape(*shape_leading, N_src_local)

        grad_src_dtensor = DTensor.from_local(
            grad_src_local.contiguous(),
            grad_output.device_mesh,
            placements,
            shape=src_global_shape,
            stride=src_global_stride,
        )

        return None, None, None, grad_src_dtensor, None, None, None, None


def distributed_scatter_reduce(
    output_size_per_rank: int,
    axis: int,
    idx: DTensor,
    src: DTensor,
    reduce: str,
    idx_mask: DTensor | None = None,
    are_ids_contiguous: bool = False,
    eps_for_mean: float = 0.0,
) -> DTensor:
    """Distributed scatter-reduce. See :class:`DistributedScatterReduce`.

    ``eps_for_mean`` (default 0.0): when ``reduce == "mean"`` and
    ``eps_for_mean > 0``, divide by ``(count + eps_for_mean)`` instead of
    ``count.clamp(min=1)``. Bioemu's serial atom→token aggregation uses
    the biased denominator ``count + 1e-6``; passing ``eps_for_mean=1e-6``
    here preserves bit-identity with the serial reference at CP=1.
    """
    return DistributedScatterReduce.apply(
        output_size_per_rank,
        axis,
        idx,
        src,
        reduce,
        idx_mask,
        are_ids_contiguous,
        eps_for_mean,
    )


__all__ = ["DistributedScatterReduce", "distributed_scatter_reduce"]
