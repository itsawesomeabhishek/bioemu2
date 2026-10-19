# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import ctypes
import importlib.util
import math
import os
import tempfile
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

import bioemu2.context_parallel.ops.outer_product_mean as cp_opm
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.utils.session import cp_session

CPTestFn = Callable[[int, torch.device, DeviceMesh, Ring2DComm, DistributedManager], None]


def _opm_ring_step_accum_cpu(
    a: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    proj_o_weight: torch.Tensor,
    tile_size: int,
    matmul_precision: str,
) -> None:
    # Replace only the CUDA kernel, retaining the real ring and normalization.
    outer = torch.einsum("bsic,bsjd->bijcd", a, b).flatten(-2)
    output.add_(torch.nn.functional.linear(outer, proj_o_weight))


def randomize_params(module: torch.nn.Module) -> None:
    """Fill zero-initialised params with random values"""
    with torch.no_grad():
        for p in module.parameters():
            if p.abs().max() == 0:
                p.copy_(torch.randn_like(p) * 0.02)


def gather_n_assert_parity_at_rank0(
    serial_out: torch.Tensor,
    cp_out: torch.Tensor | DTensor,
    *,
    label: str,
    atol: float,
    mask: torch.Tensor | None = None,
) -> float:
    """Compare the serial reference with the full CP output on rank 0.
    ``mask`` (if given) restricts the comparison to non-padded positions.
    All ranks must call this helper: ``full_tensor()`` and the verdict broadcast are collectives.
    """
    if isinstance(cp_out, DTensor):
        cp_out = cp_out.full_tensor()
    # Shape errors must raise on every rank; raising on rank 0 alone strands the
    # other ranks in their next collective.
    mask_shape = () if mask is None else mask.shape
    torch.broadcast_shapes(serial_out.shape, cp_out.shape, mask_shape)

    max_diff = torch.zeros((), dtype=torch.float32, device=cp_out.device)
    if dist.get_rank() == 0:
        diff = serial_out - cp_out
        if mask is not None:
            diff = diff * mask
        max_diff.copy_(diff.abs().max())
        print(f"\n[{label}] CP vs serial max-abs-diff = {max_diff.item():.3e}")
    dist.broadcast(max_diff, src=0)
    assert max_diff.item() <= atol, (
        f"[{label}] parity violated: max-abs-diff={max_diff.item()} (atol={atol})"
    )
    return max_diff.item()


def assert_output_placement(
    cp_out: DTensor,
    *,
    expected_placements: tuple,
    label: str,
) -> None:
    """Run on all ranks.  Assert that a DTensor output has the expected sharding placements."""
    actual = cp_out.placements
    assert actual == expected_placements, (
        f"[{label}] placement mismatch: expected {expected_placements}, got {actual}"
    )

    # Size-one mesh axes cannot reduce the local tensor shape.
    local_shape = cp_out.to_local().shape
    global_shape = cp_out.shape
    mesh = cp_out.device_mesh
    for mesh_dim, p in enumerate(actual):
        if isinstance(p, Shard) and mesh.size(mesh_dim) > 1:
            assert local_shape[p.dim] < global_shape[p.dim], (
                f"[{label}] local shape {local_shape} == global shape {global_shape} on "
                f"tensor dim {p.dim} (mesh dim {mesh_dim}) — shard did not split the tensor"
            )


def assert_replicate_values_against_rank0(
    cp_out: DTensor,
    *,
    atol: float = 1.5e-7,
    rtol: float = 0.0,
) -> None:
    """Assert every replica of ``cp_out`` matches the replica at coordinate 0 on its replicate axes.

    Collective: all ranks of ``cp_out.device_mesh`` must call this helper. The verdict
    is reduced over the whole mesh, so either every rank raises or none does.
    """
    mesh = cp_out.device_mesh
    replicate_dims = [
        mesh_dim
        for mesh_dim, placement in enumerate(cp_out.placements)
        if isinstance(placement, Replicate) and mesh.size(mesh_dim) > 1
    ]
    if not replicate_dims:
        return

    # NCCL requires a contiguous buffer; to_local() may return a strided view.
    local_value = cp_out.to_local().detach().contiguous()
    # Chaining broadcasts over the replicate axes delivers one shared reference:
    # the value held at coordinate 0 on every replicate axis.
    reference = local_value.clone()
    for mesh_dim in replicate_dims:
        group = mesh.get_group(mesh_dim)
        dist.broadcast(reference, src=dist.get_global_rank(group, 0), group=group)
    matches = torch.isclose(local_value, reference, atol=atol, rtol=rtol).all().to(torch.int32)

    for mesh_dim in range(mesh.ndim):
        if mesh.size(mesh_dim) > 1:
            dist.all_reduce(matches, op=dist.ReduceOp.MIN, group=mesh.get_group(mesh_dim))
    assert matches.item() == 1, (
        f"[assert_replicate_values_against_rank0] replica values differ (atol={atol}, rtol={rtol})"
    )


def spawn_cp_test(
    test_fn: CPTestFn,
    *,
    master_port: str = "29511",
) -> None:
    """Spawn a CP parity test on all visible GPUs (largest perfect-square mesh).

    test_fn signature: (rank: int, device, mesh, ring_comm, dm) -> None.
    Call from a test marked ``requires_4gpu``.
    """
    n_gpu = torch.cuda.device_count()
    side = math.isqrt(n_gpu)

    assert side * side == n_gpu, f"Number of GPUs ({n_gpu}) must be a perfect square."
    world_size = side * side
    grid: OrderedDict[str, int | tuple[int, ...]] = OrderedDict([("dp", 1), ("cp", (side, side))])

    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.spawn(
        fn=_spawn_worker,
        args=(grid, world_size, master_port, test_fn),
        nprocs=world_size,
        join=True,
    )


def _spawn_worker(
    rank: int,
    grid: OrderedDict[str, int | tuple[int, ...]],
    world_size: int,
    master_port: str,
    test_fn: CPTestFn,
) -> None:
    """Run the parity callback inside a deterministic CP session."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    _run_in_cp_session(grid, "cuda", test_fn)


def _run_in_cp_session(
    grid: OrderedDict[str, int | tuple[int, ...]],
    device_type: Literal["cpu", "cuda"],
    test_fn: CPTestFn,
) -> None:
    """Open a CP session on ``device_type`` and call ``test_fn`` with its mesh and ring.

    On CPU, the CUDA-only OPM ring kernel is replaced with an einsum.
    """
    cpu_opm = (
        patch.object(cp_opm, "_opm_ring_step_accum", _opm_ring_step_accum_cpu)
        if device_type == "cpu"
        else nullcontext()
    )
    with cpu_opm, cp_session(grid, device_type=device_type) as (device_manager, device, rank):
        mesh: DeviceMesh = device_manager.device_mesh_subgroups
        ring_comm = Ring2DComm(
            device_manager.group["cp"],
            device_manager.subgroups["cp"][0],
            device_manager.layout_subgroups["cp"],
        )
        test_fn(rank, device, mesh, ring_comm, device_manager)
    assert not dist.is_initialized()
    assert not DistributedManager._state


def spawn_cpu_cp_test(
    test_fn: CPTestFn,
    *,
    dp_size: int = 1,
    cp_size: int = 2,
    local_world_size: int | None = None,
    master_port: str = "29512",
) -> None:
    """Spawn a ``cp_size`` x ``cp_size`` CPU/Gloo CP mesh per DP rank.

    test_fn has the same signature as for :func:`spawn_cp_test`. Set ``local_world_size``
    to simulate several nodes.
    """
    assert dp_size > 0, "DP size must be positive."
    assert cp_size > 0, "CP size must be positive."
    world_size = cp_size**2 * dp_size
    local_world_size = world_size if local_world_size is None else local_world_size
    assert local_world_size > 0 and world_size % local_world_size == 0, (
        "Local world size must be positive and divide the world size."
    )
    grid: OrderedDict[str, int | tuple[int, ...]] = OrderedDict(
        [("dp", dp_size), ("cp", (cp_size, cp_size))]
    )
    with _cpu_cuequivariance_env():
        torch.multiprocessing.spawn(
            fn=_spawn_cpu_worker,
            args=(grid, world_size, local_world_size, master_port, test_fn),
            nprocs=world_size,
            join=True,
        )


@contextmanager
def _cpu_cuequivariance_env() -> Iterator[None]:
    """Let spawned CPU ranks load cuEquivariance kernels, with or without a GPU driver.

    CPU-only installations do not need CUDA library setup. When cuEquivariance is installed,
    importing bioemu2 loads ``libcue_ops.so``, which links against ``libcuda.so.1``. Hosts
    without a driver get the CUDA toolkit stub instead. The stub answers every driver query
    with an error, so it is used only when no real driver loads. Spawned ranks inherit
    the environment set here because the dynamic loader reads it at process start.
    """
    cue_spec = importlib.util.find_spec("cuequivariance_ops")
    if cue_spec is None:
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}):
            yield
        return
    assert cue_spec.origin is not None
    cue_lib = next(Path(cue_spec.origin).parent.rglob("libcue_ops.so"))
    with tempfile.TemporaryDirectory() as stub_dir:
        library_dirs = [str(cue_lib.parent)]
        try:
            ctypes.CDLL("libcuda.so.1")
        except OSError:
            cuda_stubs = list(
                Path("/usr/local").glob("cuda*/targets/x86_64-linux/lib/stubs/libcuda.so")
            )
            if not cuda_stubs:
                pytest.skip("A CUDA driver or driver stub is required for CPU CP tests.")
            (Path(stub_dir) / "libcuda.so.1").symlink_to(cuda_stubs[0])
            library_dirs.insert(0, stub_dir)
        with patch.dict(
            os.environ,
            {
                "OMP_NUM_THREADS": "1",
                "LD_LIBRARY_PATH": os.pathsep.join(
                    [*library_dirs, os.environ.get("LD_LIBRARY_PATH", "")]
                ),
            },
        ):
            yield


def _spawn_cpu_worker(
    rank: int,
    grid: OrderedDict[str, int | tuple[int, ...]],
    world_size: int,
    local_world_size: int,
    master_port: str,
    test_fn: CPTestFn,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.set_num_threads(1)
    os.environ["LOCAL_RANK"] = str(rank % local_world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    os.environ["GROUP_RANK"] = str(rank // local_world_size)
    # Hide GPUs and answer torch.cuda.is_available() from CUDA_VISIBLE_DEVICES alone, so
    # DeviceMesh setup never initializes a CUDA driver, real or stub, in a CPU rank.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
    _run_in_cp_session(grid, "cpu", test_fn)
