"""CP session lifecycle (init_cp, cp_session, teardown) and CUDA memory helpers."""

import datetime
import gc
import logging
import math
import os
from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
from typing import Literal

import torch
import torch.distributed as dist

from bioemu2.context_parallel.distributed.manager import DistributedManager

logger = logging.getLogger(__name__)

# Logger whose level EMU_CP_VERBOSITY sets; only this subtree is touched.
_CP_LOGGER_NAME = "bioemu2.context_parallel"

# TRACE sits just below DEBUG so `trace` can enable extra-heavy diagnostics
# (all-rank memory logging + forward hooks + NCCL INFO) that `debug` leaves off.
TRACE_LEVEL = logging.DEBUG - 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

# EMU_CP_VERBOSITY -> log level for the CP logger tree. Every CP diagnostic keys
# off this one level via logger.isEnabledFor(...); no custom subsystem, no per-diagnostic env var.
_CP_VERBOSITY_LEVELS = {
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": TRACE_LEVEL,
}


def _apply_cp_verbosity() -> None:
    """Set the CP-logger level from EMU_CP_VERBOSITY (default info).

    Only the ``bioemu2.context_parallel`` logger is touched, so the rest of the emu
    logging tree is unaffected. NCCL_DEBUG / TORCH_DISTRIBUTED_DEBUG are intentionally NOT
    set here: libnccl / c10d read them when torchrun bootstraps the process (before this
    Python code runs), so they must be exported by cp_env.sh in the shell, not from init_cp.
    """
    name = os.environ.get("EMU_CP_VERBOSITY", "info").strip().lower()
    try:
        level = _CP_VERBOSITY_LEVELS[name]
    except KeyError:
        raise ValueError(
            f"EMU_CP_VERBOSITY={name!r} must be one of {sorted(_CP_VERBOSITY_LEVELS)}."
        ) from None
    logging.getLogger(_CP_LOGGER_NAME).setLevel(level)


# Session lifecycle


def _grid_from_env() -> OrderedDict[str, int | tuple[int, ...]]:
    """Derive CP grid from WORLD_SIZE (must be perfect square) or EMU_CP_GRID='H,W'."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    grid_env = os.environ.get("EMU_CP_GRID")
    if grid_env:
        parts = [int(x) for x in grid_env.split(",")]
        if len(parts) != 2 or any(p <= 0 for p in parts):
            raise ValueError(f"EMU_CP_GRID must be 'H,W' with positive ints, got {grid_env!r}")
        cp_h, cp_w = parts
    else:
        root = math.isqrt(world_size)
        if root * root != world_size:
            raise ValueError(
                f"WORLD_SIZE={world_size} is not a perfect square; set EMU_CP_GRID='H,W' "
                "to specify the CP mesh explicitly."
            )
        cp_h = cp_w = root
    if cp_h * cp_w != world_size:
        raise ValueError(
            f"CP mesh {cp_h}x{cp_w}={cp_h * cp_w} does not match WORLD_SIZE={world_size}."
        )
    return OrderedDict([("dp", 1), ("cp", (cp_h, cp_w))])


def init_cp(
    grid: OrderedDict[str, int | tuple[int, ...]] | None = None,
    *,
    device_type: Literal["cpu", "cuda"] = "cuda",
) -> tuple[DistributedManager, torch.device, int]:
    """Initialize a square CP mesh using CUDA/NCCL (default) or CPU/Gloo.

    Grid axis insertion order matters — DistributedManager builds DeviceMesh
    axes in OrderedDict order (dp first, then cp as 2D sub-mesh).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 4:
        raise ValueError(
            "CP sessions require at least four processes in total. Launch with "
            "uv run torchrun --standalone --nproc-per-node=4 --module bioemu2.cp ..."
        )
    _apply_cp_verbosity()

    if grid is None:
        grid = _grid_from_env()

    # Ring2DComm requires a square 2D layout.
    cp = grid["cp"]
    if not (isinstance(cp, tuple) and len(cp) == 2 and cp[0] == cp[1] and cp[0] > 0):
        raise ValueError(f"CP mesh must be a square (H, H) tuple; got grid['cp']={cp!r}.")
    if device_type == "cuda" and (grid.get("dp") != 1 or cp[0] * cp[1] != world_size):
        raise ValueError("CP sampling requires DP=1 and a mesh matching WORLD_SIZE.")

    if device_type == "cuda":
        torch.cuda.set_device(local_rank)

    DistributedManager.initialize(
        grid, device_type=device_type, backend="nccl" if device_type == "cuda" else "gloo"
    )
    _create_cpu_group(world_size)

    # Structure-module work can take 5+ min per denoising step, easily
    # exceeding the 10-min default.
    if dist.is_initialized():
        dist.distributed_c10d._set_pg_timeout(
            datetime.timedelta(minutes=30), dist.distributed_c10d._get_default_group()
        )

    dm = DistributedManager()
    device = torch.device(f"cuda:{local_rank}" if device_type == "cuda" else "cpu")
    rank = dist.get_rank() if dist.is_initialized() else 0

    logger.info("CP initialised: rank=%d mesh=%dx%d device=%s", rank, cp[0], cp[1], device)
    log_cp_memory("init", device)
    return dm, device, rank


def _create_cpu_group(world_size: int) -> None:
    """Register a Gloo group spanning all ranks as ``DistributedManager().group["cpu"]``.

    Object collectives pickle into a byte tensor on the group's device. Over NCCL that
    tensor lands on every rank's GPU, so host-side objects such as prepared structures
    (full MSA, dense contact maps) are broadcast over Gloo instead. CPU sessions reuse
    the default Gloo group.
    """
    if DistributedManager().backend == "gloo":
        state = DistributedManager._state
        for key in ("_group", "_group_rank", "_group_ranks"):
            state[key]["cpu"] = state[key]["world"]
    else:
        DistributedManager.create_group("cpu", ranks=list(range(world_size)), backend="gloo")


def teardown_cp_state() -> None:
    """Destroy process groups without an extra barrier and always clear manager state.

    An extra barrier can hang if another rank never reaches cleanup.
    """
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    finally:
        DistributedManager._state = {}


@contextmanager
def cp_session(
    grid: OrderedDict[str, int | tuple[int, ...]] | None = None,
    *,
    device_type: Literal["cpu", "cuda"] = "cuda",
) -> Generator[tuple[DistributedManager, torch.device, int], None, None]:
    """Initialize CP on entry and tear it down on exit, including failed initialization.

    For one node, launch with ``uv run torchrun --standalone --nproc-per-node=4``.
    Multi-node launches are allowed but untested.
    The square mesh defaults to ``sqrt(WORLD_SIZE)`` or comes from ``EMU_CP_GRID``.
    Returns the existing ``(manager, device, rank)`` tuple; nested sessions are rejected.
    Pass ``device_type="cpu"`` for CPU/Gloo; CUDA/NCCL is the default.
    Explicit CPU grids may include DP groups; CUDA sampling requires DP=1.
    CPU sessions initialize process groups but do not enable CUDA sampling kernels on CPU.
    """
    if dist.is_initialized():
        raise RuntimeError("A process group is already active; CP sessions cannot be nested.")
    try:
        dm, device, rank = init_cp(grid, device_type=device_type)
        yield dm, device, rank
    finally:
        teardown_cp_state()


# Utilities


def barrier() -> None:
    """dist.barrier() if process group is live, else no-op."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cuda_gc() -> None:
    """Python GC + torch.cuda.empty_cache()."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_cp_memory(tag: str, device: torch.device) -> None:
    """Log CUDA alloc/reserved/peak for the CP logger at DEBUG (no-op otherwise or off-CUDA)."""
    if device.type != "cuda" or not logger.isEnabledFor(logging.DEBUG):
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    gib = 1024**3
    logger.debug(
        "cp_mem[%s] device=%s alloc=%.2fGiB reserved=%.2fGiB peak=%.2fGiB",
        tag,
        device,
        torch.cuda.memory_allocated(idx) / gib,
        torch.cuda.memory_reserved(idx) / gib,
        torch.cuda.max_memory_allocated(idx) / gib,
    )
