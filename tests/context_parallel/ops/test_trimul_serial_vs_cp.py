from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.triangle_multiplication import (
    TriangleMultiplicationIncoming as SerialTriMulIncoming,
)
from bioemu2.biomoa.ops.triangle_multiplication import (
    TriangleMultiplicationOutgoing as SerialTriMulOutgoing,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import (
    TriangleMultiplicationAddRes,
    _Direction,
)
from bioemu2.context_parallel.ops.triangular_mult_impl import _matmul_accumulate
from bioemu2.context_parallel.utils.dtensor import (
    PL_PAIR,
    shard_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

# N must be divisible by the 2x2 mesh's shard size.
B, N, D = 2, 64, 128

# Cannon's partial matmuls change the accumulation order relative to serial einsum.
FWD_ATOL = 2e-6


def _build_serial_layer(direction: str, device: torch.device) -> torch.nn.Module:
    """Build a serial TriMul layer with deterministic non-zero weights."""
    torch.manual_seed(42)
    if direction == "outgoing":
        layer = SerialTriMulOutgoing(dim=D)
    else:
        layer = SerialTriMulIncoming(dim=D)
    layer = layer.to(device=device, dtype=torch.float32).eval()
    randomize_params(layer)
    return layer


def _trimul_e2e_parity_worker(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager | None,
    direction: str,
    dtype: torch.dtype = torch.float32,
    bmm_chunk: int = 256,
) -> None:
    """Per-rank worker: compare serial forward vs CP forward."""

    serial_layer = _build_serial_layer(direction, device)

    cp_direction = _Direction.Outgoing if direction == "outgoing" else _Direction.Incoming
    dist_layer = TriangleMultiplicationAddRes(
        direction=cp_direction,
        layer=serial_layer,
        device_mesh=mesh,
        comm=ring_comm,
        bmm_chunk=bmm_chunk,
        tile_size=16,
    )

    torch.manual_seed(123)
    x_full = torch.randn(B, N, N, D, device=device, dtype=torch.float32)
    mask_full = torch.randint(0, 2, (B, N, N), device=device, dtype=torch.bool)

    with torch.no_grad():
        out_serial_fp32 = serial_layer(x_full, mask_full)
        out_serial_fp32 = x_full + out_serial_fp32

    assert out_serial_fp32.abs().max().item() > 0, "serial output is trivially zero"

    x_full_casted = x_full.to(dtype=dtype)
    [x_dt, mask_dt] = shard_pair_along_token([x_full_casted, mask_full], mesh)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype == torch.bfloat16):
        cp_casted = dist_layer(x_dt, mask_dt)

    assert cp_casted is x_dt
    assert_output_placement(cp_casted, expected_placements=PL_PAIR, label=f"TriMul/{direction}")
    atol = 6e-2 if dtype == torch.bfloat16 else FWD_ATOL
    gather_n_assert_parity_at_rank0(
        out_serial_fp32, cp_casted.float(), label=f"TriMul/{direction}", atol=atol
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("direction", ("outgoing", "incoming"))
@pytest.mark.parametrize(
    ("dtype", "bmm_chunk"),
    [(torch.float32, 256), (torch.float32, 8), (torch.bfloat16, 8)],
    ids=["32-256", "32-8", "bf16-8"],
)
def test_trimul_serial_vs_cp(direction: str, dtype: torch.dtype, bmm_chunk: int) -> None:
    """x plus the serial TriMul update matches the in-place CP residual update."""
    spawn_cp_test(
        partial(
            _trimul_e2e_parity_worker,
            direction=direction,
            dtype=dtype,
            bmm_chunk=bmm_chunk,
        )
    )


@pytest.mark.gpu
def test_matmul_accumulate_sums_bf16_ring_steps_in_float32(gpu_device: torch.device) -> None:
    """Four ring steps of bf16 products sum with float32, not bf16, rounding."""
    torch.manual_seed(0)
    lhs = torch.randn(4, 2, 3, 40, 64, device=gpu_device, dtype=torch.bfloat16)
    rhs = torch.randn(4, 2, 3, 64, 24, device=gpu_device, dtype=torch.bfloat16)
    expected = torch.matmul(lhs.double(), rhs.double()).sum(dim=0)
    out = torch.zeros(2, 3, 40, 24, device=gpu_device)
    for step in range(4):
        # Row slices of a contiguous accumulator, as _distributed_bmm chunks it.
        for start in range(0, 40, 16):
            end = min(start + 16, 40)
            _matmul_accumulate(out[..., start:end, :], lhs[step, ..., start:end, :], rhs[step])
    error = (out.double() - expected).abs().max() / expected.abs().max()
    assert error.item() < 1e-6
