from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.triangle_multiplication import (
    TriangleMultiplicationOutgoing as SerialTrimul,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import triangle_multiplication as tm
from bioemu2.context_parallel.utils.dtensor import (
    PL_PAIR,
    shard_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    randomize_params,
    spawn_cp_test,
)


def _trimul_parity_worker(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    direction: str,
    dtype: torch.dtype,
) -> None:
    """Per-rank worker: compare tiled vs single-tile TriMul residual updates."""
    torch.manual_seed(42)
    # cuEq tiles require D >= 64.
    B, N, D = 2, 512, 128

    # Weights stay float32; with bf16-mixed, pairs are bfloat16 under bfloat16 autocast.
    serial_layer = SerialTrimul(dim=D).to(device=device)
    randomize_params(serial_layer)

    cp_direction = tm._Direction.Outgoing if direction == "outgoing" else tm._Direction.Incoming
    dist_layer = tm.TriangleMultiplicationAddRes(
        direction=cp_direction,
        layer=serial_layer,
        device_mesh=mesh,
        comm=ring_comm,
    )

    x_full = torch.randn(B, N, N, D, device=device, dtype=dtype)
    mask_full = (torch.rand(B, N, N, device=device) > 0.2).to(dtype)
    [x_dt, mask_dt] = shard_pair_along_token([x_full, mask_full], mesh)

    local_rows = x_dt.to_local().shape[1]
    tile_sizes = [64, 96, 128]
    assert max(tile_sizes) < local_rows

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        dist_layer._tile_size = 0  # one tile spans the local shard
        untiled_reference = dist_layer(x_dt.clone(), mask_dt)

        assert_output_placement(
            untiled_reference,
            expected_placements=PL_PAIR,
            label="TriMul/tiled-parity/ref",
        )
        ref_local = untiled_reference.to_local()
        assert not torch.equal(ref_local, x_dt.to_local()), "tiled-parity update is trivially zero"

        for tile_size in tile_sizes:
            dist_layer._tile_size = tile_size
            residual_input = x_dt.clone()
            tiled_result = dist_layer(residual_input, mask_dt)
            assert tiled_result is residual_input

            max_diff = (tiled_result.to_local().float() - ref_local.float()).abs().max().item()
            print(f"{rank}:  tile_size={tile_size}: max_abs_diff = {max_diff:.2e}")

            assert max_diff == 0.0, (
                f"Tiled path diverged at tile_size={tile_size}: "
                f"max_abs_diff={max_diff} (rank {rank})"
            )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("direction", ("outgoing", "incoming"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16), ids=["32-true", "bf16-mixed"])
def test_trimul_tiled_parity(direction: str, dtype: torch.dtype) -> None:
    spawn_cp_test(partial(_trimul_parity_worker, direction=direction, dtype=dtype))
