from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import shardwise_cat
from bioemu2.context_parallel.utils.dtensor import (
    PL_PAIR,
    shard_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    spawn_cp_test,
)

TINY_SHAPE = dict(B=2, N=8, C=16)
REPRESENTATIVE_SHAPE = dict(B=2, N=64, C=128)


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    dims: dict[str, int],
    seed: int,
    label: str,
    three_tensors: bool,
) -> None:
    """Cat pair tensors along the channel (last) dim — replicated on mesh."""
    B, N, C = dims["B"], dims["N"], dims["C"]

    torch.manual_seed(seed)
    channels = (C, C // 2, C * 2) if three_tensors else (C, C)
    tensors = [
        torch.randn(B, N, N, width, device=device, dtype=torch.float32) for width in channels
    ]

    out_serial = torch.cat(tensors, dim=-1)
    assert out_serial.abs().max().item() > 0, "serial output is trivially zero"

    # Shard pair axes; concatenate the unsharded channel axis.
    tensors_dt = shard_pair_along_token(tensors, mesh)

    out_dt = shardwise_cat(tensors_dt, dim=-1)

    label = f"shardwise_cat/{label}"
    assert_output_placement(out_dt, expected_placements=PL_PAIR, label=label)
    assert_replicate_values_against_rank0(out_dt)
    gather_n_assert_parity_at_rank0(out_serial, out_dt, label=label, atol=0)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", [TINY_SHAPE, REPRESENTATIVE_SHAPE], ids=["tiny", "representative"])
@pytest.mark.parametrize(
    "seed, label, three_tensors",
    [(5678, "2_tensors", False), (9012, "3_tensors", True)],
    ids=["2_tensors", "3_tensors"],
)
def test_shardwise_cat_parity(
    dims: dict[str, int], seed: int, label: str, three_tensors: bool
) -> None:
    spawn_cp_test(
        partial(_run_parity, dims=dims, seed=seed, label=label, three_tensors=three_tensors)
    )
