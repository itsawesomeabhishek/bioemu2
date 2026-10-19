"""Canonical BioMoa attention tests."""

from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.attention_pair_bias import (
    AttentionPairBias as SerialAttentionPairBias,
)
from bioemu2.biomoa.ops.attention_pair_bias import (
    AttentionPairBiasSlidingWindow as SerialAttentionPairBiasSlidingWindow,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.attention import (
    AttentionPairBias as DistributedAttentionPairBias,
)
from bioemu2.context_parallel.ops.attention import (
    AttentionPairBiasSlidingWindow as DistributedAttentionPairBiasSlidingWindow,
)
from bioemu2.context_parallel.utils.dtensor import (
    PL_SEQ,
    shard_along_token,
    shard_along_window,
    shard_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

FULL_SHAPES = (
    dict(num_tokens=16, c_s=64, c_z=32, num_heads=4),
    dict(num_tokens=32, c_s=128, c_z=128, num_heads=8),
)
WINDOW_SHAPES = (
    dict(num_windows=8, window_queries=8, window_keys=16, c_s=64, c_z=32, num_heads=4),
    dict(num_windows=8, window_queries=8, window_keys=16, c_s=128, c_z=128, num_heads=8),
)
BATCH_SIZE = 2
MULTIPLICITY = 2
RING_ATOL = 5e-7
SHARDWISE_ATOL = 5e-7


def _full_attention_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
) -> None:
    torch.manual_seed(1234)
    serial = (
        SerialAttentionPairBias(
            c_s=dims["c_s"],
            c_z=dims["c_z"],
            num_heads=dims["num_heads"],
            initial_norm=True,
            tile_size=None,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    s = torch.randn(
        BATCH_SIZE * MULTIPLICITY,
        dims["num_tokens"],
        dims["c_s"],
        device=device,
    )
    z = torch.randn(
        BATCH_SIZE,
        dims["num_tokens"],
        dims["num_tokens"],
        dims["c_z"],
        device=device,
    )
    mask = torch.ones(BATCH_SIZE, dims["num_tokens"], device=device, dtype=torch.bool)
    mask[:, -2:] = False

    with torch.no_grad():
        out_serial = serial(s, z, mask, multiplicity=MULTIPLICITY)

    assert out_serial.abs().max().item() > 0

    distributed = DistributedAttentionPairBias(serial, dist_manager).to(device).eval()
    s_dt, mask_dt = shard_along_token([s, mask], mesh)
    [z_dt] = shard_pair_along_token([z], mesh)

    with torch.no_grad():
        out_distributed = distributed(s_dt, z_dt, mask_dt, multiplicity=MULTIPLICITY)

    assert_output_placement(
        out_distributed,
        expected_placements=PL_SEQ,
        label="BioMoa full attention",
    )
    assert_replicate_values_against_rank0(out_distributed)
    gather_n_assert_parity_at_rank0(
        out_serial,
        out_distributed,
        label="BioMoa full attention",
        atol=RING_ATOL,
        mask=mask.repeat_interleave(MULTIPLICITY, 0).unsqueeze(-1),
    )


def _sliding_window_attention_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
) -> None:
    torch.manual_seed(1234)
    serial = (
        SerialAttentionPairBiasSlidingWindow(
            c_s=dims["c_s"],
            c_z=dims["c_z"],
            num_heads=dims["num_heads"],
            window_queries=dims["window_queries"],
            window_keys=dims["window_keys"],
            initial_norm=True,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    s = torch.randn(
        BATCH_SIZE * MULTIPLICITY,
        dims["num_windows"],
        dims["window_queries"],
        dims["c_s"],
        device=device,
    )
    z = torch.randn(
        BATCH_SIZE,
        dims["num_windows"],
        dims["window_queries"],
        dims["window_keys"],
        dims["c_z"],
        device=device,
    )
    mask = torch.ones(
        BATCH_SIZE,
        dims["num_windows"],
        dims["window_queries"],
        device=device,
        dtype=torch.bool,
    )
    mask[:, -1, -2:] = False

    with torch.no_grad():
        out_serial = serial(
            s.flatten(0, 1),
            z,
            mask.flatten(0, 1),
            multiplicity=MULTIPLICITY,
        ).unflatten(0, (BATCH_SIZE * MULTIPLICITY, dims["num_windows"]))
    assert out_serial.abs().max().item() > 0

    distributed = DistributedAttentionPairBiasSlidingWindow(serial, mesh).to(device).eval()
    s_dt, z_dt, mask_dt = shard_along_window([s, z, mask], mesh)

    with torch.no_grad():
        out_distributed = distributed(s_dt, z_dt, mask_dt, multiplicity=MULTIPLICITY)

    assert_output_placement(
        out_distributed,
        expected_placements=PL_SEQ,
        label="BioMoa sliding-window attention",
    )
    assert_replicate_values_against_rank0(out_distributed)
    gather_n_assert_parity_at_rank0(
        out_serial,
        out_distributed,
        label="BioMoa sliding-window attention",
        atol=SHARDWISE_ATOL,
        mask=mask.repeat_interleave(MULTIPLICITY, 0).unsqueeze(-1),
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", FULL_SHAPES)
def test_biomoa_attention_pair_bias_parity(dims: dict[str, int]) -> None:
    spawn_cp_test(partial(_full_attention_parity, dims=dims))


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", WINDOW_SHAPES)
def test_biomoa_sliding_window_attention_pair_bias_parity(dims: dict[str, int]) -> None:
    spawn_cp_test(partial(_sliding_window_attention_parity, dims=dims))
