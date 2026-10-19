"""Parity tests for the forward-only row-tiled LayerNorm-Linear layer."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import tiled_params_replicated as tiled_ops
from bioemu2.context_parallel.ops.tiled_params_replicated import (
    LayerNormLinearParamsReplicatedTiled,
)
from bioemu2.context_parallel.utils.dtensor import PL_PAIR, shard_pair_along_token
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
    spawn_cpu_cp_test,
)

N_TOKEN = 10
DIM = 12
OUT_DIM = 7


def _tiled_layer_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    input_dtype: torch.dtype,
    tile_pairs: bool,
    concatenate: bool,
) -> None:
    torch.manual_seed(1234)
    serial = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, OUT_DIM)).to(device).eval()
    randomize_params(serial)
    tiled = LayerNormLinearParamsReplicatedTiled(serial, mesh).to(device).eval()
    serial.to(input_dtype)
    z = torch.randn(2, N_TOKEN, N_TOKEN, DIM, device=device).to(input_dtype)
    pairs = shard_pair_along_token([z[..., :5], z[..., 5:]] if concatenate else [z], mesh)

    # A one-element budget forces one-row tiles.
    tile_budget = patch.object(tiled_ops, "TILE_ELEMENTS", 1) if tile_pairs else nullcontext()
    with torch.no_grad(), tile_budget:
        expected = serial(z)
        actual = tiled(*pairs)

    tol = 1e-6 if input_dtype == torch.float32 else 2e-2
    label = f"tiled/{input_dtype}/tile_pairs={tile_pairs}/concatenate={concatenate}"
    assert actual.dtype == input_dtype, label
    assert actual.shape == expected.shape, label
    assert expected.abs().max().item() > 0, f"{label}: serial output is trivially zero"
    assert_output_placement(actual, expected_placements=PL_PAIR, label=label)
    assert_replicate_values_against_rank0(actual)
    gather_n_assert_parity_at_rank0(expected, actual, label=label, atol=tol)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("tile_pairs", (False, True))
@pytest.mark.parametrize("concatenate", (False, True))
def test_tiled_layer_parity(input_dtype: torch.dtype, tile_pairs: bool, concatenate: bool) -> None:
    spawn_cp_test(
        partial(
            _tiled_layer_parity,
            input_dtype=input_dtype,
            tile_pairs=tile_pairs,
            concatenate=concatenate,
        )
    )


@pytest.mark.parametrize("input_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("tile_pairs", (False, True))
@pytest.mark.parametrize("concatenate", (False, True))
def test_tiled_layer_parity_cpu(
    input_dtype: torch.dtype, tile_pairs: bool, concatenate: bool
) -> None:
    spawn_cpu_cp_test(
        partial(
            _tiled_layer_parity,
            input_dtype=input_dtype,
            tile_pairs=tile_pairs,
            concatenate=concatenate,
        )
    )
