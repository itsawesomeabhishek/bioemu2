from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.transition import Transition
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import tiled_params_replicated as tiled_ops
from bioemu2.context_parallel.ops.transition import DTensorAwareTransition
from bioemu2.context_parallel.utils.dtensor import (
    PL_PAIR,
    PL_SEQ,
    shard_along_token,
    shard_pair_along_token,
)
from bioemu2.context_parallel.utils.padding import (
    pad_along_token,
    pad_amounts,
    pad_pair_along_token,
    unpad_along_token,
    unpad_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
    spawn_cpu_cp_test,
)

DIM = 17
HIDDEN = 31
OUT_DIM = 13


def _make_transition(device: torch.device, use_norm: bool) -> Transition:
    torch.manual_seed(1234)
    serial = Transition(
        dim=DIM,
        hidden=HIDDEN,
        out_dim=OUT_DIM,
        use_norm=use_norm,
    ).to(device)
    randomize_params(serial)
    return serial.eval()


def _run_single_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    use_norm: bool,
    tile_rows: bool,
) -> None:
    n_token = 65
    n_pad = pad_amounts(n_token, n_atom=0, n_msa=0, W=1, mesh=mesh)["n_token_pad"]
    serial = _make_transition(device, use_norm)

    torch.manual_seed(5678)
    x = torch.randn(1, n_token, DIM, device=device)

    # A one-element budget forces one-row tiles.
    tile_budget = patch.object(tiled_ops, "TILE_ELEMENTS", 1) if tile_rows else nullcontext()
    with torch.no_grad(), tile_budget:
        expected = serial(x)
        [x_dt] = shard_along_token(pad_along_token([x], n_pad), mesh)
        actual_dt = DTensorAwareTransition(serial, mesh).to(device)(x_dt)

    label = f"transition/single/norm={use_norm}/tile_rows={tile_rows}"
    assert actual_dt.shape == torch.Size((1, n_token + n_pad, OUT_DIM))
    assert_output_placement(actual_dt, expected_placements=PL_SEQ, label=label)
    assert_replicate_values_against_rank0(actual_dt)
    [actual] = unpad_along_token([actual_dt.full_tensor()], n_token)
    assert expected.abs().max().item() > 0, f"{label}: serial output is trivially zero"
    gather_n_assert_parity_at_rank0(expected, actual, label=label, atol=1e-6)


def _run_pair_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    use_norm: bool,
    tile_rows: bool,
) -> None:
    n_token = 9
    n_pad = pad_amounts(n_token, n_atom=0, n_msa=0, W=1, mesh=mesh)["n_token_pad"]
    serial = _make_transition(device, use_norm)

    torch.manual_seed(9012)
    z = torch.randn(1, n_token, n_token, DIM, device=device)

    # A one-element budget forces one-row tiles.
    tile_budget = patch.object(tiled_ops, "TILE_ELEMENTS", 1) if tile_rows else nullcontext()
    with torch.no_grad(), tile_budget:
        expected = serial(z)
        [z_dt] = shard_pair_along_token(pad_pair_along_token([z], n_pad), mesh)
        actual_dt = DTensorAwareTransition(serial, mesh).to(device)(z_dt)

    label = f"transition/pair/norm={use_norm}/tile_rows={tile_rows}"
    assert actual_dt.shape == torch.Size((1, n_token + n_pad, n_token + n_pad, OUT_DIM))
    assert_output_placement(actual_dt, expected_placements=PL_PAIR, label=label)
    assert_replicate_values_against_rank0(actual_dt)
    [actual] = unpad_pair_along_token([actual_dt.full_tensor()], n_token)
    assert expected.abs().max().item() > 0, f"{label}: serial output is trivially zero"
    gather_n_assert_parity_at_rank0(expected, actual, label=label, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("use_norm", (False, True))
@pytest.mark.parametrize("tile_rows", (False, True))
def test_single_transition_parity(use_norm: bool, tile_rows: bool) -> None:
    """Transition preserves serial values, output width, and CP placements."""
    spawn_cp_test(partial(_run_single_parity, use_norm=use_norm, tile_rows=tile_rows))


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("use_norm", (False, True))
@pytest.mark.parametrize("tile_rows", (False, True))
def test_pair_transition_parity(use_norm: bool, tile_rows: bool) -> None:
    spawn_cp_test(partial(_run_pair_parity, use_norm=use_norm, tile_rows=tile_rows))


@pytest.mark.parametrize("run", (_run_single_parity, _run_pair_parity))
@pytest.mark.parametrize("tile_rows", (False, True))
def test_transition_parity_cpu(run: Callable[..., None], tile_rows: bool) -> None:
    spawn_cpu_cp_test(partial(run, use_norm=True, tile_rows=tile_rows))
