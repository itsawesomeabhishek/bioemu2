"""Canonical BioMoa pairmixer tests."""

from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.pairmixer import PairMixerLayer
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.modules.pairmixer import DistributedPairMixerLayer
from bioemu2.context_parallel.utils.dtensor import (
    PL_PAIR,
    gather_pair_along_token,
    shard_pair_along_token,
)
from bioemu2.context_parallel.utils.padding import (
    pad_amounts,
    pad_pair_along_token,
    unpad_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

N_TOKENS = 7
TINY_SHAPE = dict(token_s=32, token_z=32)
DEEP_SHAPE = dict(token_s=128, token_z=128)


def _run_pairmixer_layer_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
) -> None:
    mesh = dist_manager.device_mesh_subgroups

    torch.manual_seed(1234)
    serial = (
        PairMixerLayer(
            token_s=dims["token_s"],
            token_z=dims["token_z"],
            dropout=0.0,
            trimul_variant="full",
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    z = torch.randn(1, N_TOKENS, N_TOKENS, dims["token_z"], device=device)
    pair_mask = torch.ones(1, N_TOKENS, N_TOKENS, dtype=torch.bool, device=device)
    pair_mask[:, -1, :] = False
    pair_mask[:, :, -1] = False

    with torch.no_grad():
        _, serial_z = serial(
            torch.randn(1, N_TOKENS, dims["token_s"], device=device),
            z,
            pair_mask,
        )
    assert serial_z.abs().max().item() > 0

    n_pad = pad_amounts(N_TOKENS, n_atom=0, n_msa=0, W=1, mesh=mesh)["n_token_pad"]
    z_dt, mask_dt = shard_pair_along_token(
        pad_pair_along_token([z, pair_mask], n_pad),
        mesh,
    )
    distributed = DistributedPairMixerLayer(serial, dist_manager).to(device).eval()
    with torch.no_grad():
        cp_z = distributed(z_dt, mask_dt)

    assert_output_placement(cp_z, expected_placements=PL_PAIR, label="BioMoaPairMixer")
    assert_replicate_values_against_rank0(cp_z)
    cp_z = unpad_pair_along_token(gather_pair_along_token([cp_z]), N_TOKENS)[0]
    gather_n_assert_parity_at_rank0(serial_z, cp_z, label="BioMoaPairMixer", atol=1e-6)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("shape", (TINY_SHAPE, DEEP_SHAPE))
def test_pairmixer_parity(shape: dict[str, int]) -> None:
    spawn_cp_test(partial(_run_pairmixer_layer_parity, dims=shape))
