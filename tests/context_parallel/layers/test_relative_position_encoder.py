"""Canonical BioMoa relative-position encoder tests."""

from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.relative_position_encoder import FusedRelativePositionEncoder
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.layers.relative_position_encoder import (
    DistributedRelativePositionEncoder,
)
from bioemu2.context_parallel.utils.dtensor import PL_PAIR, gather_pair_along_token
from bioemu2.context_parallel.utils.padding import (
    pad_along_token,
    pad_amounts,
    unpad_pair_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

TINY_SHAPE = dict(n_tokens=7, token_z=11)
LARGE_SHAPE = dict(n_tokens=125, token_z=128)


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
) -> None:
    n_tokens = dims["n_tokens"]
    token_z = dims["token_z"]
    n_pad = pad_amounts(n_tokens, n_atom=0, n_msa=0, W=1, mesh=mesh)["n_token_pad"]

    torch.manual_seed(1234)
    serial = FusedRelativePositionEncoder(token_z=token_z).to(device).float().eval()
    randomize_params(serial)

    token_ids = torch.arange(n_tokens, device=device)
    feats = {
        "asym_id": (token_ids // 4).unsqueeze(0),
        "entity_id": (token_ids // 8).unsqueeze(0),
        "residue_index": (token_ids % 6).unsqueeze(0),
        "sym_id": ((token_ids // 3) % 4).unsqueeze(0),
    }
    keys = list(feats)
    padded = dict(zip(keys, pad_along_token([feats[key] for key in keys], n_pad)))

    with torch.no_grad():
        expected = serial(**feats)
        actual_dt = DistributedRelativePositionEncoder(serial, mesh)(
            asym_id=padded["asym_id"],
            entity_id=padded["entity_id"],
            residue_index=padded["residue_index"],
            sym_id=padded["sym_id"],
        )

    assert expected.abs().max().item() > 0
    assert_output_placement(actual_dt, expected_placements=PL_PAIR, label="BioMoaRelPos")
    assert_replicate_values_against_rank0(actual_dt)
    actual = unpad_pair_along_token(gather_pair_along_token([actual_dt]), n_real=n_tokens)[0]
    gather_n_assert_parity_at_rank0(expected, actual, label="BioMoaRelPos", atol=0)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", [TINY_SHAPE, LARGE_SHAPE])
def test_relative_position_encoder_parity(dims: dict[str, int]) -> None:
    spawn_cp_test(partial(_run_parity, dims=dims))
