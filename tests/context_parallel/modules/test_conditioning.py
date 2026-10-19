"""Canonical BioMoa method-conditioning tests."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.conditioning import MethodConditioning
from bioemu2.conditioning.method import num_method_types
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.modules.conditioning import DistributedMethodConditioning
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

TINY_SHAPE = dict(n_tokens=7, token_s=13, token_z=11)
LARGE_SHAPE = dict(n_tokens=125, token_s=384, token_z=128)


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
    embed_into_pairs: bool,
) -> None:
    n_tokens = dims["n_tokens"]
    n_pad = pad_amounts(n_tokens, n_atom=0, n_msa=0, W=1, mesh=mesh)["n_token_pad"]

    torch.manual_seed(1234)
    serial = (
        MethodConditioning(
            token_s=dims["token_s"],
            token_z=dims["token_z"],
            embed_into_pairs=embed_into_pairs,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    method_feature = torch.randint(0, num_method_types, (1, n_tokens), device=device)
    token_pad_mask = torch.ones(1, n_tokens, dtype=torch.bool, device=device)
    token_pad_mask[:, -2:] = False
    padded_feature, padded_mask = pad_along_token([method_feature, token_pad_mask], n_pad)

    dist.broadcast(padded_feature, src=0)
    dist.broadcast(padded_mask, src=0)

    with torch.no_grad():
        single_serial, pair_serial = serial(method_feature, token_pad_mask)
        single, pair_dt = DistributedMethodConditioning(serial, mesh)(padded_feature, padded_mask)

    label = f"BioMoaMethodConditioning(embed_into_pairs={embed_into_pairs})"
    gather_n_assert_parity_at_rank0(
        single_serial, single[:, :n_tokens], label=f"{label}/single", atol=0
    )
    if not embed_into_pairs:
        assert pair_serial is None and pair_dt is None
        return

    assert pair_serial is not None and pair_dt is not None
    assert pair_serial.abs().max().item() > 0
    assert_output_placement(pair_dt, expected_placements=PL_PAIR, label=f"{label}/pair")
    assert_replicate_values_against_rank0(pair_dt)
    actual_pair = unpad_pair_along_token(gather_pair_along_token([pair_dt]), n_tokens)[0]
    gather_n_assert_parity_at_rank0(pair_serial, actual_pair, label=f"{label}/pair", atol=0)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", [TINY_SHAPE, LARGE_SHAPE])
@pytest.mark.parametrize("embed_into_pairs", [False, True])
def test_method_conditioning_parity(dims: dict[str, int], embed_into_pairs: bool) -> None:
    spawn_cp_test(partial(_run_parity, dims=dims, embed_into_pairs=embed_into_pairs))
