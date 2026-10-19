"""Canonical BioMoa diffusion tests."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.diffusion import DualStreamDiffusionModule
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.modules.diffusion import (
    DistributedDualStreamDiffusionModule,
)
from bioemu2.context_parallel.ops import tiled_params_replicated as tiled_ops
from bioemu2.context_parallel.utils.dtensor import PL_SEQ, shard_pair_along_token
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
    spawn_cpu_cp_test,
)

BATCH = 2
MULTIPLICITY = 2
TOKENS = 8
ATOMS = 32
WINDOW = 8
REPLICA_ATOL = 5e-7


def _diffusion_module_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dm: DistributedManager,
    *,
    tile_pairs: bool = False,
) -> None:
    torch.manual_seed(1234)
    serial = (
        DualStreamDiffusionModule(
            token_s=16,
            token_z=8,
            atom_s=16,
            atom_z=8,
            s_input_dim=6,
            a_s=16,
            a_z=8,
            num_layers=1,
            atoms_per_window_queries=WINDOW,
            atoms_per_window_keys=WINDOW,
            dim_fourier=16,
            atom_attention_heads=2,
            token_attention_heads=2,
            conditioning_transition_layers=1,
            use_lns=False,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    s_inputs = torch.randn(BATCH, TOKENS, 6, device=device)
    s_trunk = torch.randn(BATCH, TOKENS, 16, device=device)
    z_trunk = torch.randn(BATCH, TOKENS, TOKENS, 8, device=device)
    rpe = torch.randn(BATCH, TOKENS, TOKENS, 8, device=device)
    r_noisy = torch.randn(BATCH * MULTIPLICITY, ATOMS, 3, device=device)
    times = torch.full((BATCH * MULTIPLICITY,), 0.1, device=device)
    atom_ids = (torch.arange(ATOMS, device=device) // 4).repeat(BATCH, 1)
    center_indices = (torch.arange(TOKENS, device=device) * 4 + 2).repeat(BATCH, 1)
    feats = {
        "template_coords": torch.randn(BATCH, ATOMS, 3, device=device),
        "charge": torch.randn(BATCH, ATOMS, device=device),
        "element": torch.randint(0, 128, (BATCH, ATOMS), device=device),
        "name_encoding": torch.randint(0, 64, (BATCH, ATOMS, 4), device=device),
        "ref_space_uid": (torch.arange(ATOMS, device=device) // 2).repeat(BATCH, 1),
        "atom_pad_mask": torch.ones(BATCH, ATOMS, dtype=torch.bool, device=device),
        "atom_mask": torch.ones(BATCH, ATOMS, dtype=torch.bool, device=device),
        "token_idx_at_atom": atom_ids,
        "token_pad_mask": torch.ones(BATCH, TOKENS, dtype=torch.bool, device=device),
        "token_center_atom_index": center_indices,
    }
    for tensor in [s_inputs, s_trunk, r_noisy, times, *feats.values()]:
        dist.broadcast(tensor, src=0)

    with torch.no_grad():
        output_serial = serial(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            r_noisy=r_noisy,
            times=times,
            relative_position_encoding=rpe,
            feats=feats,
            multiplicity=MULTIPLICITY,
        )

    distributed = DistributedDualStreamDiffusionModule(serial, dm).to(device).eval()
    z_trunk_dt, rpe_dt = shard_pair_along_token(
        [z_trunk, rpe],
        mesh,
        src_data_rank=None,
    )
    # A one-element budget forces one-row tiles in every tiled pair layer.
    tile_budget = patch.object(tiled_ops, "TILE_ELEMENTS", 1) if tile_pairs else nullcontext()
    with torch.no_grad(), tile_budget:
        actual = distributed(
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk_dt,
            r_noisy=r_noisy,
            times=times,
            relative_position_encoding=rpe_dt,
            feats=feats,
            multiplicity=MULTIPLICITY,
        )

    for key in ("r_update", "token_a"):
        assert_output_placement(
            actual[key], expected_placements=PL_SEQ, label=f"BioMoaC diffusion module/{key}"
        )
        assert_replicate_values_against_rank0(actual[key], atol=REPLICA_ATOL)
    gather_n_assert_parity_at_rank0(
        output_serial["r_update"],
        actual["r_update"],
        label="BioMoaC distributed diffusion module/readout",
        atol=5e-5,
    )
    gather_n_assert_parity_at_rank0(
        output_serial["token_a"],
        actual["token_a"],
        label="BioMoaC distributed diffusion module/token",
        atol=5e-5,
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("tile_pairs", [False, True])
def test_diffusion_module_parity(tile_pairs: bool) -> None:
    spawn_cp_test(partial(_diffusion_module_parity, tile_pairs=tile_pairs))


@pytest.mark.parametrize("tile_pairs", [False, True])
def test_diffusion_module_parity_cpu(tile_pairs: bool) -> None:
    spawn_cpu_cp_test(partial(_diffusion_module_parity, tile_pairs=tile_pairs))
