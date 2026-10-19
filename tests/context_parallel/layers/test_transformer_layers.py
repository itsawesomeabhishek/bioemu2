"""Canonical BioMoa transformer-layer tests."""

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.diffusion import DualStreamDiffusionLayer
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.layers.transformer_layers import (
    DistributedDualStreamDiffusionLayer,
)
from bioemu2.context_parallel.ops.distributed_primitives.flatten_and_unflatten import (
    shardwise_flatten_sharded,
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

BATCH = 1
MULTIPLICITY = 2
TOKENS = 8
ATOMS = 32
WINDOW = 8


def _dual_stream_layer_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dm: DistributedManager,
) -> None:
    torch.manual_seed(1234)
    serial = (
        DualStreamDiffusionLayer(
            a_s=16,
            a_z=8,
            atom_s=16,
            atom_z=8,
            atoms_per_window_queries=WINDOW,
            atoms_per_window_keys=WINDOW,
            atom_attention_heads=2,
            token_attention_heads=2,
            use_lns=False,
        )
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    s = torch.randn(BATCH * MULTIPLICITY, TOKENS, 16, device=device)
    z = torch.randn(BATCH, TOKENS, TOKENS, 8, device=device)
    q = torch.randn(BATCH * MULTIPLICITY, ATOMS, 16, device=device)
    p = torch.randn(BATCH, ATOMS // WINDOW, WINDOW, WINDOW, 8, device=device)
    c = torch.randn(BATCH * MULTIPLICITY, TOKENS, 16, device=device)
    atom_mask = torch.ones(BATCH, ATOMS, dtype=torch.bool, device=device)
    token_mask = torch.ones(BATCH, TOKENS, dtype=torch.bool, device=device)
    atom_ids = (torch.arange(ATOMS, device=device) // 4).unsqueeze(0)
    is_padded_atom = torch.zeros(BATCH, ATOMS, dtype=torch.bool, device=device)
    center_indices = (torch.arange(TOKENS, device=device) * 4 + 2).unsqueeze(0)

    with torch.no_grad():
        s_expected, q_expected = serial(
            s,
            z,
            q,
            p,
            c,
            atom_mask,
            token_mask,
            atom_ids,
            is_padded_atom,
            center_indices,
            MULTIPLICITY,
        )

    distributed = DistributedDualStreamDiffusionLayer(serial, dm).to(device).eval()
    s_dt, c_dt, token_mask_dt, center_indices_dt = shard_along_token(
        [
            s,
            c,
            token_mask.repeat_interleave(MULTIPLICITY, 0),
            center_indices.repeat_interleave(MULTIPLICITY, 0).unsqueeze(-1),
        ],
        mesh,
        src_data_rank=None,
    )
    (token_attention_mask_dt,) = shard_along_token(
        [token_mask],
        mesh,
        src_data_rank=None,
    )
    (z_dt,) = shard_pair_along_token([z], mesh, src_data_rank=None)
    q_dt, atom_ids_dt, is_padded_atom_dt = shard_along_window(
        [
            q.view(BATCH * MULTIPLICITY, ATOMS // WINDOW, WINDOW, 16),
            atom_ids.repeat_interleave(MULTIPLICITY, 0).view(
                BATCH * MULTIPLICITY, ATOMS // WINDOW, WINDOW
            ),
            is_padded_atom.repeat_interleave(MULTIPLICITY, 0).view(
                BATCH * MULTIPLICITY, ATOMS // WINDOW, WINDOW
            ),
        ],
        mesh,
        src_data_rank=None,
    )
    p_dt, atom_mask_dt = shard_along_window(
        [p, atom_mask.view(BATCH, ATOMS // WINDOW, WINDOW)],
        mesh,
        src_data_rank=None,
    )
    with torch.no_grad():
        s_actual_dt, q_actual_dt = distributed(
            s_dt,
            z_dt,
            q_dt,
            p_dt,
            c_dt,
            atom_mask_dt,
            token_attention_mask_dt,
            token_mask_dt,
            atom_ids_dt,
            is_padded_atom_dt,
            center_indices_dt,
            MULTIPLICITY,
        )

    for label, output in (("token", s_actual_dt), ("atom", q_actual_dt)):
        assert_output_placement(
            output, expected_placements=PL_SEQ, label=f"BioMoaC dual-stream layer/{label}"
        )
        assert_replicate_values_against_rank0(output)
    gather_n_assert_parity_at_rank0(
        s_expected,
        s_actual_dt,
        label="BioMoaC distributed dual-stream layer/token",
        atol=5e-5,
    )
    gather_n_assert_parity_at_rank0(
        q_expected,
        shardwise_flatten_sharded(q_actual_dt, start_dim=1, end_dim=2),
        label="BioMoaC distributed dual-stream layer/atom",
        atol=5e-5,
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_dual_stream_layer_parity() -> None:
    spawn_cp_test(_dual_stream_layer_parity)
