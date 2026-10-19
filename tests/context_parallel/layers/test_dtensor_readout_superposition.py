from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.modules.diffusion import ReadoutSuperposition
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import DTensorReadoutSuperposition
from bioemu2.context_parallel.ops.distributed_primitives.flatten_and_unflatten import (
    shardwise_flatten_sharded,
)
from bioemu2.context_parallel.utils.dtensor import (
    PL_SEQ,
    gather_along_atom,
    shard_along_atom,
    shard_along_token,
    shard_along_window,
)
from bioemu2.context_parallel.utils.padding import (
    pad_along_atom,
    pad_along_token,
    pad_amounts,
    unpad_along_atom,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
)

# Non-divisible lengths exercise padding; readout uses W=1 (no windowed attention).
TINY_SHAPE = dict(batch=2, multiplicity=2, n_token=7, n_atom=13, a_s=32, atom_s=24, hidden=48)
REPRESENTATIVE_SHAPE = dict(
    batch=2, multiplicity=2, n_token=125, n_atom=63, a_s=128, atom_s=128, hidden=384
)

# Tolerance covers Linear/LayerNorm rounding; one-hot gathers are exact.
FWD_ATOL = 5e-5


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dm: DistributedManager,
    dims: dict[str, int],
) -> None:
    B = dims["batch"]
    M = dims["multiplicity"]
    BM = B * M
    N_token_real = dims["n_token"]
    N_atom_real = dims["n_atom"]

    pads = pad_amounts(N_token_real, N_atom_real, n_msa=0, W=1, mesh=mesh)
    n_token_pad = pads["n_token_pad"]
    n_atom_pad = pads["n_atom_pad"]

    torch.manual_seed(1234)
    serial = (
        ReadoutSuperposition(atom_s=dims["atom_s"], token_s=dims["a_s"], hidden_dim=dims["hidden"])
        .to(device)
        .float()
        .eval()
    )
    randomize_params(serial)

    torch.manual_seed(5678)
    a_real = torch.randn(BM, N_token_real, dims["a_s"], device=device, dtype=torch.float32)
    q_real = torch.randn(BM, N_atom_real, dims["atom_s"], device=device, dtype=torch.float32)
    # All atoms map to real tokens; unassigned atoms are outside this test's scope.
    token_ids = torch.randint(0, N_token_real, (B, N_atom_real), device=device)
    atom_pad_mask_real = torch.ones(B, N_atom_real, device=device)

    with torch.no_grad():
        out_serial = serial(
            a=a_real,
            q=q_real,
            atom_to_token_ids=token_ids,
            is_padded_atom=~atom_pad_mask_real.bool(),
            multiplicity=M,
        )
    assert out_serial.abs().max().item() > 0, "serial output is trivially zero"

    a_dt = shard_along_token(pad_along_token([a_real], n_pad=n_token_pad), mesh=mesh)[0]
    q_dt = shard_along_atom(pad_along_atom([q_real], n_pad=n_atom_pad), mesh=mesh)[0]

    dist_readout = DTensorReadoutSuperposition(serial, dm.device_mesh_subgroups).to(device).eval()

    # Match production's window sharding; zero-padded indices are excluded by unpadding.
    atom_to_token_ids_padded = pad_along_atom([token_ids], n_atom_pad)[0]
    N_atom_pad = atom_to_token_ids_padded.shape[1]
    tok_idx_mult = atom_to_token_ids_padded.repeat_interleave(M, 0)  # (B*M, N_atom_pad)
    tok_idx_windowed = tok_idx_mult.view(BM, N_atom_pad, 1)  # K=N_atom_pad, W=1
    [tok_idx_dt] = shard_along_window([tok_idx_windowed], mesh, src_data_rank=None)
    tok_idx_flat_dt = shardwise_flatten_sharded(tok_idx_dt, start_dim=1, end_dim=2)

    out_cp_dt = dist_readout(
        a=a_dt,
        q=q_dt,
        tok_idx_flat_dt=tok_idx_flat_dt,
    )
    assert_output_placement(
        out_cp_dt,
        expected_placements=PL_SEQ,
        label="DTensorReadoutSuperposition/padded",
    )

    assert_replicate_values_against_rank0(out_cp_dt)
    out_cp_real = unpad_along_atom(gather_along_atom([out_cp_dt]), n_real=N_atom_real)[0]
    gather_n_assert_parity_at_rank0(
        out_serial,
        out_cp_real,
        label=(
            f"DTensorReadoutSuperposition/padded("
            f"n_tok={N_token_real}->{N_token_real + n_token_pad},"
            f"n_atom={N_atom_real}->{N_atom_real + n_atom_pad})"
        ),
        atol=FWD_ATOL,
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", [TINY_SHAPE, REPRESENTATIVE_SHAPE])
def test_dtensor_readout_superposition_parity(dims: dict[str, int]) -> None:
    """Real (non-divisible) input length: serial(real) vs unpad(CP(pad(real)))."""
    spawn_cp_test(partial(_run_parity, dims=dims))
