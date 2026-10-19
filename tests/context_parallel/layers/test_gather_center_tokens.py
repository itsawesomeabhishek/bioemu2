from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.atom_token_ops import (
    gather_center_tokens_from_atomic as serial_gather_center_tokens_from_atomic,
)
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.atom_token_ops import (
    gather_center_tokens_from_atomic,
)
from bioemu2.context_parallel.ops.distributed_primitives.gather import distributed_gather
from bioemu2.context_parallel.utils.dtensor import (
    PL_SEQ,
    gather_along_token,
    gather_along_window,
    shard_along_token,
    shard_along_window,
)
from bioemu2.context_parallel.utils.padding import (
    pad_along_atom,
    pad_along_token,
    pad_amounts,
    pad_dim,
    unpad_along_atom,
    unpad_along_token,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    spawn_cp_test,
)

W = 4  # atoms per window (queries) — must divide N_atom_padded.

# Non-divisible lengths exercise token padding and W*shard_multiple atom padding.
TINY_SHAPE = dict(batch=1, n_token=7, n_atom=13, d=32)
REPRESENTATIVE_SHAPE = dict(batch=1, n_token=125, n_atom=125, d=64)

ATOL = 1e-6


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    dims: dict[str, int],
    multiplicity: int,
) -> None:
    B = dims["batch"]
    N_token_real = dims["n_token"]
    N_atom_real = dims["n_atom"]
    D = dims["d"]

    pads = pad_amounts(N_token_real, N_atom_real, n_msa=0, W=W, mesh=mesh)
    n_token_pad = pads["n_token_pad"]
    n_atom_pad = pads["n_atom_pad"]
    N_atom_padded = N_atom_real + n_atom_pad
    assert N_atom_padded % W == 0, (
        f"N_atom_padded={N_atom_padded} not divisible by W={W} (pad_amounts bug?)"
    )
    K = N_atom_padded // W

    # Atom-to-token gather.
    BM = B * multiplicity
    label = (
        f"gather_center_tokens/M={multiplicity}/"
        f"(N_tok={N_token_real}->{N_token_real + n_token_pad},"
        f"N_atom={N_atom_real}->{N_atom_padded},K={K})"
    )

    torch.manual_seed(42 + multiplicity)
    q_flat_real = torch.randn(BM, N_atom_real, D, device=device, dtype=torch.float32)

    # Center indices reference real atoms, never atom padding.
    token_center_atom_index = torch.randint(0, N_atom_real, (B, N_token_real), device=device)

    token_mask = torch.ones(B, N_token_real, device=device, dtype=torch.float32)
    if N_token_real >= 2:
        token_mask[:, -2:] = 0.0  # last two tokens are padding-style

    with torch.no_grad():
        out_serial = serial_gather_center_tokens_from_atomic(
            x_atomic=q_flat_real,
            token_center_atom_index=token_center_atom_index,
            token_mask=token_mask.bool(),
            multiplicity=multiplicity,
        )  # (BM, N_token_real, D)
    assert out_serial.abs().max().item() > 0, "serial output is trivially zero"

    # Unreferenced atom padding must not affect real-token outputs.
    [q_flat_padded] = pad_along_atom([q_flat_real], n_atom_pad)
    q_win = q_flat_padded.view(BM, K, W, D)
    [q_dt] = shard_along_window([q_win], mesh)

    # Padded center indices point to atom 0; masked outputs are excluded by unpadding.
    center_idx = token_center_atom_index.repeat_interleave(multiplicity, 0).unsqueeze(
        -1
    )  # (BM, N_token_real, 1)
    center_idx_dt = shard_along_token(pad_along_token([center_idx], n_pad=n_token_pad), mesh=mesh)[
        0
    ]

    tmask = token_mask.repeat_interleave(multiplicity, 0)  # (BM, N_token_real)
    tmask_dt = shard_along_token(pad_along_token([tmask], n_pad=n_token_pad), mesh=mesh)[0]

    out_dt = gather_center_tokens_from_atomic(
        q_windowed=q_dt,
        token_center_atom_idx=center_idx_dt,
        token_mask=tmask_dt,
    )

    assert_output_placement(out_dt, expected_placements=PL_SEQ, label=label)
    assert_replicate_values_against_rank0(out_dt)
    out_cp_real = unpad_along_token(gather_along_token([out_dt]), n_real=N_token_real)[0]
    gather_n_assert_parity_at_rank0(out_serial, out_cp_real, label=label, atol=ATOL)

    # Token-to-atom gather.
    label = (
        f"token_to_atom_gather/M={multiplicity}/"
        f"(N_tok={N_token_real}->{N_token_real + n_token_pad},"
        f"N_atom={N_atom_real}->{N_atom_padded},K={K})"
    )

    torch.manual_seed(99 + multiplicity)
    c_real = torch.randn(BM, N_token_real, D, device=device, dtype=torch.float32)

    # Atom indices reference real tokens, never token padding.
    tok_idx_real = torch.randint(0, N_token_real, (B, N_atom_real), device=device)

    pad_mask_real = torch.zeros(B, N_atom_real, device=device, dtype=torch.float32)
    if N_atom_real >= W:
        pad_mask_real[:, -W:] = 1.0

    tok_idx_full = tok_idx_real.repeat_interleave(multiplicity, 0)  # (BM, N_atom_real)
    pad_mask_full = pad_mask_real.repeat_interleave(multiplicity, 0)  # (BM, N_atom_real)

    with torch.no_grad():
        out_serial = c_real[
            torch.arange(BM, device=device).unsqueeze(1),
            tok_idx_full,
            :,
        ]  # (BM, N_atom_real, D)
        out_serial = out_serial * (1.0 - pad_mask_full.unsqueeze(-1))
    assert out_serial.abs().max().item() > 0, "serial output is trivially zero"

    c_dt = shard_along_token(pad_along_token([c_real], n_pad=n_token_pad), mesh=mesh)[0]
    # CP shards tokens, not B*M; c_real already includes multiplicity.

    [tok_idx_padded] = pad_along_atom([tok_idx_full], n_atom_pad)
    tok_idx_win = tok_idx_padded.view(BM, K, W)

    # Pad with mask=1 so padded atoms contribute nothing; pad_along_atom defaults to 0.
    pad_mask_padded = pad_dim(pad_mask_full, dim=1, pad_len=n_atom_pad, value=1.0)
    pad_mask_win = pad_mask_padded.view(BM, K, W)
    [tok_idx_dt, pad_mask_dt] = shard_along_window([tok_idx_win, pad_mask_win], mesh)

    c_to_q = distributed_gather(c_dt, tok_idx_dt, axis=1, are_ids_contiguous=True)
    c_to_q = c_to_q * (1.0 - pad_mask_dt.unsqueeze(-1).float())

    assert_output_placement(c_to_q, expected_placements=PL_SEQ, label=label)
    assert_replicate_values_against_rank0(c_to_q)
    c_to_q_full = gather_along_window([c_to_q])[0].reshape(BM, N_atom_padded, D)
    [c_to_q_real] = unpad_along_atom([c_to_q_full], n_real=N_atom_real)

    gather_n_assert_parity_at_rank0(out_serial, c_to_q_real, label=label, atol=ATOL)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dims", [TINY_SHAPE, REPRESENTATIVE_SHAPE])
@pytest.mark.parametrize("multiplicity", [1, 2])
def test_gather_center_tokens_parity(dims: dict[str, int], multiplicity: int) -> None:
    """Real (non-divisible) input length: serial(real) vs unpad(CP(pad(real)))."""
    spawn_cp_test(partial(_run_parity, dims=dims, multiplicity=multiplicity))
