from functools import partial
from typing import Any

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops import distributed_scatter_reduce
from bioemu2.context_parallel.utils.dtensor import (
    PL_SEQ,
    shard_along_atom,
)
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    spawn_cp_test,
)

N_TOKENS = 64
N_ATOMS = 128
C = 32


def _serial_scatter_reduce(
    idx: torch.Tensor,
    src: torch.Tensor,
    output_size: int,
    reduce: str,
    mask: torch.Tensor | None = None,
    eps_for_mean: float = 0.0,
) -> torch.Tensor:
    """Serial reference: scatter idx from src into output along axis=1."""
    B, N_src, C_feat = src.shape
    out = torch.zeros(B, output_size, C_feat, device=src.device, dtype=src.dtype)
    count = torch.zeros(B, output_size, 1, device=src.device, dtype=src.dtype)

    for b in range(B):
        for i in range(N_src):
            if mask is not None and not mask[b, i]:
                continue
            t = idx[b, i].item()
            out[b, t] += src[b, i]
            count[b, t] += 1

    if reduce == "mean":
        if eps_for_mean > 0.0:
            denom = count + eps_for_mean
        else:
            denom = count.clamp(min=1)
        out = out / denom

    return out


def _run_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    reduce: str,
    seed: int,
    label: str,
    masked: bool = False,
    eps_for_mean: float = 0.0,
) -> None:
    torch.manual_seed(seed)
    idx = torch.arange(N_ATOMS, device=device).unsqueeze(0) % N_TOKENS  # (1, N_ATOMS)
    idx = idx.long()
    src = torch.randn(1, N_ATOMS, C, device=device, dtype=torch.float32)
    mask = (torch.rand(1, N_ATOMS, device=device) > 0.3).bool() if masked else None

    out_serial = _serial_scatter_reduce(
        idx, src, N_TOKENS, reduce, mask=mask, eps_for_mean=eps_for_mean
    )
    assert out_serial.abs().max().item() > 0, "serial output is trivially zero"

    # With dp=1, only cp_axis_0 shards the atom axis; cp_axis_1 replicates it.
    cp_size = mesh.size(1)  # cp_axis_0 dimension
    assert N_ATOMS % cp_size == 0

    if mask is None:
        [idx_dt, src_dt] = shard_along_atom([idx, src], mesh)
        mask_dt = None
    else:
        [idx_dt, src_dt, mask_dt] = shard_along_atom([idx, src, mask], mesh)

    output_size_per_rank = N_TOKENS // cp_size

    out_dt = distributed_scatter_reduce(
        output_size_per_rank=output_size_per_rank,
        axis=1,
        idx=idx_dt,
        src=src_dt,
        reduce=reduce,
        idx_mask=mask_dt,
        are_ids_contiguous=True,
        eps_for_mean=eps_for_mean,
    )

    label = f"scatter_reduce/{label}"
    assert_output_placement(out_dt, expected_placements=PL_SEQ, label=label)
    assert_replicate_values_against_rank0(out_dt)
    gather_n_assert_parity_at_rank0(out_serial, out_dt, label=label, atol=5e-5)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize(
    "reduce_mode, seed, label, extr_args",
    [
        ("sum", 1234, "sum", {}),
        ("mean", 2345, "mean", {}),
        ("mean", 3456, "mean_eps", {"eps_for_mean": 1e-6}),
        ("sum", 4567, "masked_sum", {"masked": True}),
    ],
)
def test_scatter_parity(reduce_mode: str, seed: int, label: str, extr_args: dict[str, Any]) -> None:
    spawn_cp_test(partial(_run_parity, reduce=reduce_mode, seed=seed, label=label, **extr_args))
