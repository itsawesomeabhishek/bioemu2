"""Canonical BioMoa outer-product mean tests."""

from functools import partial

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.biomoa.ops.outer_product_mean import OuterProductMeanVanilla
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.outer_product_mean import (
    DistributedOuterProductMean,
    _opm_ring_step_accum,
)
from bioemu2.context_parallel.utils.dtensor import PL_MSA, shard_along_msa
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    randomize_params,
    spawn_cp_test,
    spawn_cpu_cp_test,
)


def _outer_product_mean_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    device_manager: DistributedManager,
    dtype: torch.dtype = torch.float32,
) -> None:
    torch.manual_seed(1234)
    serial = OuterProductMeanVanilla(8, 32, 32).to(device).eval()
    randomize_params(serial)

    m = torch.randn(2, 4, 8, 8, device=device, dtype=torch.float32)
    mask = torch.ones(2, 4, 8, dtype=torch.bool, device=device)
    mask[:, 1, -3:] = False
    mask[:, 2, :2] = False
    mask[:, 3] = False
    row_count = mask.any(dim=-1).sum(dim=1).clamp(min=1).float()
    serial_fp32_output = serial(m, mask)

    distributed = (
        DistributedOuterProductMean(
            serial,
            device_manager.device_mesh_subgroups,
            Ring2DComm(
                device_manager.group["cp"],
                device_manager.subgroups["cp"][0],
                device_manager.layout_subgroups["cp"],
            ),
        )
        .to(device)
        .eval()
    )
    m = m.to(dtype=dtype)
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        m_dt, mask_dt = shard_along_msa([m, mask], mesh, src_data_rank=None)
        distributed_output = distributed(m_dt, mask_dt, row_count)

    assert_output_placement(
        distributed_output,
        expected_placements=PL_MSA,
        label="BioMoaOuterProductMean",
    )
    assert_replicate_values_against_rank0(distributed_output)
    gather_n_assert_parity_at_rank0(
        serial_fp32_output,
        distributed_output,
        label="BioMoaOuterProductMean",
        atol=3e-2 if dtype == torch.bfloat16 else 5e-5,
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_outer_product_mean_parity(dtype: torch.dtype) -> None:
    spawn_cp_test(partial(_outer_product_mean_parity, dtype=dtype))


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_outer_product_mean_parity_cpu(dtype: torch.dtype) -> None:
    spawn_cpu_cp_test(partial(_outer_product_mean_parity, dtype=dtype))


@pytest.mark.gpu
@pytest.mark.parametrize("msa_depth", [19, 33])  # one and two _OPM_BLOCK_S passes
@pytest.mark.parametrize(("hidden_dim", "output_dim"), [(8, 5), (32, 33), (32, 256)])
@pytest.mark.parametrize(
    ("dtype", "precision"),
    [(torch.float32, "ieee"), (torch.bfloat16, "ieee")],  # we don't test unused combo
)
def test_opm_ring_step_odd_shapes(
    gpu_device: torch.device,
    msa_depth: int,
    hidden_dim: int,
    output_dim: int,
    dtype: torch.dtype,
    precision: str,
) -> None:
    """``(32, 256)`` is the BioMoa OPM size; with ``msa_depth <= _OPM_BLOCK_S`` the kernel
    must still fit in shared memory."""
    torch.manual_seed(123)
    a = torch.randn(2, msa_depth, 17, hidden_dim, device=gpu_device, dtype=dtype)
    b = torch.randn_like(a)
    weight = torch.randn(output_dim, hidden_dim**2, device=gpu_device, dtype=dtype)
    output = torch.randn(2, 17, 17, output_dim, device=gpu_device)
    float64_reference = output.double() + torch.nn.functional.linear(
        torch.einsum("bsic,bsjd->bijcd", a.double(), b.double()).flatten(-2),
        weight.double(),
    )
    _opm_ring_step_accum(a, b, output, weight, tile_size=7, matmul_precision=precision)
    relative_error = (
        output.double() - float64_reference
    ).abs().max() / float64_reference.abs().max()
    assert relative_error.item() < 1e-5
