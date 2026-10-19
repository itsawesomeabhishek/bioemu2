import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from bioemu2.denoising import cp_sync


@pytest.mark.parametrize(
    "device_type",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.gpu,
                pytest.mark.requires_4gpu,
            ],
        ),
    ],
)
def test_sampling_random_inputs_agree_despite_different_seeds(device_type: str):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--module",
            "tests.denoising.test_cp_sync",
            device_type,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=os.environ | {"OMP_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _worker(device_type: str) -> None:
    device = (
        torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if device_type == "cuda"
        else torch.device("cpu")
    )
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl" if device_type == "cuda" else "gloo")
    try:
        torch.manual_seed(42 + dist.get_rank())
        independent = torch.randn(2, 8, 3, device=device)
        independent_ranks = [torch.empty_like(independent) for _ in range(dist.get_world_size())]
        dist.all_gather(independent_ranks, independent)
        assert not torch.equal(independent_ranks[0], independent_ranks[1])

        noise = cp_sync.randn((2, 8, 3), device=device, dtype=torch.float32)
        rotation, translation = cp_sync.random_augmentation(2, device, torch.float32)
        ancestry = cp_sync.broadcast_from_rank0(torch.randperm(8, device=device))
        for value in (noise, rotation, translation, ancestry):
            ranks = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(ranks, value)
            for other in ranks:
                torch.testing.assert_close(value, other, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _worker(sys.argv[1])
