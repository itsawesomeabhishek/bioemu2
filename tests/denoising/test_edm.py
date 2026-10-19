import pytest
import torch

from bioemu2.denoising.edm import edm_sample_schedule


def test_edm_sample_schedule_rejects_single_step() -> None:
    # num_steps=1 would divide by zero and yield NaN sigmas.
    with pytest.raises(ValueError, match="at least 2"):
        edm_sample_schedule(1, device=torch.device("cpu"))
    sigmas = edm_sample_schedule(2, device=torch.device("cpu"))
    assert torch.isfinite(sigmas).all() and sigmas[-1] == 0
