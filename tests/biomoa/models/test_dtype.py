import pytest
import torch

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.batch import batch_collate


@pytest.mark.gpu
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ],
    ids=[
        "float32",
        "bfloat16",
        "float16",
    ],
)
def test_biomoa_dtype_behaviour(
    test_structure: AllAtomStructure,
    toy_biomoa_model: BioMoaCModel,
    gpu_device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_bf16_supported() and dtype == torch.bfloat16:
        pytest.skip("bfloat16 is not supported on this CUDA device.")

    torch.manual_seed(42)
    batch = batch_collate([test_structure]).to(gpu_device)
    sigmas = torch.full((batch.coords.shape[0],), 0.1, device=gpu_device)

    model = toy_biomoa_model.to(gpu_device).eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            predictions = model(batch, sigmas.to(dtype))
            score = predictions.score

    assert score.dtype == torch.float32, f"Expected score dtype {torch.float32}, got {score.dtype}"
    assert predictions.distogram.dtype == dtype, (
        f"Expected distogram dtype {dtype}, got {predictions.distogram.dtype}"
    )
    assert predictions.denoised_coords.dtype == torch.float32, (
        f"Expected denoised_coords dtype {torch.float32}, got {predictions.denoised_coords.dtype}"
    )
    assert score.shape == batch.coords.shape
    assert not torch.isnan(score).any(), f"Score contains NaN values for dtype {dtype}"
    assert not torch.isinf(score).any(), f"Score contains Inf values for dtype {dtype}"
    assert torch.isfinite(predictions.distogram).all()
    assert torch.isfinite(predictions.denoised_coords).all()

    if dtype == torch.float32:
        return

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            score_f32 = model(batch, sigmas.float()).score

    assert score_f32.shape == score.shape, "Output shapes do not match between dtypes."
    rtol = 1e-2
    # BF16 drift varies across GPU architectures and CUDA versions.
    atol = 0.03 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(score_f32, score.float(), rtol=rtol, atol=atol)
