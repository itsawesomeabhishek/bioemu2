import pytest
import scipy
import torch

from bioemu2.denoising.rigid_align import masked_mean, weighted_rigid_align


@pytest.fixture(params=[43], ids=["seed=43"])
def seed(request):
    return request.param


@pytest.fixture(params=[10, 200], ids=["BS=10", "BS=200"])
def batch_size(request):
    return request.param


@pytest.fixture(params=[201], ids=["Atoms=201"])
def num_atoms(request):
    return request.param


def _rigid_align_gradient_loss(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    dim = rotation.shape[-1]
    rotation_weight = torch.arange(dim * dim, dtype=rotation.dtype, device=rotation.device).reshape(
        1, dim, dim
    )
    return rotation.mul(rotation_weight).sum() + translation.square().sum()


def test_rigid_align_fullmask(batch_size, num_atoms, seed):
    """
    Test the alignment with a full mask.
    This is effectively the Kabsch algorithm on all positions in the [BS, Atoms, 3] tensor.

    Parameters
    ----------
    batch_size: scalar
        The number of samples in the batch
    num_atoms: scalar
        The number of atoms in each sample
    seed: scalar
        The random seed for reproducibility
    """

    # Create a set of coordinates [BS, N, 3] with a global translation
    coord = torch.randn((batch_size, num_atoms, 3), generator=torch.Generator().manual_seed(seed))
    coord += torch.randn((batch_size, 1, 3), generator=torch.Generator().manual_seed(seed + 1))

    # Create full mask
    mask = torch.ones((batch_size, num_atoms), dtype=torch.float32)
    mask = mask.bool()

    # Create rotation matrices and translation vector
    rotvec = torch.randn((batch_size, 3), generator=torch.Generator().manual_seed(seed + 2))
    true_rot = torch.from_numpy(
        scipy.spatial.transform.Rotation.from_rotvec(rotvec.numpy()).as_matrix()
    ).float()
    trans = torch.randn((batch_size, 1, 3), generator=torch.Generator().manual_seed(seed + 3))

    # ref_coords: [Center -> Apply Rotation -> Translate] the original coordinates
    ref_coord = (
        torch.einsum("bij,bnj->bni", true_rot, coord - coord.mean(dim=-2, keepdim=True)) + trans
    )
    true_translation = -coord.mean(dim=-2) + ref_coord.mean(dim=-2)
    aligned_coord, rotation, translation = weighted_rigid_align(
        coord, ref_coord, torch.ones_like(mask.float()), mask
    )
    # Test for correct rotation, translation and alignment
    assert torch.allclose(true_rot, rotation, atol=1e-4)
    assert torch.allclose(translation.squeeze(), true_translation, atol=1e-4)
    assert torch.allclose(aligned_coord, ref_coord, atol=1e-4)


def test_rigid_align_partial_mask(batch_size, num_atoms, seed):
    """
    Test the alignment with a partial mask.
    Applies a mask to the coordinates, simulating a scenario where only a subset of atoms is considered for alignment.
    [BS, Atoms=[13, 27, 5, ...], 3]

    Parameters
    ----------
    batch_size: scalar
        The number of samples in the batch
    num_atoms: scalar
        The number of atoms in each sample
    seed: scalar
        The random seed for reproducibility
    """
    # Create initial coordinates[BS, N, 3] with a global translation
    coord = torch.randn((batch_size, num_atoms, 3), generator=torch.Generator().manual_seed(seed))
    coord += torch.randn((batch_size, 1, 3), generator=torch.Generator().manual_seed(seed + 1))

    # Create partial mask
    # For numerical stability, we use a mask with a minimum of 25 atom positions
    # This ensures that the covariance matrix is well-defined and the SVD can be computed without issues.
    mask_indices = torch.randint(
        25, num_atoms, (batch_size,), generator=torch.Generator().manual_seed(seed + 2)
    )
    mask = torch.zeros((batch_size, num_atoms), dtype=torch.float32)
    for i, idx in enumerate(mask_indices):
        mask[i, : idx + 1] = 1.0
    mask = mask.bool()

    # Create random weights ∈[0,1] for each atom in the batch[BS, Atoms]
    weights = torch.rand((batch_size, num_atoms), generator=torch.Generator().manual_seed(seed + 3))

    # Create a global rotation and translation for each sample[Atom, 3] in the batch[BS, Atom, 3]
    rotvec = torch.randn((batch_size, 3), generator=torch.Generator().manual_seed(seed + 4))
    true_rot = torch.from_numpy(
        scipy.spatial.transform.Rotation.from_rotvec(rotvec.numpy()).as_matrix()
    ).float()
    trans = torch.randn((batch_size, 1, 3), generator=torch.Generator().manual_seed(seed + 5))

    # Calculate the weighted mean of the coordinates while respecting the mask and weights
    weighted_masked_coord_mean = masked_mean(
        x=coord, mask=weights * mask.float(), dim=-2
    ).unsqueeze(1)  # [BS, 1, 3]

    # Apply SE(3): Center -> Rotate -> Translate
    ref_coord = torch.einsum("bij,bnj->bni", true_rot, coord - weighted_masked_coord_mean) + trans

    # Calculate the weighted mean of the target/reference coordinates while respecting the mask and weights
    weighted_masked_ref_coord_mean = masked_mean(
        x=ref_coord, mask=weights * mask.float(), dim=-2
    ).unsqueeze(1)  # [BS, 1, 3]

    # Calculate the true translation vector while respecting the mask
    true_translation = -weighted_masked_coord_mean + weighted_masked_ref_coord_mean

    # Apply weighted rigid align with particular mask and particular weightings
    aligned_coord, rotation, translation = weighted_rigid_align(coord, ref_coord, weights, mask)

    assert torch.allclose(true_rot, rotation, atol=1e-4)
    assert torch.allclose(translation, true_translation, atol=1e-4)
    assert torch.allclose(aligned_coord, ref_coord, atol=1e-4)


def test_rigid_align_svd_failure_fallback(monkeypatch, caplog):
    """
    On real GPU data, ``torch.linalg.svd`` with the ``gesvd`` driver can raise
    ``torch._C._LinAlgError`` (error code 3) for ill-conditioned / repeated
    singular value inputs. An uncaught exception inside the DDP training loop
    deadlocks every other rank on the next NCCL collective. Verify that
    ``weighted_rigid_align`` catches the error and returns a translation-only
    alignment so the training step can complete with matching collective shapes.

    Forcing this from CPU input is awkward, so we monkeypatch ``torch.linalg.svd``
    to raise unconditionally — the contract under test is the fallback path, not
    the conditions that trigger it.
    """
    batch_size, num_atoms, dim = 2, 32, 3
    coord = torch.randn((batch_size, num_atoms, dim))
    ref_coord = torch.randn((batch_size, num_atoms, dim))
    weights = torch.ones((batch_size, num_atoms))
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)

    def _raise(*args, **kwargs):
        raise torch._C._LinAlgError(
            "linalg.svd: (Batch element 0): The algorithm failed to converge "
            "because the input matrix is ill-conditioned or has too many "
            "repeated singular values (error code: 3)."
        )

    monkeypatch.setattr(torch.linalg, "svd", _raise)

    with caplog.at_level("WARNING", logger="bioemu2.denoising.rigid_align"):
        aligned_coord, rotation, translation = weighted_rigid_align(coord, ref_coord, weights, mask)

    # Fallback emitted the expected warning.
    assert any("SVD failed" in record.message for record in caplog.records)

    # Shapes are preserved so the caller's downstream tensor ops still match.
    assert aligned_coord.shape == coord.shape
    assert rotation.shape == (batch_size, dim, dim)
    assert translation.shape == (batch_size, 1, dim)

    # Identity rotation + centroid-difference translation.
    expected_rotation = torch.eye(dim).expand(batch_size, dim, dim)
    assert torch.allclose(rotation, expected_rotation)

    source_centroid = coord.mean(dim=1, keepdim=True)
    dest_centroid = ref_coord.mean(dim=1, keepdim=True)
    assert torch.allclose(translation, -source_centroid + dest_centroid, atol=1e-5)

    # Aligned coords are source coords with translation-only correction.
    expected_aligned = coord - source_centroid + dest_centroid
    assert torch.allclose(aligned_coord, expected_aligned, atol=1e-5)


def test_rigid_align_sample_mask_skips_inactive_rows():
    """Inactive samples are returned unchanged instead of being aligned to invalid targets."""
    batch_size, num_atoms, dim = 3, 16, 3
    coord = torch.randn((batch_size, num_atoms, dim), generator=torch.Generator().manual_seed(1))
    ref_coord = coord.clone()
    ref_coord[0] = coord[0] + torch.tensor([1.0, 2.0, 3.0])
    ref_coord[1] = 0.0
    ref_coord[2] = coord[2] + torch.tensor([-2.0, 0.5, 1.0])
    weights = torch.ones((batch_size, num_atoms))
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)
    sample_mask = torch.tensor([True, False, True])

    aligned_coord, rotation, translation = weighted_rigid_align(
        coord, ref_coord, weights, mask, sample_mask=sample_mask
    )

    torch.testing.assert_close(aligned_coord[0], ref_coord[0])
    torch.testing.assert_close(aligned_coord[1], coord[1])
    torch.testing.assert_close(aligned_coord[2], ref_coord[2])
    torch.testing.assert_close(rotation[1], torch.eye(dim))
    torch.testing.assert_close(translation[1], torch.zeros(1, dim))


def test_rigid_align_all_true_sample_mask_matches_no_sample_mask():
    """An all-true sample mask is equivalent to the unmasked batch path."""
    batch_size, num_atoms, dim = 3, 16, 3
    generator = torch.Generator().manual_seed(2)
    coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    ref_coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    weights = torch.rand((batch_size, num_atoms), generator=generator) + 0.1
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)
    sample_mask = torch.ones(batch_size, dtype=torch.bool)

    unmasked_outputs = weighted_rigid_align(coord, ref_coord, weights, mask)
    masked_outputs = weighted_rigid_align(coord, ref_coord, weights, mask, sample_mask=sample_mask)

    for unmasked, masked in zip(unmasked_outputs, masked_outputs):
        torch.testing.assert_close(masked, unmasked)


def test_rigid_align_all_true_sample_mask_preserves_gradients():
    """The sample-mask path keeps gradients identical when every sample is active."""
    batch_size, num_atoms, dim = 3, 16, 3
    generator = torch.Generator().manual_seed(3)
    coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    ref_coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    weights = torch.rand((batch_size, num_atoms), generator=generator) + 0.1
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)
    sample_mask = torch.ones(batch_size, dtype=torch.bool)

    def _input_grads(use_sample_mask: bool) -> tuple[torch.Tensor, torch.Tensor]:
        coord_ = coord.detach().clone().requires_grad_(True)
        ref_coord_ = ref_coord.detach().clone().requires_grad_(True)
        _, rotation, translation = weighted_rigid_align(
            coord_,
            ref_coord_,
            weights,
            mask,
            sample_mask=sample_mask if use_sample_mask else None,
        )
        # Aligned coordinates are intentionally detached; rotation and translation
        # cover the differentiable part of this helper's return values.
        _rigid_align_gradient_loss(rotation, translation).backward()
        assert coord_.grad is not None
        assert ref_coord_.grad is not None
        return coord_.grad, ref_coord_.grad

    unmasked_coord_grad, unmasked_ref_grad = _input_grads(use_sample_mask=False)
    masked_coord_grad, masked_ref_grad = _input_grads(use_sample_mask=True)

    torch.testing.assert_close(masked_coord_grad, unmasked_coord_grad)
    torch.testing.assert_close(masked_ref_grad, unmasked_ref_grad)


def test_rigid_align_partial_sample_mask_preserves_active_gradients():
    """Active rows keep the same gradients as aligning only that active subset."""
    batch_size, num_atoms, dim = 4, 16, 3
    generator = torch.Generator().manual_seed(4)
    coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    ref_coord = torch.randn((batch_size, num_atoms, dim), generator=generator)
    weights = torch.rand((batch_size, num_atoms), generator=generator) + 0.1
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)
    sample_mask = torch.tensor([True, False, True, False])

    def _sample_mask_grads() -> tuple[torch.Tensor, torch.Tensor]:
        coord_ = coord.detach().clone().requires_grad_(True)
        ref_coord_ = ref_coord.detach().clone().requires_grad_(True)
        _, rotation, translation = weighted_rigid_align(
            coord_, ref_coord_, weights, mask, sample_mask=sample_mask
        )
        _rigid_align_gradient_loss(rotation, translation).backward()
        assert coord_.grad is not None
        assert ref_coord_.grad is not None
        return coord_.grad, ref_coord_.grad

    def _active_reference_grads() -> tuple[torch.Tensor, torch.Tensor]:
        coord_ = coord[sample_mask].detach().clone().requires_grad_(True)
        ref_coord_ = ref_coord[sample_mask].detach().clone().requires_grad_(True)
        _, rotation, translation = weighted_rigid_align(
            coord_, ref_coord_, weights[sample_mask], mask[sample_mask]
        )
        _rigid_align_gradient_loss(rotation, translation).backward()
        assert coord_.grad is not None
        assert ref_coord_.grad is not None
        return coord_.grad, ref_coord_.grad

    masked_coord_grad, masked_ref_grad = _sample_mask_grads()
    active_coord_grad, active_ref_grad = _active_reference_grads()

    torch.testing.assert_close(masked_coord_grad[sample_mask], active_coord_grad)
    torch.testing.assert_close(masked_ref_grad[sample_mask], active_ref_grad)
    torch.testing.assert_close(
        masked_coord_grad[~sample_mask], torch.zeros_like(masked_coord_grad[~sample_mask])
    )
    torch.testing.assert_close(
        masked_ref_grad[~sample_mask], torch.zeros_like(masked_ref_grad[~sample_mask])
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 only affects CUDA cuBLAS matmul")
@pytest.mark.parametrize("allow_tf32", [False, True], ids=["tf32=off", "tf32=on"])
def test_rigid_align_tf32_orthogonality(allow_tf32):
    """Regression test: the returned rotation must stay orthogonal (and aligned coords must
    match the target) even when the global TF32 matmul flag is enabled.

    ``weighted_rigid_align`` still reduces the covariance matrix with a real matmul (cheap for
    ``num_atoms`` reductions, unlike the tiny 3x3 reductions ``matmul_transpose_no_tf32``
    protects), so some TF32 rounding noise unavoidably reaches ``aligned`` via the SVD inputs.
    A 200-seed sweep over this scenario found ``max|aligned - ref_coord|`` distributed as
    mean=1.83e-3, std=2.45e-4 with allow_tf32=True (vs. mean=1.63e-6 with it off); the atol below
    is set to mean + 4*std (~2.8e-3, rounded up) from that sweep so the assertion tracks the TF32
    noise floor instead of the fp32 one.
    """
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        device = "cuda"
        batch_size, num_atoms = 64, 201
        g = torch.Generator(device="cpu").manual_seed(0)
        coord = torch.randn((batch_size, num_atoms, 3), generator=g).to(device)

        # A random proper rotation per batch element via scipy, then apply + translate.
        rotvec = torch.randn((batch_size, 3), generator=g)
        true_rot = (
            torch.from_numpy(
                scipy.spatial.transform.Rotation.from_rotvec(rotvec.numpy()).as_matrix()
            )
            .float()
            .to(device)
        )
        trans = torch.randn((batch_size, 1, 3), generator=g).to(device)
        centered = coord - coord.mean(dim=-2, keepdim=True)
        ref_coord = torch.einsum("bij,bnj->bni", true_rot, centered) + trans

        mask = torch.ones((batch_size, num_atoms), device=device)
        with torch.autocast("cuda", enabled=False):
            aligned, R, _ = weighted_rigid_align(coord, ref_coord, torch.ones_like(mask), mask)

        # Orthogonality: ||R R^T - I||_F should be at the fp32-noise floor, not TF32-noise.
        I = torch.eye(3, device=device).expand_as(R)
        ortho_err = (torch.einsum("bij,bkj->bik", R, R) - I).norm(dim=(-2, -1))
        assert ortho_err.max().item() < 1e-4, f"max ortho err {ortho_err.max().item():.3e}"

        # Proper rotation.
        det = torch.linalg.det(R)
        assert (det - 1.0).abs().max().item() < 1e-4, f"max |det-1| {(det - 1).abs().max():.3e}"

        # Downstream invariant: aligned source ~ ref (this is what the loss actually consumes).
        # atol tracks the TF32 (vs. fp32) noise floor measured by the sweep referenced above;
        # tf32=off stays orders of magnitude below its tighter budget. rtol=0 makes this
        # equivalent to a plain `(aligned - ref_coord).abs() <= atol` check.
        atol = 3e-3 if allow_tf32 else 1e-4
        max_err = (aligned - ref_coord).abs().max().item()
        assert torch.allclose(aligned, ref_coord, atol=atol, rtol=0), (
            f"max |aligned - ref_coord| {max_err:.3e} (atol={atol:.1e})"
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn
