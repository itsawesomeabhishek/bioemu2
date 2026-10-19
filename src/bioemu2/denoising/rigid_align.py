# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Weighted rigid (Kabsch) alignment of coordinate sets and helpers."""

import logging

import torch
from einops import einsum

logger = logging.getLogger(__name__)


def matmul_transpose_no_tf32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute ``a @ b.mT`` via broadcast + sum, so the reduction never touches cuBLAS.

    Equivalent to ``torch.einsum("... i k, ... j k -> ... i j", a, b)`` but immune to
    the global TF32 flag (``torch.backends.cuda.matmul.allow_tf32``): matmul/einsum
    lower to cuBLAS gemm which honors TF32, whereas elementwise-multiply + reduction
    always runs in fp32. Only worth doing for tiny inner dims (here, 3).

    Parameters
    ----------
    a: torch.Tensor[..., i, k]
        Left-hand side tensor.
    b: torch.Tensor[..., j, k]
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor[..., i, j]
        The result of ``a @ b.mT``.
    """
    return (a[..., :, None, :] * b[..., None, :, :]).sum(dim=-1)


def masked_mean(
    x: torch.Tensor,
    dim: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the mean of a tensor along a specified dimension, applying a mask.

    Parameters
    ----------
    x: torch.Tensor
        The input tensor.
    mask: torch.Tensor
        A boolean mask tensor of the same shape as `x`.
    dim: int, optional
        The dimension along which to compute the mean. Default is -1 (last dimension).

    Returns
    -------
    torch.Tensor
        The masked mean of the input tensor.
    """
    # Check that mask[BS, N] and x[BS, N, ...] match in [BS, N] dimensions
    assert mask.shape == x.shape[: mask.dim()], (
        f"mask shape {mask.shape} is not broadcastable to x shape {x.shape} along dim {dim}"
    )

    mask_ = mask.float().unsqueeze(-1)  # [BS, N] -> [BS, N 1]
    # Weight contribution in x with mask
    masked_x = x * mask_
    # Sum weighted x and normalize by weighted mask
    return masked_x.sum(dim=dim) / mask_.sum(dim=dim).clamp(min=1e-8)


def weighted_rigid_align(
    source_coords: torch.Tensor,
    dest_coords: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute weighted alignment.

    Parameters
    ----------
    source_coords: torch.Tensor[BS, N, 3]
        The atom coordinates to be aligned
    dest_coords: torch.Tensor[BS, N, 3]
        The alignment target coordinates
    weights: torch.Tensor[BS, N]
        The weights for alignment
    mask: torch.Tensor[BS, N]
        The atoms mask
    sample_mask: torch.Tensor[BS] | None
        Optional per-sample mask. Rows marked False are returned unaligned, with identity rotation
        and zero translation.

    Returns
    -------
    Aligned source_coords on dest_coords: torch.Tensor[BS, N, 3]
    Rotation matrices: torch.tensor[BS, 3, 3]
    Translation vector: torch.Tensor[BS, 3]


    Notes: This function always operates in float32 context. autocast should be disabled by caller.
    For example:
        with torch.autocast("cuda", enabled=False):
            aligned_coords, rot_matrix, translation = weighted_rigid_align(
                source_coords.float(),
                dest_coords.float(),
                weights=alignment_weights.float(),
                mask=atom_mask.float(),
            )
    """

    batch_size, num_points, dim = source_coords.shape

    if sample_mask is not None:
        sample_mask = sample_mask.reshape(-1).bool()
        assert sample_mask.shape == (batch_size,), (
            f"sample_mask shape {sample_mask.shape} does not match batch size {batch_size}."
        )
        aligned_coords = source_coords.detach().clone()
        rot_matrix = (
            torch.eye(dim, dtype=source_coords.dtype, device=source_coords.device)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        translation = torch.zeros(
            (batch_size, 1, dim), dtype=source_coords.dtype, device=source_coords.device
        )
        active_aligned, active_rot, active_translation = weighted_rigid_align(
            source_coords[sample_mask],
            dest_coords[sample_mask],
            weights[sample_mask],
            mask[sample_mask],
        )
        aligned_coords[sample_mask] = active_aligned
        rot_matrix[sample_mask] = active_rot
        translation[sample_mask] = active_translation
        return aligned_coords, rot_matrix, translation

    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    source_centroid = (source_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    dest_centroid = (dest_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    assert dest_centroid.dtype == torch.float32 and source_centroid.dtype == torch.float32

    # Center the coordinates
    source_coords_centered = source_coords - source_centroid
    dest_coords_centered = dest_coords - dest_centroid

    if num_points < (dim + 1):
        logger.warning(
            "weighted_rigid_align: point cloud size <= dim+1; cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix, still use matmul since reduction is over `n`
    cov_matrix = einsum(
        weights * dest_coords_centered, source_coords_centered, "b n i, b n j -> b i j"
    )

    # ``gesvd`` raises ``_LinAlgError`` on ill-conditioned inputs; catch so a single rank
    # cannot deadlock DDP. Fall back to translation-only alignment for the whole micro-batch.
    try:
        U, S, Vh = torch.linalg.svd(cov_matrix, driver="gesvd" if cov_matrix.is_cuda else None)
    except torch._C._LinAlgError as e:
        logger.warning(
            "weighted_rigid_align: SVD failed (%s); falling back to translation-only "
            "alignment for this micro-batch (batch_size=%d).",
            e,
            batch_size,
        )
        rot_matrix = (
            torch.eye(dim, dtype=torch.float32, device=cov_matrix.device)
            .unsqueeze(0)
            .expand(batch_size, dim, dim)
        )
        translation = -source_centroid + dest_centroid
        aligned_coords = (source_coords_centered + dest_centroid).detach()
        return aligned_coords, rot_matrix, translation

    V = Vh.mH  # torch.linalg.svd returns conjugate transpose of left singular vectors V

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        logger.warning(
            "weighted_rigid_align: excessively low rank of cross-correlation between aligned "
            "point clouds; cannot return a unique rotation."
        )

    # R = U V^T; broadcast+sum keeps R orthogonal even if global TF32 is on (~3e-3 -> ~1e-6).
    rot_matrix = matmul_transpose_no_tf32(U, V)

    # Enforce det(R) = +1 via a rank-1 update: flips u_3 v_3^T when det is negative; sign(det)
    # (not raw det) keeps the correction exactly +/-1 so it cannot rescale the last singular axis.
    sign_correction = torch.sign(torch.det(rot_matrix)) - 1.0  # [B]; 0 or -2
    u3, v3 = U[:, :, -1], V[:, :, -1]  # [B, 3] each
    rot_matrix = rot_matrix + sign_correction[:, None, None] * (u3[:, :, None] * v3[:, None, :])

    translation = -source_centroid + dest_centroid

    # Apply rotation: aligned[b,n,i] = sum_j R[b,i,j] * source[b,n,j] = (source @ R^T)[b,n,i].
    aligned_coords = matmul_transpose_no_tf32(source_coords_centered, rot_matrix) + dest_centroid
    aligned_coords.detach_()

    return aligned_coords, rot_matrix, translation
