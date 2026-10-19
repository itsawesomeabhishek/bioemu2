# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.
# Quaternion and rotation helpers adapted from PyTorch3D. See NOTICES.md and licenses/LICENSE.PYTORCH3D-BSD.

"""Random rigid-body augmentation used by diffusion samplers."""

import torch


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _random_quaternions(
    n: int,
    dtype: torch.dtype,
    device: torch.types.Device | None,
) -> torch.Tensor:
    quaternions = torch.randn((n, 4), dtype=dtype, device=device)
    squared_norm = (quaternions * quaternions).sum(1)
    return quaternions / _copysign(torch.sqrt(squared_norm), quaternions[:, 0])[:, None]


def _quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    assert quaternions.shape[-1] == 4
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    matrices = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return matrices.reshape(quaternions.shape[:-1] + (3, 3))


def compute_random_augmentation(
    multiplicity: int,
    s_trans: float = 1.0,
    device: torch.types.Device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random rotation matrices and translation vectors."""
    rotations = _quaternion_to_matrix(_random_quaternions(multiplicity, dtype, device))
    translations = torch.randn(
        (multiplicity, 1, 3),
        dtype=dtype,
        device=device,
    )
    return rotations, translations * s_trans


def augment_coordinates(
    coords: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Center, rotate, and translate coordinates using non-padding atoms."""
    mask = atom_mask[..., None]
    centered = coords - (coords * mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True)
    return torch.einsum("...md,...ds->...ms", centered, rotation) + translation
