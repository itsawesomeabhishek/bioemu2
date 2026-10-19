from dataclasses import replace

import torch

from bioemu2.biomoa.modules.diffusion import StaticContext
from bioemu2.context_parallel.modules.diffusion import CPStaticContext


def resample_conditioning(
    context: StaticContext | CPStaticContext, indices: torch.Tensor
) -> StaticContext | CPStaticContext:
    """Keep ITO conditioning aligned with resampled particle ancestry.

    Call only during ITO sampling, after augmentation has expanded conditioning
    coordinates to all particles. The samplers require one input structure
    (``B = 1``), so the flattened coordinate batch ``B * M`` equals ``M``.

    Args:
        context: Context with ``coords_cond`` shaped ``(M, N_atoms, 3)`` and
            ``dt`` and ``is_dynamics_conditioned`` shaped ``(1, 1)`` or ``(1, M)``.
            All three fields must be present. A ``True`` conditioning flag means
            the particle uses its conditioning coordinates and time lag.
        indices: Integer tensor of shape ``(M,)`` giving the source particle for
            each output slot, identical to the indices used to resample coordinates.

    Returns:
        A new context with per-particle conditioning reindexed. Singleton
        metadata and all other context fields remain unchanged.
    """
    coords_cond = context.coords_cond
    dt = context.dt
    is_dynamics_conditioned = context.is_dynamics_conditioned
    assert coords_cond is not None and dt is not None and is_dynamics_conditioned is not None
    return replace(
        context,
        coords_cond=coords_cond[indices],
        dt=dt[:, indices] if dt.shape[1] > 1 else dt,
        is_dynamics_conditioned=(
            is_dynamics_conditioned[:, indices]
            if is_dynamics_conditioned.shape[1] > 1
            else is_dynamics_conditioned
        ),
    )
