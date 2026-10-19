# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Axis-wise (row/column) dropout helpers for pair and MSA representations."""

import torch


def get_axiswise_dropout_mask(
    dropout: float,
    z: torch.Tensor,
    training: bool,
    columnwise: bool = False,
) -> torch.Tensor:
    """Build an axiswise broadcast dropout mask for a rank-4 tensor.

    The mask is shared along one of the two middle axes so entire "rows" or
    "columns" of the tensor are dropped together — the row/column-wise
    dropout used by AlphaFold-style trunks on both pair tensors ``[B, N, N, D]``
    and MSA-shaped tensors ``[B, S, N, D]``. The returned mask is already
    scaled by ``1 / (1 - dropout)`` (inverted dropout), so callers can multiply
    without any eval-time correction.

    Parameters
    ----------
    dropout : float
        The dropout rate in [0, 1].
    z : torch.Tensor
        The tensor of shape ``[B, X, Y, D]`` this mask will be broadcast
        against. Only its shape and dtype/device are read.
    training : bool
        Whether the model is in training mode. When ``False`` the mask is
        forced to all ones.
    columnwise : bool, optional
        If ``True`` share the mask along the ``X`` axis (drop whole
        ``Y``-columns); otherwise share along the ``Y`` axis (drop whole
        ``X``-rows).

    Returns
    -------
    torch.Tensor
        Mask of shape ``[B, 1, Y, 1]`` when ``columnwise`` else ``[B, X, 1, 1]``,
        broadcastable to ``z``.
    """
    dropout = dropout * training
    # Slice out a singleton along the axis we want to broadcast over — this is
    # how the mask ends up shared across a whole row or column.
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    if dropout == 1.0:
        return torch.zeros_like(v)
    elif dropout == 0.0:
        d = torch.ones_like(v)
    else:
        d = torch.rand_like(v) > dropout
        d = d * 1.0 / (1.0 - dropout)
    return d


def apply_axiswise_dropout(
    dropout: float,
    x: torch.Tensor,
    training: bool,
    columnwise: bool = False,
) -> torch.Tensor:
    """Apply row/column-wise dropout to a rank-4 tensor ``x`` of shape ``[B, X, Y, D]``.

    Skips mask allocation when dropout is disabled (eval mode or rate 0).
    """
    if not training or dropout == 0.0:
        return x
    return get_axiswise_dropout_mask(dropout, x, training, columnwise=columnwise) * x
