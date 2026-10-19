"""Forward-only, row-tiled replicated-parameter ops on token and pair DTensors.

Each op acts on every token or ``(i, j)`` pair independently, so each rank processes its
local shard in row tiles along dimension 1 without communication. The intermediates of one
tile hold at most ``TILE_ELEMENTS`` values, so small shards run as a single tile.

These layers skip the replicated-parameter gradient reduction of the untiled layers and
raise when autograd is enabled; use them for inference only.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.context_parallel.ops.distributed_primitives.layernorm import (
    LayerNormParamsReplicated,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.utils.dtensor import contiguous_stride

# Per-rank element budget for the intermediates of one row tile: 4 GiB in bfloat16 and
# 8 GiB in float32.
TILE_ELEMENTS = 2**31


def tile_rows(x: torch.Tensor, width: int) -> int:
    """Rows of ``x`` per tile when each position creates ``width`` intermediate values."""
    positions_per_row = math.prod((x.shape[0], *x.shape[2:-1]))
    return max(1, TILE_ELEMENTS // max(1, positions_per_row * width))


class LayerNormLinearParamsReplicatedTiled(nn.Sequential):
    """Forward-only concat, LayerNorm, and Linear on pair DTensors in one row-tiled pass.

    Parameters live in ``LayerNormParamsReplicated`` and ``LinearParamsReplicated``; only
    this forward runs.

    Fusing the three steps per tile means neither the concatenated pairs nor their
    normalization exist at full size. Parameters are cast to the input dtype and keep the
    ``0``/``1`` keys of the serial ``nn.Sequential``.
    """

    def __init__(self, serial: nn.Sequential, device_mesh: DeviceMesh) -> None:
        norm, linear = serial
        super().__init__(
            LayerNormParamsReplicated(norm, device_mesh),
            LinearParamsReplicated(linear, device_mesh),
        )

    def forward(self, *pairs: DTensor) -> DTensor:
        """Project pairs concatenated along channels, global shape ``(B, N_token, N_token, *)``.

        The output keeps the first input's placements.
        """
        if torch.is_grad_enabled():
            raise RuntimeError(f"{type(self).__name__} is forward-only; use torch.no_grad().")
        norm, linear = self
        dtype = pairs[0].dtype
        norm_weight = None if norm.weight is None else norm.weight.to_local().to(dtype)
        norm_bias = None if norm.bias is None else norm.bias.to_local().to(dtype)
        weight = linear.weight.to_local().to(dtype)
        bias = None if linear.bias is None else linear.bias.to_local().to(dtype)
        locals_ = [pair.to_local() for pair in pairs]
        out = locals_[0].new_empty((*locals_[0].shape[:-1], weight.shape[0]))
        tile = tile_rows(locals_[0], 2 * weight.shape[1] + weight.shape[0])
        for i in range(0, out.shape[1], tile):
            z = torch.cat([x[:, i : i + tile] for x in locals_], dim=-1)
            z = F.layer_norm(z, norm.normalized_shape, norm_weight, norm_bias, norm.eps)
            out[:, i : i + tile] = F.linear(z, weight, bias)
        shape = torch.Size((*pairs[0].shape[:-1], weight.shape[0]))
        return DTensor.from_local(
            out,
            pairs[0].device_mesh,
            pairs[0].placements,
            shape=shape,
            stride=contiguous_stride(shape),
        )
