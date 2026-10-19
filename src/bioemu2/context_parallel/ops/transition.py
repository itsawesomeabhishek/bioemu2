"""Canonical DTensor-aware BioMoa transition."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.ops.transition import Transition
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import LayerNormParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.tiled_params_replicated import tile_rows
from bioemu2.context_parallel.utils.dtensor import contiguous_stride


class DTensorAwareTransition(nn.Module):
    """Forward-only BioMoa transition on token or pair DTensors, computed per row tile."""

    def __init__(self, serial: Transition, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.use_norm = serial.use_norm
        if self.use_norm:
            self.norm = LayerNormParamsReplicated(serial.norm, device_mesh)
        self.fc1 = LinearParamsReplicated(serial.fc1, device_mesh)
        self.fc2 = LinearParamsReplicated(serial.fc2, device_mesh)
        self.fc3 = LinearParamsReplicated(serial.fc3, device_mesh)

    def forward(self, x: DTensor) -> DTensor:
        """Apply ``fc3(silu(fc1(norm(x))) * fc2(norm(x)))``, keeping the input placements. No gradient"""
        x_local = x.to_local()
        weight_1 = self.fc1.weight.to_local().to(x.dtype)
        weight_2 = self.fc2.weight.to_local().to(x.dtype)
        weight_3 = self.fc3.weight.to_local().to(x.dtype)
        bias_1 = None if self.fc1.bias is None else self.fc1.bias.to_local().to(x.dtype)
        bias_2 = None if self.fc2.bias is None else self.fc2.bias.to_local().to(x.dtype)
        bias_3 = None if self.fc3.bias is None else self.fc3.bias.to_local().to(x.dtype)
        if self.use_norm:
            norm = self.norm
            norm_weight = None if norm.weight is None else norm.weight.to_local().to(x.dtype)
            norm_bias = None if norm.bias is None else norm.bias.to_local().to(x.dtype)

        def transition(h: torch.Tensor) -> torch.Tensor:
            if self.use_norm:
                h = F.layer_norm(h, norm.normalized_shape, norm_weight, norm_bias, norm.eps)
            # SiLU and the gate run in place, so only two hidden-width tensors are live.
            gate = F.silu(F.linear(h, weight_1, bias_1), inplace=True)
            hidden = F.linear(h, weight_2, bias_2)
            hidden *= gate
            del gate, h
            return F.linear(hidden, weight_3, bias_3)

        # Per position: the normalized input, fc1, fc2 (gated in place), and fc3.
        tile = tile_rows(x_local, x_local.shape[-1] + 2 * weight_1.shape[0] + weight_3.shape[0])
        if tile >= x_local.shape[1]:
            # One tile covers the shard; writing into a preallocated output would add a copy.
            out = transition(x_local)
        else:
            out = x_local.new_empty((*x_local.shape[:-1], weight_3.shape[0]))
            for i in range(0, out.shape[1], tile):
                out[:, i : i + tile] = transition(x_local[:, i : i + tile])
        shape = torch.Size((*x.shape[:-1], weight_3.shape[0]))
        return DTensor.from_local(
            out, x.device_mesh, x.placements, shape=shape, stride=contiguous_stride(shape)
        )


__all__ = ["DTensorAwareTransition"]
