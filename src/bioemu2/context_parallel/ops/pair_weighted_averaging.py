"""Canonical context-parallel BioMoa pair-weighted averaging."""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.ops.pair_weighted_averaging import PairWeightedAveraging
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import LayerNormParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.pair_averaging import (
    Ring2DCommPairAveraging,
    _PairWeightedAveragingImpl,
)


class DistributedPairWeightedAveraging(torch.nn.Module):
    """Distributed wrapper for BioMoa pair-weighted averaging."""

    def __init__(
        self,
        serial: PairWeightedAveraging,
        device_mesh: DeviceMesh,
        comm: Ring2DCommPairAveraging,
    ) -> None:
        super().__init__()
        self.comm = comm
        self.c_h = serial.c_h
        self.num_heads = serial.num_heads
        self.inf = serial.inf
        self.norm_m = LayerNormParamsReplicated(serial.norm_m, device_mesh)
        self.norm_z = LayerNormParamsReplicated(serial.norm_z, device_mesh)
        self.proj_m = LinearParamsReplicated(serial.proj_m, device_mesh)
        self.proj_g = LinearParamsReplicated(serial.proj_g, device_mesh)
        self.proj_z = LinearParamsReplicated(serial.proj_z, device_mesh)
        self.proj_o = LinearParamsReplicated(serial.proj_o, device_mesh)

    def forward(
        self,
        m: DTensor,
        z: DTensor,
        mask: DTensor,
        chunk_heads: bool = False,
    ) -> DTensor:
        m = self.norm_m(m)
        z = self.norm_z(z)
        g = self.proj_g(m)
        v = self.proj_m(m)
        b = self.proj_z(z)
        if chunk_heads and not self.training:
            chunks = []
            for head in range(self.num_heads):
                start = head * self.c_h
                end = (head + 1) * self.c_h
                chunks.append(
                    _PairWeightedAveragingImpl.apply(
                        v[..., start:end],
                        b[..., head : head + 1],
                        mask,
                        g[..., start:end],
                        self.comm,
                        1,
                        self.inf,
                    )
                )
            output = torch.cat(chunks, dim=-1)
        else:
            output = _PairWeightedAveragingImpl.apply(
                v,
                b,
                mask,
                g,
                self.comm,
                self.num_heads,
                self.inf,
            )
        return self.proj_o(output)


__all__ = ["DistributedPairWeightedAveraging"]
