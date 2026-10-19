"""Canonical context-parallel BioMoa embedders."""

import torch.nn as nn
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.modules.embedders import DistogramModule
from bioemu2.context_parallel.distributed.comm import TransposeComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.distributed_primitives.redistribute_transpose import (
    redistribute_transpose,
)
from bioemu2.context_parallel.utils import apply_pointwise


class DistributedDistogramWrap(nn.Module):
    """Distributed wrapper for the BioMoa distogram head."""

    def __init__(self, serial: DistogramModule, dist_manager: DistributedManager) -> None:
        super().__init__()
        self.distogram = serial.distogram
        self.transpose_comm = TransposeComm(
            dist_manager.group["cp"],
            dist_manager.layout_subgroups["cp"],
        )

    def forward(self, z: DTensor) -> DTensor:
        z_t = redistribute_transpose(z, self.transpose_comm, z.placements, 1, 2)
        return apply_pointwise(self.distogram, z + z_t)


__all__ = ["DistributedDistogramWrap"]
