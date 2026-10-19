import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from bioemu2.biomoa.modules.diffusion import ReadoutSuperposition
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import LayerNormParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.utils import contiguous_stride


class DTensorReadoutSuperposition(nn.Module):
    """Context-parallel readout superposition of token features onto atoms."""

    def __init__(self, serial: ReadoutSuperposition, device_mesh: DeviceMesh) -> None:
        super().__init__()
        self.device_mesh = device_mesh
        self.s_to_hidden = LinearParamsReplicated(
            layer_local=serial.s_to_hidden, device_mesh=device_mesh
        )
        self.q_to_hidden = LinearParamsReplicated(
            layer_local=serial.q_to_hidden, device_mesh=device_mesh
        )
        self.to_out = nn.Sequential(
            LayerNormParamsReplicated(serial.to_out[0], device_mesh=device_mesh),
            LinearParamsReplicated(layer_local=serial.to_out[1], device_mesh=device_mesh),
        )

    def forward(
        self,
        a: DTensor,  # Shard(0), Shard(1)=tokens, Replicate(); (B*M, N_token, a_s)
        q: DTensor,  # Shard(0), Shard(1)=atoms, Replicate(); (B*M, N_atom, atom_s)
        tok_idx_flat_dt: DTensor,  # Shard(0), Shard(1)=atoms, Replicate(); (B*M, N_atom) long
    ) -> DTensor:  # Shard(0), Shard(1)=atoms, Replicate(); (B*M, N_atom, 3)
        mesh = self.device_mesh

        with torch.autocast("cuda", enabled=False):
            a_to_q = self.s_to_hidden(
                a.float()
            )  # Shard(0), Shard(1)=tokens, Replicate(); (B*M, N_token, hidden)
            a_to_q = a_to_q.redistribute(
                device_mesh=mesh, placements=(Shard(0), Replicate(), Replicate())
            )
            local_tok_idx = tok_idx_flat_dt.to_local()  # (B*M/dp, N_atom/cp0) long
            hidden = a_to_q.shape[-1]
            idx_exp = local_tok_idx.unsqueeze(-1).expand(-1, -1, hidden)
            local_a_to_q = a_to_q.to_local().gather(1, idx_exp)  # (B*M/dp, N_atom/cp0, hidden)

        global_shape = torch.Size((q.shape[0], q.shape[1], local_a_to_q.shape[-1]))
        a_to_q_dt = DTensor.from_local(
            local_a_to_q.contiguous(),
            device_mesh=mesh,
            placements=(Shard(0), Shard(1), Replicate()),
            shape=global_shape,
            stride=contiguous_stride(global_shape),
        )

        q_hidden = self.q_to_hidden(q) + a_to_q_dt  # (B*M, N_atom, hidden)
        return self.to_out(q_hidden)  # (B*M, N_atom, 3)
