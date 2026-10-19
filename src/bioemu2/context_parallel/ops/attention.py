"""Canonical BioMoa CP attention wrappers; serial counterpart is biomoa attention_pair_bias.

They reuse the Fold-CP ring and shardwise autograd primitives with BioMoa's
fused QKV projections.
"""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor

from bioemu2.biomoa.ops.attention_pair_bias import (
    AttentionPairBias as SerialAttentionPairBias,
)
from bioemu2.biomoa.ops.attention_pair_bias import (
    AttentionPairBiasSlidingWindow as SerialAttentionPairBiasSlidingWindow,
)
from bioemu2.context_parallel.distributed.comm import AttentionPairBiasComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.ops.attention_impl import (
    _AttentionPairBiasContextVecParams,
    _AttentionPairBiasContexVecImpl,
    _AttentionPairBiasShardwiseImpl,
)
from bioemu2.context_parallel.ops.distributed_primitives.layernorm import LayerNormParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.ops.distributed_primitives.sigmoid_gate import sigmoid_gate
from bioemu2.context_parallel.ops.distributed_primitives.utils import (
    convert_single_repr_window_batched_query_to_key,
)


def _replicated_parameter(parameter: nn.Parameter, device_mesh: DeviceMesh) -> nn.Parameter:
    placements = [Replicate()] * device_mesh.ndim
    return nn.Parameter(distribute_tensor(parameter.detach(), device_mesh, placements))


def _windowed_atom_to_keys(x: DTensor, window_queries: int, window_keys: int) -> DTensor:
    if x.ndim == 3:
        return convert_single_repr_window_batched_query_to_key(
            x.unsqueeze(-1), W=window_queries, H=window_keys
        ).squeeze(-1)
    return convert_single_repr_window_batched_query_to_key(x, W=window_queries, H=window_keys)


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        attention: SerialAttentionPairBias,
        dist_manager: DistributedManager,
    ) -> None:
        super().__init__()
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.inf = attention.inf
        self.initial_norm = attention.initial_norm

        device_mesh = dist_manager.device_mesh_subgroups
        self.proj_qkv = LinearParamsReplicated(attention.proj_qkv, device_mesh=device_mesh)
        self.proj_q_bias = _replicated_parameter(attention.proj_q_bias, device_mesh)
        self.proj_g = LinearParamsReplicated(attention.proj_g, device_mesh=device_mesh)
        self.norm_z = LayerNormParamsReplicated(attention.norm_z, device_mesh=device_mesh)
        self.proj_z = LinearParamsReplicated(attention.proj_z, device_mesh=device_mesh)
        self.proj_o = LinearParamsReplicated(attention.proj_o, device_mesh=device_mesh)
        if self.initial_norm:
            self.norm_s = LayerNormParamsReplicated(attention.norm_s, device_mesh=device_mesh)

        self.ring_comm = AttentionPairBiasComm(
            dist_manager.group["cp"],
            dist_manager.layout_subgroups["cp"],
            dist_manager.subgroups["cp"][0],
            dist_manager.subgroups["cp"][1],
        )

    def forward(
        self,
        s: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, N, c_s)
        z: DTensor,  # Shard(0), Shard(1), Shard(2); (B, N, N, c_z)
        mask: DTensor,  # Shard(0), Shard(1), Replicate(); (B, N)
        multiplicity: int = 1,
    ) -> DTensor:  # Shard(0), Shard(1), Replicate(); (B*M, N, c_s)
        if self.initial_norm:
            s = self.norm_s(s)

        q, k, v = self.proj_qkv(s).chunk(3, dim=-1)
        q = q + self.proj_q_bias
        g = self.proj_g(s)
        z = self.proj_z(self.norm_z(z))

        context = _AttentionPairBiasContexVecImpl.apply(
            q,
            k,
            v,
            z,
            mask,
            None,
            _AttentionPairBiasContextVecParams(
                ring_comm=self.ring_comm,
                multiplicity=multiplicity,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                inf=self.inf,
                use_window_batching=False,
            ),
        )
        return self.proj_o(sigmoid_gate(x=context, g=g))


class AttentionPairBiasSlidingWindow(nn.Module):
    def __init__(
        self,
        attention: SerialAttentionPairBiasSlidingWindow,
        device_mesh: DeviceMesh,
    ) -> None:
        super().__init__()
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.inf = attention.inf
        self.initial_norm = attention.initial_norm

        self.proj_qkv = LinearParamsReplicated(attention.proj_qkv, device_mesh=device_mesh)
        self.proj_q_bias = _replicated_parameter(attention.proj_q_bias, device_mesh)
        self.proj_g = LinearParamsReplicated(attention.proj_g, device_mesh=device_mesh)
        self.norm_z = LayerNormParamsReplicated(attention.norm_z, device_mesh=device_mesh)
        self.proj_z = LinearParamsReplicated(attention.proj_z, device_mesh=device_mesh)
        self.proj_o = LinearParamsReplicated(attention.proj_o, device_mesh=device_mesh)
        if self.initial_norm:
            self.norm_s = LayerNormParamsReplicated(attention.norm_s, device_mesh=device_mesh)

    def forward(
        self,
        s: DTensor,  # Shard(0), Shard(1), Replicate(); (B*M, K, W, c_s)
        z: DTensor,  # Shard(0), Shard(1), Replicate(); (B, K, W, H, c_z)
        mask: DTensor,  # Shard(0), Shard(1), Replicate(); (B, K, W)
        multiplicity: int = 1,
    ) -> DTensor:  # Shard(0), Shard(1), Replicate(); (B*M, K, W, c_s)
        del multiplicity
        if self.initial_norm:
            s = self.norm_s(s)

        window_queries, window_keys = z.shape[2:4]
        q, kv = self.proj_qkv(s).split(
            [s.shape[-1], 2 * s.shape[-1]],
            dim=-1,
        )
        q = q.contiguous()
        kv = kv.contiguous()
        k, v = _windowed_atom_to_keys(kv, window_queries, window_keys).chunk(2, dim=-1)
        q = q + self.proj_q_bias
        g = self.proj_g(s)
        z = self.proj_z(self.norm_z(z))
        mask_keys = _windowed_atom_to_keys(mask, window_queries, window_keys)

        context = _AttentionPairBiasShardwiseImpl.apply(
            q,
            k,
            v,
            z,
            None,
            mask_keys,
            self.num_heads,
            self.head_dim,
            self.inf,
        )
        return self.proj_o(sigmoid_gate(x=context, g=g))
