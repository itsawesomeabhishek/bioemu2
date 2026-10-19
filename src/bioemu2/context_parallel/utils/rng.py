import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


def broadcast_across_cp1(tensor: torch.Tensor, cp1_group: dist.ProcessGroup) -> torch.Tensor:
    """Broadcast a tensor from rank 0 of the cp_axis_1 subgroup to all peers.

    Used for dropout masks and other random values that must be identical across
    cp_axis_1 ranks (because they broadcast over a dimension sharded on cp_axis_1).
    """
    dist.broadcast(tensor, src=dist.get_global_rank(cp1_group, 0), group=cp1_group)
    return tensor


@contextmanager
def preserve_rng_state():
    """Snapshot and restore all RNG states around a block.

    Used for rank-0 cache warming: the cache-miss path (e.g. RDKit conformer
    generation) may draw from RNGs, desyncing rank 0 from ranks that hit the
    warm cache and breaking the "all ranks produce identical coordinates"
    invariant required for correct collective communication.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
