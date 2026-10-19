# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Dependency-light context-parallelization utilities.

Keep this package independent of model packages so low-level CP helpers remain reusable.
"""

from bioemu2.context_parallel.utils.dtensor import (
    apply_pointwise,
    contiguous_stride,
    gather_along_atom,
    gather_along_token,
    pair_dtensor_from_local,
    shard_along_atom,
    shard_along_token,
    shard_along_window,
)
from bioemu2.context_parallel.utils.mesh import (
    get_row_column_for_tile_z_ops,
    pad_amount,
    pair_placements,
    rank_pair_ranges,
    shard_multiple,
)
from bioemu2.context_parallel.utils.padding import (
    CPPadSizes,
    make_cp_pad_sizes,
    pad_along_atom,
    pad_along_token,
    pad_along_window,
    unpad_along_atom,
    unpad_along_msa,
    unpad_along_token,
)
from bioemu2.context_parallel.utils.rng import (
    broadcast_across_cp1,
    preserve_rng_state,
)
from bioemu2.context_parallel.utils.session import (
    TRACE_LEVEL,
    barrier,
    cp_session,
    cuda_gc,
    init_cp,
    log_cp_memory,
    teardown_cp_state,
)

__all__ = [
    "apply_pointwise",
    "barrier",
    "broadcast_across_cp1",
    "contiguous_stride",
    "cp_session",
    "cuda_gc",
    "get_row_column_for_tile_z_ops",
    "init_cp",
    "pad_amount",
    "pair_dtensor_from_local",
    "pair_placements",
    "preserve_rng_state",
    "rank_pair_ranges",
    "shard_multiple",
    "teardown_cp_state",
    "shard_along_atom",
    "shard_along_token",
    "shard_along_window",
    "gather_along_token",
    "gather_along_atom",
    "pad_along_atom",
    "pad_along_token",
    "unpad_along_atom",
    "unpad_along_token",
    "unpad_along_msa",
    "unpad_along_window",
    "CPPadSizes",
    "log_cp_memory",
    "make_cp_pad_sizes",
    "TRACE_LEVEL",
    "pad_along_window",
]
