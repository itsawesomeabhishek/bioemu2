"""Context-parallel triangle multiplication (incoming/outgoing) via ring BMM.

Serial counterpart: bioemu2.biomoa.ops.triangle_multiplication.
Ported from Fold-CP triangular_mult.py; adds cuEq fused prologue, BMM chunking, and tiled prologue/epilogue.
"""

from enum import Enum, auto

import torch
from torch.distributed.tensor import DTensor, Shard

from bioemu2.biomoa.utils.warnings import suppress_cuequivariance_jit_deprecation

try:
    with suppress_cuequivariance_jit_deprecation():
        from cuequivariance_ops.triton.utils import Precision as _CueqPrecision
        from cuequivariance_ops_torch.fused_layer_norm_torch import (
            layer_norm_transpose as _cueq_layer_norm_transpose,
        )
        from cuequivariance_ops_torch.gated_gemm_torch import (
            fused_sigmoid_gated_dual_gemm as _cueq_fused_sigmoid_gated_dual_gemm,
        )

    _HAS_CUEQ = True
except ImportError:
    _HAS_CUEQ = False

from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.utils import update_exhaustive_strides

_DEFAULT_BMM_CHUNK = 2048
_DEFAULT_TILE_SIZE = 256


def cueq_fused_prologue(
    x_local: torch.Tensor,
    mask_local: torch.Tensor,
    norm_in_weight: torch.Tensor,
    norm_in_bias: torch.Tensor,
    p_in_weight: torch.Tensor,
    g_in_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Fused {norm_in, p_in, g_in, sigmoid, mask} on a CP-local pair shard.

    Replaces the vanilla pytorch chain

    .. code-block:: python

        h = norm_in(x_local)
        proj = p_in(h) * g_in(h).sigmoid() * mask_local.unsqueeze(-1)

    with two cueq fused kernels, mirroring the allatom
    :class:`TriangleMultiplicationCueqTiled` projection step on a tile of
    shape ``[B, N/2, N/2, D]``. The output layout matches the
    ``transpose_out=True`` convention of ``fused_sigmoid_gated_dual_gemm``:
    the leading axis carries the doubled feature dim ``2*D``. Callers that
    need a ``[B, N/2, N/2, 2*D]`` layout should ``permute(1, 2, 3, 0)``.

    Returns
    -------
    torch.Tensor
        ``[2*D, B, N/2, N/2]`` masked-gated projection ready to be chunked
        into ``a_local`` and ``b_local`` halves for the ring BMM.
    """
    if not _HAS_CUEQ:
        raise RuntimeError(
            "The context-parallel cuEq fused prologue requires the optional CUDA dependencies. "
            "Install them with `uv sync --extra cuda`."
        )

    normed = _cueq_layer_norm_transpose(
        x_local,
        norm_in_weight,
        norm_in_bias,
        eps=eps,
        layout="bijd->bijd",
    )
    return _cueq_fused_sigmoid_gated_dual_gemm(
        normed,
        g_in_weight,
        p_in_weight,
        mask_local,
        transpose_out=True,
        precision=_CueqPrecision.IEEE.value,
    )


class _XposeArgs(Enum):
    lhs = auto()
    rhs = auto()


def _matmul_accumulate(out: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    """Add ``lhs @ rhs`` into ``out`` in place."""
    if out.dtype == lhs.dtype:
        out.add_(torch.matmul(lhs, rhs))
        return
    with torch.autocast(out.device.type, enabled=False):
        if out.is_cuda:
            out_3d = out.flatten(0, -3)  # A view: ``out`` slices a contiguous accumulator.
            torch.baddbmm(
                out_3d,
                lhs.flatten(0, -3),
                rhs.flatten(0, -3),
                out_dtype=out.dtype,
                out=out_3d,
            )
        else:
            out.add_(torch.matmul(lhs.to(out.dtype), rhs.to(out.dtype)))


def _distributed_bmm(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    comm: Ring2DComm,
    permute_lhs: tuple[int, ...] | None = None,
    permute_rhs: tuple[int, ...] | None = None,
    permute_out: tuple[int, ...] | None = None,
    xpose_args: _XposeArgs | None = None,
    bmm_chunk: int = _DEFAULT_BMM_CHUNK,
    copy_inputs: bool = True,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Perform distributed batch matrix multiplication using ring communication.

    This function implements a memory-efficient distributed batch matrix
    multiply operation across a 2D process grid using ring communication patterns.
    It computes the matrix multiplication of two tensors while minimizing memory usage
    through double buffering and overlapping computation with communication.

    The algorithm works by:
    1. Optionally permuting input tensors to desired layouts
    2. Setting up communication buffers based on transpose requirements
    3. Using ring communication to rotate tensor chunks across processes
    4. Performing overlapped computation and communication with double buffering
    5. Accumulating partial results to compute the final distributed bmm

    Communication Patterns
    ----------------------
    The function uses Ring2DComm to implement sophisticated communication patterns
    across a 2D process grid. Below are ASCII diagrams illustrating the key phases:

    **Phase 1: Initial 2D Grid Setup**

    For a 3x3 process grid, each process (i,j) initially owns tensor chunks:
    ```
    ┌─────┬─────┬─────┐
    │(0,0)│(0,1)│(0,2)│  ← Row 0
    ├─────┼─────┼─────┤
    │(1,0)│(1,1)│(1,2)│  ← Row 1
    ├─────┼─────┼─────┤
    │(2,0)│(2,1)│(2,2)│  ← Row 2
    └─────┴─────┴─────┘
      ↑     ↑     ↑
     Col 0 Col 1 Col 2
    ```

    **Phase 2: Transpose Communication (if xpose_args specified)**

    e.g., when xpose_args=_XposeArgs.rhs, RHS tensor is transposed across the 2D grid:
    ```
    Original RHS Ownership    After Transpose Communication
    ┌─────┬─────┬─────┐      ┌─────┬─────┬─────┐
    │ R00 │ R01 │ R02 │      │ R00 │ R10 │ R20 │
    ├─────┼─────┼─────┤  →   ├─────┼─────┼─────┤
    │ R10 │ R11 │ R12 │      │ R01 │ R11 │ R21 │
    ├─────┼─────┼─────┤      ├─────┼─────┼─────┤
    │ R20 │ R21 │ R22 │      │ R02 │ R12 │ R22 │
    └─────┴─────┴─────┘      └─────┴─────┴─────┘
    ```
    When xpose_args=_XposeArgs.lhs, LHS tensor is similarly transposed across the 2D grid

    **Phase 3: Initial Ring Setup**

    Row initialization (comm_row_init): Each row i shifts left by i positions
    ```
    Before Row Init              After Row Init
    ┌─────┬─────┬─────┐         ┌─────┬─────┬─────┐
    │ L00 │ L01 │ L02 │ ←shift 0│ L00 │ L01 │ L02 │
    ├─────┼─────┼─────┤         ├─────┼─────┼─────┤
    │ L10 │ L11 │ L12 │ ←shift 1│ L11 │ L12 │ L10 │
    ├─────┼─────┼─────┤         ├─────┼─────┼─────┤
    │ L20 │ L21 │ L22 │ ←shift 2│ L22 │ L20 │ L21 │
    └─────┴─────┴─────┘         └─────┴─────┴─────┘
    ```

    Column initialization (comm_col_init): Each column j shifts up by j positions
    ```
    Before Col Init              After Col Init
    ┌─────┬─────┬─────┐         ┌─────┬─────┬─────┐
    │ R00 │ R01 │ R02 │         │ R00 │ R11 │ R22 │
    ├─────┼─────┼─────┤  shift  ├─────┼─────┼─────┤
    │ R10 │ R11 │ R12 │   ↑     │ R10 │ R21 │ R02 │
    ├─────┼─────┼─────┤   0,1,2 ├─────┼─────┼─────┤
    │ R20 │ R21 │ R22 │         │ R20 │ R01 │ R12 │
    └─────┴─────┴─────┘         └─────┴─────┴─────┘
    ```

    **Phase 4: Ring Computation Loop**

    For each iteration k in range(grid_size):
    1. Compute partial matmul: out += matmul(lhs_chunk, rhs_chunk)
    2. Ring shift both tensors for next iteration

    Ring communication pattern (each step shifts by 1):
    ```
    Step 0 → Step 1 → Step 2 (back to original)

    LHS Row Shifts (left by 1):
    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
    │ L00 │ L01 │ L02 │ →  │ L01 │ L02 │ L00 │ →  │ L02 │ L00 │ L01 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ L11 │ L12 │ L10 │ →  │ L12 │ L10 │ L11 │ →  │ L10 │ L11 │ L12 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ L22 │ L20 │ L21 │ →  │ L20 │ L21 │ L22 │ →  │ L21 │ L22 │ L20 │
    └─────┴─────┴─────┘    └─────┴─────┴─────┘    └─────┴─────┴─────┘

    RHS Column Shifts (up by 1):
    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
    │ R00 │ R11 │ R22 │    │ R10 │ R21 │ R02 │    │ R20 │ R01 │ R12 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ R10 │ R21 │ R02 │ →  │ R20 │ R01 │ R12 │ →  │ R00 │ R11 │ R22 │
    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤    ├─────┼─────┼─────┤
    │ R20 │ R01 │ R12 │    │ R00 │ R11 │ R22 │    │ R10 │ R21 │ R02 │
    └─────┴─────┴─────┘    └─────┴─────┴─────┘    └─────┴─────┴─────┘
    ```

    **Double Buffering Strategy**

    The algorithm uses double buffering to overlap communication with computation:
    ```
    Time →  │ Compute │ Compute │ Compute │
            │ Buffer0 │ Buffer1 │ Buffer0 │
            │    ↓    │    ↓    │    ↓    │
    Comm →  │   Send  │   Send  │   Send  │
            │  Buffer1│ Buffer0 │ Buffer1 │
            │   Recv  │   Recv  │   Recv  │
            │ Buffer1 │ Buffer0 │ Buffer1 │
    ```

    This ensures that while one buffer is being used for computation, the other
    buffer is being prepared through communication for the next iteration.


    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side tensor for matrix multiplication.
        Typically has shape (B, ...) where B is batch dimension.
    rhs : torch.Tensor
        Right-hand side tensor for matrix multiplication.
        Must be compatible with lhs for matrix multiplication after permutations.
    comm : Ring2DComm
        Ring communication object configured for 2D process grid communication.
        Provides row and column communication groups for distributed computation.
    permute_lhs : tuple[int, ...] | None, optional
        Permutation indices to apply to lhs tensor before computation. Typically
        the permutation with group the batch-like axes into leading axes and reshape
        the last two axes into "N" and "K" dimensions (in the NMK notation)
        If None, no permutation is applied. Default is None.
    permute_rhs : tuple[int, ...] | None, optional
        Permutation indices to apply to rhs tensor before computation. Typically
        the permutation with group the batch-like axes into leading axes and reshape
        the last two axes into "K" and "M" dimensions (in the NMK notation)
        If None, no permutation is applied. Default is None.
    permute_out : tuple[int, ...] | None, optional
        Permutation indices to apply to output tensor after computation. Typically
        the permutation reverts the resulting permutation of the output matrix
        due to the permutation of the input lhs' and rhs' axes.
        If None, no permutation is applied. Default is None.
    xpose_args : _XposeArgs | None, optional
        Specifies which tensor requires transpose communication:
        - _XposeArgs.lhs: Transpose communication for left-hand side tensor
        - _XposeArgs.rhs: Transpose communication for right-hand side tensor
        - None: No transpose communication required
        Default is None.
    copy_inputs : bool, optional
        Clone ``lhs`` and ``rhs`` before the ring overwrites them. Pass False only
        when the caller hands over contiguous buffers it no longer needs.
    out_dtype : torch.dtype | None, optional
        Accumulator dtype.

    Returns
    -------
    torch.Tensor
        Result of the distributed batch matrix multiplication.
        Shape depends on input shapes and permutation arguments.

    Examples
    --------
    Typical usage in triangle multiplication:

    >>> # For outgoing triangle multiplication
    >>> result = _distributed_bmm(
    ...     lhs=tensor_a,
    ...     rhs=tensor_b,
    ...     comm=ring_comm,
    ...     permute_lhs=(0, 3, 1, 2),  # (B, n, k, D) -> (B, D, n, k)
    ...     permute_rhs=(0, 3, 2, 1),  # (B, m, k, D) -> (B, D, k, m)
    ...     permute_out=(0, 2, 3, 1),  # (B, D, n, m) -> (B, n, m, D)
    ...     xpose_args=_XposeArgs.rhs
    ... )
    """
    if permute_lhs is not None:
        lhs = lhs.permute(permute_lhs)
    # this enforces lhs and rhs to be a clone so that the in-place modification
    # does not affect the input tensor
    if copy_inputs:
        lhs = lhs.clone(memory_format=torch.contiguous_format)
    if permute_rhs is not None:
        rhs = rhs.permute(permute_rhs)
    if copy_inputs:
        rhs = rhs.clone(memory_format=torch.contiguous_format)

    if xpose_args == _XposeArgs.lhs:
        lhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(lhs)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    elif xpose_args == _XposeArgs.rhs:
        rhs_recv = comm.comm_2d_trans.enqueue_to_dispatch(rhs)
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
    elif xpose_args is None:
        lhs_recv = lhs
        lhs = torch.empty_like(lhs_recv)
        rhs_recv = rhs
        rhs = torch.empty_like(rhs_recv)
    else:
        raise ValueError(f"Invalid xpose_args: {xpose_args}")

    # post the comm_2d_trans.wait_until_finished() (or no wait if xpose_args is not None),
    # *_recv are the correct tensors to operate on
    i_ready = 0
    i_recv = i_ready ^ 1
    lhs_buffer = [lhs_recv, lhs]
    rhs_buffer = [rhs_recv, rhs]

    if xpose_args is not None:
        comm.comm_2d_trans.wait_until_finished()

    lhs_buffer[i_recv] = comm.comm_row_init.enqueue_to_dispatch(
        lhs_buffer[i_ready], lhs_buffer[i_recv]
    )
    rhs_buffer[i_recv] = comm.comm_col_init.enqueue_to_dispatch(
        rhs_buffer[i_ready], rhs_buffer[i_recv]
    )

    i_ready ^= 1
    i_recv ^= 1

    # Allocate the accumulator on the first ring step to avoid another full-shard buffer.
    out = None  # type: ignore[assignment]

    comm.comm_row_init.wait_until_finished()
    comm.comm_col_init.wait_until_finished()

    # Double buffering computation
    for k_step in range(comm.group_layout.shape[1]):
        lhs_ready = lhs_buffer[i_ready]
        rhs_ready = rhs_buffer[i_ready]
        if k_step < comm.group_layout.shape[1] - 1:
            lhs_buffer[i_recv] = comm.comm_row.enqueue_to_dispatch(lhs_ready, lhs_buffer[i_recv])
            rhs_buffer[i_recv] = comm.comm_col.enqueue_to_dispatch(rhs_ready, rhs_buffer[i_recv])
        if out is None:
            out = torch.zeros(
                (*lhs_ready.shape[:-1], rhs_ready.shape[-1]),
                dtype=out_dtype or lhs_ready.dtype,
                device=lhs_ready.device,
            )
        n_local_dim = lhs_ready.shape[-2]
        chunk = bmm_chunk if bmm_chunk > 0 else n_local_dim
        for cs in range(0, n_local_dim, chunk):
            ce = min(cs + chunk, n_local_dim)
            _matmul_accumulate(out[..., cs:ce, :], lhs_ready[..., cs:ce, :], rhs_ready)
        if k_step < comm.group_layout.shape[1] - 1:
            comm.comm_row.wait_until_finished()
            comm.comm_col.wait_until_finished()
            i_ready = i_ready ^ 1
            i_recv = i_recv ^ 1

    assert out is not None  # loop runs at least once (grid_size >= 1)
    if permute_out is not None:
        out = out.permute(permute_out)
    return out


class _Direction(Enum):
    Outgoing = auto()
    Incoming = auto()


class _TriangleMultiplicationImpl(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        x: DTensor,
        mask: DTensor,
        g: DTensor,
        comm: Ring2DComm,
        direction: _Direction,
    ) -> DTensor:
        """Forward pass of distributed triangle multiplication computation.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object for saving information needed in backward pass.
        x : DTensor
            Input tensor with shape (B, N_token1, N_token2, c_hidden * 2).
            Must be sharded on dimensions 1 and 2.
        mask : DTensor
            Mask tensor with shape (B, N_token1, N_token2) indicating valid positions.
            Must be sharded on dimensions 1 and 2.
        g : DTensor
            pre-sigmoid gate tensor with shape (B, N_token1, N_token2, c_hidden * 2) indicating valid positions.
            Must be sharded on dimensions 1 and 2.
        comm : Ring2DComm
            Ring communication object configured for the distributed computation.
        direction : _Direction
            Direction of the triangle multiplication, Outgoing or Incoming.

        Returns
        -------
        DTensor
            Output tensor with shape (B, N_token1, N_token2, c_hidden).
            Contains the distributed triangle multiplication result.
        """
        # Check if inputs are of type DTensor
        if not isinstance(x, DTensor):
            raise TypeError(f"Input 'x' must be of type DTensor. Got type {type(x)}.")
        if not isinstance(mask, DTensor):
            raise TypeError(f"Input 'mask' must be of type DTensor. Got type {type(mask)}.")
        if not isinstance(g, DTensor):
            raise TypeError(f"Input 'g' must be of type DTensor. Got type {type(g)}.")

        # Check if inputs have identical device mesh
        device_mesh_input = x.device_mesh
        if device_mesh_input != mask.device_mesh:
            raise ValueError(
                f"Input tensors 'x' and 'mask' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {mask.device_mesh}."
            )
        if device_mesh_input != g.device_mesh:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {g.device_mesh}."
            )

        # Check if inputs have identical placements
        placements_input = x.placements
        if placements_input != mask.placements:
            raise ValueError(
                f"Input tensors 'x' and 'mask' must have identical placements. "
                f"Got placements {placements_input} and {mask.placements}."
            )
        if placements_input != g.placements:
            raise ValueError(
                f"Input tensors 'x' and 'g' must have identical placements. "
                f"Got placements {placements_input} and {g.placements}."
            )
        if placements_input != (Shard(0), Shard(1), Shard(2)):
            # Triangle multiplication requires pair tensors sharded on both token axes.
            raise ValueError(
                f"Input tensor 'x's placements are not (Shard(0), Shard(1), Shard(2)). "
                f"Got placements {placements_input}."
            )

        # Check input shapes
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                f"Input tensor 'x' must have an even number of hidden dimension size. Got {x.shape[-1]}"
            )

        if x.ndim != 4:
            raise ValueError(f"Input tensor 'x' must have 4 dimensions. Got {x.ndim} dimensions.")

        if mask.ndim != 3:
            raise ValueError(
                f"Input tensor 'mask' must have 3 dimensions. Got {mask.ndim} dimensions."
            )

        if mask.shape != x.shape[:3]:
            raise ValueError(
                f"Input tensor 'mask' must have the same shape as the first 3 dimensions of 'x'. "
                f"Got mask shape: {mask.shape} vs x shape[:3]: {x.shape[:3]}"
            )
        if g.shape != x.shape:
            raise ValueError(
                f"Input tensor 'g' must have the same shape as 'x'. Got g shape: {g.shape} vs x shape: {x.shape}"
            )

        # Perform consistency check between the ring_comm and the device_mesh_input
        i_tensor_dim_to_i_grid_axis = [-1] * x.ndim
        for i_grid_axis, placement in enumerate(placements_input):
            if isinstance(placement, Shard):
                i_tensor_dim_to_i_grid_axis[placement.dim] = i_grid_axis
        if i_tensor_dim_to_i_grid_axis[1] == -1 or i_tensor_dim_to_i_grid_axis[2] == -1:
            raise ValueError(
                f"Input tensors' dimensions 1 and 2 must be sharded. Got placements {placements_input}."
            )

        # Check ring_comm consistency
        if comm.group_col != device_mesh_input.get_group(i_tensor_dim_to_i_grid_axis[1]):
            raise ValueError(
                "Input ring_comm's group_col process group is not the same as the group sharding the input tensors' axis 1"
            )

        coord_device_mesh_input = device_mesh_input.get_coordinate()
        if coord_device_mesh_input is None:
            raise ValueError(
                f"ring_comm.coord_2d {comm.coord_2d} is not on device_mesh_input {device_mesh_input}."
            )
        if comm.coord_2d != (
            coord_device_mesh_input[i_tensor_dim_to_i_grid_axis[1]],
            coord_device_mesh_input[i_tensor_dim_to_i_grid_axis[2]],
        ):
            raise ValueError(
                f"Input ring_comm's coord_2d {comm.coord_2d} does not match the "
                f"device mesh's rank coordinates {coord_device_mesh_input} for the sharded dimensions."
            )

        ctx.mark_non_differentiable(mask)

        # Apply mask and prepare for computation
        mask_local = mask.to_local().unsqueeze(-1)
        g_local = g.to_local().sigmoid()
        x_local = x.to_local() * mask_local
        x_local *= g_local

        # the _distributed_bmm will permute a_local and b_local and make
        # the resulting tensors contiguous so we don't need to clone them here
        a_local, b_local = torch.chunk(x_local, 2, dim=-1)

        # Store tensors for backward pass
        if x.requires_grad:
            # here x_local is masked and gated
            ctx.save_for_backward(a_local, b_local, mask_local, x_local, g_local)
            ctx.comm = comm
            ctx.shape_x_input = x.shape
            ctx.stride_x_input = x.stride()
            ctx.shape_g_input = g.shape
            ctx.stride_g_input = g.stride()
            ctx.placements_input = placements_input
            ctx.device_mesh_input = device_mesh_input
            ctx.direction = direction

        if direction == _Direction.Outgoing:
            permute_lhs = (0, 3, 1, 2)  # from (B, n, k, D) to (B, D, n, k)
            permute_rhs = (0, 3, 2, 1)  # from (B, m, k, D) to (B, D, k, m)
            permute_out = (0, 2, 3, 1)  # from (B, D, n, m) to (B, n, m, D)
            xpose_args = _XposeArgs.rhs
        elif direction == _Direction.Incoming:
            permute_lhs = (0, 3, 2, 1)  # from (B, k, n, D) to (B, D, n, k)
            permute_rhs = (0, 3, 1, 2)  # from (B, k, m, D) to (B, D, k, m)
            permute_out = (0, 2, 3, 1)  # from (B, D, n, m) to (B, n, m, D)
            xpose_args = _XposeArgs.lhs
        else:
            raise ValueError(f"Invalid direction: {direction}")

        out_local = _distributed_bmm(
            a_local,
            b_local,
            comm,
            permute_lhs=permute_lhs,
            permute_rhs=permute_rhs,
            permute_out=permute_out,
            xpose_args=xpose_args,
        ).contiguous()

        shape_output = x.shape[:-1] + (out_local.shape[-1],)
        stride_output = update_exhaustive_strides(x.shape, x.stride(), shape_output)
        # Convert back to DTensor
        out = DTensor.from_local(
            out_local,
            device_mesh=device_mesh_input,
            placements=placements_input,
            shape=shape_output,
            stride=stride_output,
        )
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, d_loss_d_out: DTensor) -> tuple[DTensor, None, DTensor, None, None]:
        """Backward pass of distributed triangle multiplication computation."""
        if not isinstance(d_loss_d_out, DTensor):
            raise TypeError(
                f"Input 'd_loss_d_out' must be of type DTensor. Got type {type(d_loss_d_out)}."
            )

        if d_loss_d_out.device_mesh != ctx.device_mesh_input:
            raise ValueError(
                f"Input 'd_loss_d_out' must have the same device mesh as the input tensors. "
                f"Got device meshes {d_loss_d_out.device_mesh} and {ctx.device_mesh_input}."
            )

        if d_loss_d_out.placements != ctx.placements_input:
            raise ValueError(
                f"Input 'd_loss_d_out' must have the same placements as the input tensors. "
                f"Got placements {d_loss_d_out.placements} and {ctx.placements_input}."
            )

        a, b, mask_local, x_masked_gated_local, g_local = ctx.saved_tensors
        comm = ctx.comm
        direction = ctx.direction

        # cast d_loss_d_out to the same dtype as a (saved tensor) to avoid type promotion to FP32
        # Note: torch.amp.custom_bwd disables autocast, so operations run in the input dtype.
        # If the upstream adjoint (d_loss_d_out) arrives as FP32 (e.g. from loss scaling or downstream FP32 layers),
        # mixed-precision ops with saved BF16 tensors would promote to FP32, causing potential communication
        # buffer mismatches and NCCL hangs. Explicit casting ensures consistent precision.
        d_loss_d_out_local = d_loss_d_out.to_local().to(dtype=a.dtype)

        if direction == _Direction.Outgoing:
            lhs_da = d_loss_d_out_local
            rhs_da = b
            permute_lhs_da = (0, 3, 1, 2)  # from (B, n, m, D) to (B, D, n, m)
            permute_rhs_da = (0, 3, 1, 2)  # from (B, m, k, D) to (B, D, m, k)
            permute_out_da = (0, 2, 3, 1)  # from (B, D, n, k) to (B, n, k, D)
            xpose_args_da = None

            lhs_db = d_loss_d_out_local
            rhs_db = a
            permute_lhs_db = (0, 3, 2, 1)  # from (B, n, m, D) to (B, D, m, n)
            permute_rhs_db = (0, 3, 1, 2)  # from (B, n, k, D) to (B, D, n, k)
            permute_out_db = (0, 2, 3, 1)  # from (B, D, m, k) to (B, m, k, D)
            xpose_args_db = _XposeArgs.lhs

        elif direction == _Direction.Incoming:
            lhs_da = b
            rhs_da = d_loss_d_out_local
            permute_lhs_da = (0, 3, 1, 2)  # from (B, k, m, D) to (B, D, k, m)
            permute_rhs_da = (0, 3, 2, 1)  # from (B, n, m, D) to (B, D, m, n)
            permute_out_da = (0, 2, 3, 1)  # from (B, D, k, n) to (B, k, n, D)
            xpose_args_da = _XposeArgs.rhs

            lhs_db = a
            rhs_db = d_loss_d_out_local
            permute_lhs_db = (0, 3, 1, 2)  # from (B, k, n, D) to (B, D, k, n)
            permute_rhs_db = (0, 3, 1, 2)  # from (B, n, m, D) to (B, D, n, m)
            permute_out_db = (0, 2, 3, 1)  # from (B, D, k, m) to (B, k, m, D)
            xpose_args_db = None
        else:
            raise ValueError(f"Invalid direction: {direction}")

        d_loss_d_a_local = _distributed_bmm(
            lhs_da,
            rhs_da,
            comm,
            permute_lhs=permute_lhs_da,
            permute_rhs=permute_rhs_da,
            permute_out=permute_out_da,
            xpose_args=xpose_args_da,
        ).contiguous()

        # Phase 2: d_loss_d_b
        d_loss_d_b_local = _distributed_bmm(
            lhs_db,
            rhs_db,
            comm,
            permute_lhs=permute_lhs_db,
            permute_rhs=permute_rhs_db,
            permute_out=permute_out_db,
            xpose_args=xpose_args_db,
        ).contiguous()

        # concatenate and apply mask to gradients
        dab_local = torch.cat([d_loss_d_a_local, d_loss_d_b_local], dim=-1)

        x_masked_gated_local *= 1 - g_local
        dg_local = dab_local * x_masked_gated_local

        dg = DTensor.from_local(
            dg_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.shape_g_input,
            stride=ctx.stride_g_input,
        )

        dx_local = dab_local
        dx_local *= mask_local
        dx_local *= g_local

        # Convert gradients back to DTensors
        dx = DTensor.from_local(
            dx_local,
            device_mesh=ctx.device_mesh_input,
            placements=ctx.placements_input,
            shape=ctx.shape_x_input,
            stride=ctx.stride_x_input,
        )

        return dx, None, dg, None, None
