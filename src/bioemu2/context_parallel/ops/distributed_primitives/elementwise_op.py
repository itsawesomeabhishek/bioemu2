# Vendored from NVIDIA Fold-CP (MIT): https://github.com/NVIDIA-BioNeMo/boltz-cp @ f76f37a77c854b56b6250b426af8c2d63b501f7f
# Source: src/boltz/distributed/model/layers/elementwise_op.py

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from enum import Enum, auto

import torch
from torch.distributed.tensor import DTensor, Partial, Shard


class ElementwiseOp(Enum):
    """Enumeration of supported elementwise operations."""

    # n-ary ops
    SUM = auto()
    SUB = auto()
    PROD = auto()
    DIV = auto()
    EQUAL = auto()
    BITAND = auto()


class _ElementwiseOpImpl(torch.autograd.Function):
    """Distributed implementation of elementwise operations using DTensors.

    This autograd function implements distributed elementwise operations that can perform
    summation, multiplication, division, and logical operations between two input tensors. The operations are
    performed element-wise across distributed tensors while maintaining proper gradient
    computation.

    Supported operations:
    - SUM: output = a + b
    - SUB: output = a - b
    - PROD: output = a * b
    - DIV: output = a / b
    - EQUAL: output = a & b
    - BITAND: output = a & b

    Key features:
    - Distributed computation across device meshes with various sharding strategies
    - Memory-efficient implementation that operates on local tensor chunks
    - Supports gradient computation through custom backward pass
    - Validates tensor compatibility (device mesh, placements, shapes)

    Notes
    -----
    Input tensors must be DTensors with:
    - Identical device mesh and placements
    - Compatible shapes (a and b must have the same shape)
    - No Partial placements (not currently supported)
    """

    @staticmethod
    def forward(ctx, a: DTensor, b: DTensor, op: ElementwiseOp) -> DTensor:
        """Forward pass of distributed elementwise operation.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object for saving information needed in backward pass.
        a : DTensor
            First input tensor. Can have any shape and sharding strategy.
        b : DTensor
            Second input tensor. Must have identical shape, device mesh,
            and placements as a.
        op : ElementwiseOp
            The operation to perform (SUM, PROD, DIV, EQUAL, or BITAND).

        Returns
        -------
        DTensor
            Output tensor with shape identical to input tensors.
            Contains the result of the specified operation.

        Raises
        ------
        TypeError
            If inputs are not DTensors.
        ValueError
            If tensors have incompatible device meshes, placements, or if
            Partial placements are used (not supported), or if op is invalid.
        """
        if not isinstance(a, DTensor):
            raise TypeError(f"Input 'a' must be of type DTensor. Got type {type(a)}.")
        if not isinstance(b, DTensor):
            raise TypeError(f"Input 'b' must be of type DTensor. Got type {type(b)}.")
        if not isinstance(op, ElementwiseOp):
            raise TypeError(f"Input 'op' must be of type ElementwiseOp. Got type {type(op)}.")

        device_mesh_input = a.device_mesh
        if b.device_mesh != device_mesh_input:
            raise ValueError(
                f"Input tensors 'a' and 'b' must have identical device mesh. "
                f"Got device meshes {device_mesh_input} and {b.device_mesh}."
            )

        placements_input = a.placements
        for i_dim_device_mesh, placement in enumerate(placements_input):
            if isinstance(placement, Partial):
                raise ValueError("Partial placements are not supported")
            elif isinstance(placement, Shard):
                if a.shape[placement.dim] % device_mesh_input.shape[i_dim_device_mesh] != 0:
                    raise ValueError(
                        f"Uneven sharding tensor dimension {placement.dim} of size {a.shape[placement.dim]} "
                        f"along device mesh dimension {i_dim_device_mesh} of size {device_mesh_input.shape[i_dim_device_mesh]} is not supported"
                    )

        if b.placements != placements_input:
            raise ValueError(
                f"Input tensors 'a' and 'b' must have identical placements. "
                f"Got placements {placements_input} and {b.placements}."
            )

        input_shape = a.shape
        if input_shape != b.shape:
            raise ValueError(
                f"Input tensors 'a' and 'b' must have identical shapes. Got shapes {input_shape} and {b.shape}."
            )

        a_local = a.to_local()
        b_local = b.to_local()

        # Perform the operation
        if op == ElementwiseOp.SUM:
            output_local = a_local + b_local
        elif op == ElementwiseOp.SUB:
            output_local = a_local - b_local
        elif op == ElementwiseOp.PROD:
            output_local = a_local * b_local
            a_local_copy = a_local.detach().clone() if b.requires_grad else None
            b_local_copy = b_local.detach().clone() if a.requires_grad else None
            ctx.save_for_backward(a_local_copy, b_local_copy)
        elif op == ElementwiseOp.DIV:
            output_local = a_local / b_local
            # Save tensors for backward pass:
            # - if a.requires_grad, we need b for: da = grad_output / b
            # - if b.requires_grad, we need b and output for: db = -grad_output * output / b
            b_local_copy = b_local.detach().clone() if a.requires_grad else None
            output_over_b_local = output_local / b_local if b.requires_grad else None
            ctx.save_for_backward(b_local_copy, output_over_b_local)
        elif op == ElementwiseOp.EQUAL:
            if a.requires_grad or b.requires_grad:
                raise ValueError("EQUAL operation is not differentiable but requires_grad is True")
            output_local = a_local & b_local
        elif op == ElementwiseOp.BITAND:
            if a.requires_grad or b.requires_grad:
                raise ValueError("BITAND operation is not differentiable but requires_grad is True")
            output_local = a_local & b_local
        else:
            raise ValueError(f"Unsupported operation: {op}")

        ctx.device_mesh_input = device_mesh_input
        ctx.placements_input = placements_input
        ctx.input_shape = input_shape
        ctx.op = op

        out = DTensor.from_local(
            output_local,
            device_mesh=device_mesh_input,
            placements=placements_input,
            shape=a.shape,
            stride=a.stride(),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output) -> tuple[DTensor | None, DTensor | None, None]:
        """Backward pass of distributed elementwise operation.

        Computes gradients with respect to both input tensors a and b.

        The gradients are:
        - For SUM: da = grad_output, db = grad_output
        - For SUB: da = grad_output, db = -grad_output
        - For PROD: da = grad_output * b, db = grad_output * a
        - For DIV: da = grad_output / b, db = -grad_output * output / b
        - For EQUAL: da = 0, db = 0 (not differentiable)

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFrame
            Context object containing saved tensors and metadata from forward pass.
        grad_output : DTensor
            Gradient of the loss with respect to the output tensor.
            Must have identical device mesh and placements as the input tensors.

        Returns
        -------
        tuple[DTensor, DTensor, None]
            Gradients with respect to a, b, and None for the op parameter.
            Both gradients have the same shape and distribution as their corresponding inputs.

        Raises
        ------
        TypeError
            If grad_output is not a DTensor.
        ValueError
            If grad_output has incompatible device mesh or placements compared
            to the input tensors from the forward pass.
        """

        if not isinstance(grad_output, DTensor):
            raise TypeError(
                f"Input 'grad_output' must be of type DTensor. Got type {type(grad_output)}."
            )

        if grad_output.device_mesh != ctx.device_mesh_input:
            raise ValueError(
                f"Input 'grad_output' must have the same device mesh as the input tensor. "
                f"Got device meshes {grad_output.device_mesh} and {ctx.device_mesh_input}."
            )

        if grad_output.placements != ctx.placements_input:
            raise ValueError(
                f"Input 'grad_output' must have the same placements as the input tensor. "
                f"Got placements {grad_output.placements} and {ctx.placements_input}."
            )

        if grad_output.shape != ctx.input_shape:
            raise ValueError(
                f"Input 'grad_output' must have the same shape as the input tensor. "
                f"Got shapes {grad_output.shape} and {ctx.input_shape}."
            )

        grad_output_local = grad_output.to_local()

        # Compute gradients based on operation
        da = None
        db = None

        if ctx.op == ElementwiseOp.SUM:
            if ctx.needs_input_grad[0]:
                da_local = grad_output_local
                da = DTensor.from_local(
                    da_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
            if ctx.needs_input_grad[1]:
                db_local = grad_output_local
                db = DTensor.from_local(
                    db_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
        elif ctx.op == ElementwiseOp.SUB:
            if ctx.needs_input_grad[0]:
                da_local = grad_output_local
                da = DTensor.from_local(
                    da_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
            if ctx.needs_input_grad[1]:
                db_local = -grad_output_local
                db = DTensor.from_local(
                    db_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
        elif ctx.op == ElementwiseOp.PROD:
            # Unpack saved_tensors once for checkpoint compatibility - must only be done once
            a_local, b_local = ctx.saved_tensors

            if ctx.needs_input_grad[0]:
                da_local = grad_output_local * b_local
                da = DTensor.from_local(
                    da_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
            if ctx.needs_input_grad[1]:
                db_local = grad_output_local * a_local
                db = DTensor.from_local(
                    db_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
        elif ctx.op == ElementwiseOp.DIV:
            # Unpack saved_tensors once for checkpoint compatibility - must only be done once
            b_local, output_over_b_local = ctx.saved_tensors

            if ctx.needs_input_grad[0]:
                # Gradient w.r.t. numerator: da = grad_output / b
                da_local = grad_output_local / b_local
                da = DTensor.from_local(
                    da_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
            if ctx.needs_input_grad[1]:
                # Gradient w.r.t. denominator: db = -grad_output * output / b
                db_local = -grad_output_local * output_over_b_local
                db = DTensor.from_local(
                    db_local,
                    device_mesh=ctx.device_mesh_input,
                    placements=ctx.placements_input,
                    shape=grad_output.shape,
                    stride=grad_output.stride(),
                )
        elif ctx.op == ElementwiseOp.EQUAL:  # dummy op, not differentiable
            pass
        elif ctx.op == ElementwiseOp.BITAND:  # dummy op, not differentiable
            pass
        else:
            raise ValueError(f"Unsupported operation: {ctx.op}")

        return da, db, None


class _ElementwiseOpInPlaceImpl(torch.autograd.Function):
    """Apply an elementwise operation to the local shard of ``a`` without allocating. Inference-only."""

    @staticmethod
    def forward(ctx, a: DTensor, b: DTensor, op: ElementwiseOp) -> DTensor:
        if a.requires_grad or b.requires_grad:
            raise ValueError("In-place elementwise operations do not support autograd")
        if b.device_mesh != a.device_mesh or b.placements != a.placements or b.shape != a.shape:
            raise ValueError(
                "Input tensors 'a' and 'b' must have identical device mesh, placements and shape. "
                f"Got {a.device_mesh}, {a.placements}, {a.shape} and "
                f"{b.device_mesh}, {b.placements}, {b.shape}."
            )

        # to_local() returns a view of a's shard, so augmented assignment updates a.
        a_local = a.to_local()
        b_local = b.to_local()
        if op == ElementwiseOp.SUM:
            a_local += b_local
        elif op == ElementwiseOp.SUB:
            a_local -= b_local
        elif op == ElementwiseOp.PROD:
            a_local *= b_local
        elif op == ElementwiseOp.DIV:
            a_local /= b_local
        elif op in (ElementwiseOp.EQUAL, ElementwiseOp.BITAND):
            a_local &= b_local
        else:
            raise ValueError(f"Unsupported operation: {op}")
        ctx.mark_dirty(a)
        return a


def elementwise_op(a: DTensor, b: DTensor, op: ElementwiseOp, *, in_place: bool = False) -> DTensor:
    """Apply elementwise operation to two distributed tensors.

    This function performs element-wise operations between two distributed tensors.
    Supported operations are summation (a + b), subtraction (a - b), multiplication (a * b),
    division (a / b), logical AND (a & b), and bitwise AND (a & b). The operation is performed
    efficiently using local tensor operations while maintaining gradient
    computation capabilities.

    Parameters
    ----------
    a : DTensor
        First input tensor. Can have any shape and sharding strategy.
    b : DTensor
        Second input tensor. Must have identical shape, device mesh,
        and placements as a.
    op : ElementwiseOp
        The operation to perform (ElementwiseOp.SUM, ElementwiseOp.SUB, ElementwiseOp.PROD, ElementwiseOp.DIV, ElementwiseOp.EQUAL, or ElementwiseOp.BITAND).
    in_place : bool, optional
        Write the result into the local shard of ``a`` and return ``a``. This
        does not support autograd, so neither input may require gradients.
        Default is False.

    Returns
    -------
    DTensor
        Output tensor with shape identical to input tensors.
        Contains the result of the specified operation.

    Examples
    --------
    >>> # Assume we have distributed tensors a and b with shape (B, N, D)
    >>> sum_output = elementwise_op(a, b, ElementwiseOp.SUM)
    >>> # sum_output = a + b, computed in distributed fashion
    >>>
    >>> sub_output = elementwise_op(a, b, ElementwiseOp.SUB)
    >>> # sub_output = a - b, computed in distributed fashion
    >>>
    >>> prod_output = elementwise_op(a, b, ElementwiseOp.PROD)
    >>> # prod_output = a * b, computed in distributed fashion
    >>>
    >>> div_output = elementwise_op(a, b, ElementwiseOp.DIV)
    >>> # div_output = a / b, computed in distributed fashion
    >>>
    >>> equal_output = elementwise_op(a, b, ElementwiseOp.EQUAL)
    >>> # equal_output = a == b, computed in distributed fashion
    >>>
    >>> bitand_output = elementwise_op(a, b, ElementwiseOp.BITAND)
    >>> # bitand_output = a & b, computed in distributed fashion

    Notes
    -----
    - Both input tensors must be DTensors with compatible device meshes and placements
    - Partial placements are not currently supported
    - The function is differentiable and supports gradient computation
    - The operation is performed on local tensor chunks for efficiency
    """
    if in_place:
        return _ElementwiseOpInPlaceImpl.apply(a, b, op)  # type: ignore
    return _ElementwiseOpImpl.apply(a, b, op)  # type: ignore
