# Vendored from NVIDIA Fold-CP (MIT): https://github.com/NVIDIA-BioNeMo/boltz-cp @ f76f37a77c854b56b6250b426af8c2d63b501f7f
# Source: src/boltz/distributed/model/layers/dtensor_metadata_tools.py

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

from typing import Union

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Placement


def raise_if_incorrect_dtensor_metadata_args(
    dtensor_instance: DTensor,
    dtensor_name: str,
    expected_shape: Union[tuple[int, ...], None] = None,
    expected_device_mesh: Union[DeviceMesh, None] = None,
    expected_placements: Union[tuple[Placement], None] = None,
    check_for_partial_placements: bool = False,
):
    """Check if the DTensor metadata is correct.

    This function will check in order:
        - that the DTensor instance is a DTensor
        - that the DTensor instance has the expected shape
        - that the DTensor instance has the expected device mesh
        - that the DTensor instance has the expected placements
        - that the DTensor instance does not have a Partial placement

    If any of the checks fail, an Exception is raised.


    Parameters
    ----------
    dtensor_instance: DTensor
        The DTensor instance to check.
    dtensor_name: str
        The name of the DTensor instance to check.
    expected_shape: Union[tuple, None]
        The expected shape of the DTensor.  If None, the check is skipped.
    expected_device_mesh: Union[DeviceMesh, None]
        The expected device mesh.  If None, the check is skipped.
    expected_placements: Union[tuple[Placement], None]
        The expected placements.  If None, the check is skipped.
    check_for_partial_placements: bool
        Whether or not to check for partial placements in the input dtensor.

    Notes
    -----
    (1) A DTensor with a Partial placement is considered invalid, in this library
    as a temporary measure to avoid its complexity.
    (2) A valid usage of this function is to check the equality of the
    placements between two DTensor instances, and to check that the placements
    of the input DTensor do not contain Partial placements.

    Raises
    ------
    TypeError
        If dtensor_instance is not a DTensor.
    ValueError
        If the DTensor metadata is incorrect.
    """

    if not isinstance(dtensor_instance, DTensor):
        raise TypeError(
            ", ".join(
                [
                    f"DTensor instance '{dtensor_instance}' should have type {DTensor}",
                    f"but instead has type {type(dtensor_instance)}.",
                ]
            )
        )

    # Consolidate if statements, each is >~ 10ns
    if expected_shape is not None and not dtensor_instance.shape == expected_shape:
        raise ValueError(
            ", ".join(
                [
                    f"DTensor instance '{dtensor_name}' should have shape {expected_shape}",
                    f"but instead has shape {dtensor_instance.shape}.",
                ]
            )
        )

    if (
        expected_device_mesh is not None
        and not dtensor_instance.device_mesh == expected_device_mesh
    ):
        raise ValueError(
            ", ".join(
                [
                    f"DTensor instance '{dtensor_name}' should have device mesh {expected_device_mesh}",
                    f"but instead has device mesh {dtensor_instance.device_mesh}.",
                ]
            )
        )

    if expected_placements is not None and not tuple(dtensor_instance.placements) == tuple(
        expected_placements
    ):
        raise ValueError(
            ", ".join(
                [
                    f"DTensor instance '{dtensor_name}' should have placements {expected_placements}",
                    f"but instead has placements {dtensor_instance.placements}.",
                ]
            )
        )

    if check_for_partial_placements:
        for placement in dtensor_instance.placements:
            if isinstance(placement, Partial):
                raise ValueError(
                    ", ".join(
                        [
                            f"DTensor instance '{dtensor_name}' should not have have placement of type {Partial}",
                            f"but instead has placements {dtensor_instance.placements}.",
                        ]
                    )
                )
