# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""CP logger configuration and shell environment derivation."""

import logging
import os
import subprocess
from pathlib import Path

import pytest

import bioemu2.context_parallel as _cp
from bioemu2.context_parallel.utils.session import (
    _CP_LOGGER_NAME,
    TRACE_LEVEL,
    _apply_cp_verbosity,
)

_MANAGED_ENV = (
    "EMU_CP_VERBOSITY",
    "NCCL_DEBUG",
    "TORCH_DISTRIBUTED_DEBUG",
    "EMU_CP_GRID",
    "CUDA_VISIBLE_DEVICES",
    "EMU_CP_NNODES",
    "EMU_CP_GPU_COUNT",
)
_CP_ENV_SH = (Path(_cp.__file__).parent / "scripts" / "cp_env.sh").resolve()


@pytest.fixture(autouse=True)
def _isolate_env():
    saved = {k: os.environ.pop(k, None) for k in _MANAGED_ENV}
    cp_logger = logging.getLogger(_CP_LOGGER_NAME)
    saved_level = cp_logger.level
    yield
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v
    cp_logger.setLevel(saved_level)


@pytest.mark.parametrize(
    ("verbosity", "level"),
    [
        (None, logging.INFO),
        ("warning", logging.WARNING),
        ("info", logging.INFO),
        ("debug", logging.DEBUG),
        ("trace", TRACE_LEVEL),
        ("  DEBUG  ", logging.DEBUG),
    ],
)
def test_apply_cp_verbosity_sets_logger_level(verbosity, level):
    if verbosity is not None:
        os.environ["EMU_CP_VERBOSITY"] = verbosity
    _apply_cp_verbosity()
    assert logging.getLogger(_CP_LOGGER_NAME).level == level
    assert "NCCL_DEBUG" not in os.environ
    assert "TORCH_DISTRIBUTED_DEBUG" not in os.environ


@pytest.mark.parametrize(
    ("verbosity", "preset_nccl", "nccl", "tdd"),
    [
        (None, None, "WARN", "OFF"),
        ("warning", None, "WARN", "OFF"),
        ("info", None, "WARN", "OFF"),
        ("debug", None, "WARN", "OFF"),
        ("trace", None, "INFO", "DETAIL"),
        ("trace", "VERSION", "VERSION", "DETAIL"),
        ("info", "INFO", "WARN", "OFF"),
    ],
)
def test_cp_env_sh_sets_nccl_debug_env(verbosity, preset_nccl, nccl, tdd):
    env = os.environ | {"EMU_CP_GPU_COUNT": "4"}
    if verbosity is not None:
        env["EMU_CP_VERBOSITY"] = verbosity
    if preset_nccl is not None:
        env["NCCL_DEBUG"] = preset_nccl
    result = subprocess.run(
        ["bash", "-c", f'source "{_CP_ENV_SH}" && echo "$NCCL_DEBUG|$TORCH_DISTRIBUTED_DEBUG"'],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{nccl}|{tdd}"


@pytest.mark.parametrize(
    ("env", "nnodes", "grid", "devices"),
    [
        ({"EMU_CP_GPU_COUNT": "4"}, "", "2,2", "0,1,2,3"),
        ({"EMU_CP_GPU_COUNT": "8"}, "2", "4,4", "0,1,2,3,4,5,6,7"),
        ({"EMU_CP_GPU_COUNT": "1"}, "", "1,1", "0"),
        ({"EMU_CP_GPU_COUNT": "4"}, "4", "4,4", "0,1,2,3"),
        ({"EMU_CP_GPU_COUNT": "4", "EMU_CP_NNODES": "1"}, "4", "4,4", "0,1,2,3"),
        ({"EMU_CP_GPU_COUNT": "4", "CUDA_VISIBLE_DEVICES": "4,5,6,7"}, "", "unset", "4,5,6,7"),
        ({"EMU_CP_GRID": "2,4"}, "", "2,4", "unset"),
    ],
)
def test_cp_env_sh_derives_grid_from_nnodes_and_gpu_count(env, nnodes, grid, devices):
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{_CP_ENV_SH}" {nnodes} && '
            'echo "${EMU_CP_GRID-unset}|${CUDA_VISIBLE_DEVICES-unset}"',
        ],
        capture_output=True,
        text=True,
        env=os.environ | env,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"{grid}|{devices}"
