from functools import partial
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.utils.session import cp_session, teardown_cp_state
from tests.context_parallel.helpers import spawn_cpu_cp_test


@pytest.mark.parametrize(
    ("world", "local", "grid", "message"),
    [
        (1, 1, None, "torchrun"),
        (8, 4, None, "perfect square"),
        (4, 4, "1,4", "square"),
    ],
)
def test_session_rejects_invalid_launcher(monkeypatch, world, local, grid, message):
    monkeypatch.setenv("WORLD_SIZE", str(world))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(local))
    monkeypatch.delenv("EMU_CP_GRID", raising=False)
    if grid is not None:
        monkeypatch.setenv("EMU_CP_GRID", grid)
    with pytest.raises(ValueError, match=message), cp_session(device_type="cpu"):
        pytest.fail("Invalid launcher entered a session")


def test_session_rejects_nesting_and_cleans_up_on_cpu(tmp_path: Path):
    """Check nesting, visible cleanup errors, and staggered teardown with real Gloo."""
    spawn_cpu_cp_test(partial(_cpu_worker, directory=tmp_path))
    assert (tmp_path / "rank0-cleaned-up").read_text() == "done"


def _cpu_worker(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    manager: DistributedManager,
    *,
    directory: Path,
) -> None:
    with pytest.raises(RuntimeError, match="cannot be nested"), cp_session(device_type="cpu"):
        pytest.fail("Nested session was accepted")
    assert dist.is_initialized()
    assert DistributedManager.is_initialized()

    with (
        patch.object(
            dist,
            "destroy_process_group",
            side_effect=RuntimeError("intentional destroy failure"),
        ),
        pytest.raises(RuntimeError, match="intentional destroy failure"),
    ):
        teardown_cp_state()
    assert dist.is_initialized()
    assert not DistributedManager.is_initialized()

    marker = directory / "rank0-cleaned-up"
    if dist.get_rank() == 0:
        teardown_cp_state()
        marker.write_text("done")
    else:
        deadline = monotonic() + 15
        while not marker.exists():
            if monotonic() >= deadline:
                raise TimeoutError("Rank 0 did not finish cleanup before rank 1")
            sleep(0.01)
        teardown_cp_state()
    assert not dist.is_initialized()
    assert not DistributedManager._state
