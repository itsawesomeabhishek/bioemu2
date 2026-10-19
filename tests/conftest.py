import os
import tarfile
from pathlib import Path

# Cap intra-op threads before torch is imported (it reads OMP_NUM_THREADS once at
# import). Leaving it uncapped lets the reduction order in CPU float32 matmuls vary
# with the runner's core count, which pushes the padding-invariance gradient checks
# in tests/biomoa/ops/test_triangle_multiplication.py over their tight tolerances.
# setdefault lets the caller still override from the shell.
os.environ.setdefault("OMP_NUM_THREADS", "2")

from collections.abc import Generator
from contextlib import contextmanager

import pytest
import torch


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if torch.cuda.device_count() >= 4:
        return
    skip = pytest.mark.skip(reason="Requires at least four visible CUDA GPUs")
    for item in items:
        if item.get_closest_marker("requires_4gpu") is not None:
            item.add_marker(skip)


@pytest.fixture
def gpu_device() -> torch.device:
    """Return a CUDA device, skipping the test if unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA GPU")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def pdb_files_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Extract immutable RCSB structures downloaded on 2026-08-24 for offline tests."""
    target_path = tmp_path_factory.mktemp("pdb_files")
    for archive_path in (Path(__file__).parent / "assets").glob("rcsb_structures_*.tar.gz"):
        with tarfile.open(archive_path) as archive:
            archive.extractall(target_path, filter="data")
    return target_path


@contextmanager
def set_matmul_precision(precision: str) -> Generator[None, None, None]:
    """Context manager for temporarily setting torch float32 matmul precision."""
    prev_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_precision)
