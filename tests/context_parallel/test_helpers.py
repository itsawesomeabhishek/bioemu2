import os
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.context_parallel.helpers import _cpu_cuequivariance_env


@pytest.mark.parametrize("body_fails", [False, True])
def test_cpu_env_without_cuequivariance(monkeypatch: pytest.MonkeyPatch, body_fails: bool) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/libs")
    with (
        patch("tests.context_parallel.helpers.importlib.util.find_spec", return_value=None),
        pytest.raises(RuntimeError, match="test body failed") if body_fails else nullcontext(),
    ):
        with _cpu_cuequivariance_env():
            assert os.environ["OMP_NUM_THREADS"] == "1"
            assert os.environ["LD_LIBRARY_PATH"] == "/existing/libs"
            if body_fails:
                raise RuntimeError("test body failed")
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/libs"


@pytest.mark.parametrize("has_driver", [False, True])
def test_cpu_env_with_cuequivariance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, has_driver: bool
) -> None:
    cue_dir = tmp_path / "cuequivariance_ops"
    cue_dir.mkdir()
    (cue_dir / "libcue_ops.so").touch()
    cuda_stub = tmp_path / "libcuda.so"
    cuda_stub.touch()
    cue_spec = ModuleSpec("cuequivariance_ops", loader=None, origin=str(cue_dir / "__init__.py"))
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/libs")

    with (
        patch("tests.context_parallel.helpers.importlib.util.find_spec", return_value=cue_spec),
        patch(
            "tests.context_parallel.helpers.ctypes.CDLL",
            side_effect=None if has_driver else OSError("no CUDA driver"),
        ),
        patch.object(Path, "glob", return_value=iter([cuda_stub])),
    ):
        with _cpu_cuequivariance_env():
            assert os.environ["OMP_NUM_THREADS"] == "1"
            library_dirs = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
            if has_driver:
                assert library_dirs == [str(cue_dir), "/existing/libs"]
            else:
                assert library_dirs[1:] == [str(cue_dir), "/existing/libs"]
                stub_link = Path(library_dirs[0]) / "libcuda.so.1"
                assert stub_link.resolve() == cuda_stub
        if not has_driver:
            assert not stub_link.exists()
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["LD_LIBRARY_PATH"] == "/existing/libs"
