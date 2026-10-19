import warnings
from collections.abc import Generator
from contextlib import contextmanager

_TORCH_JIT_SCRIPT_DEPRECATION = (
    "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."
)


@contextmanager
def suppress_cuequivariance_jit_deprecation() -> Generator[None, None, None]:
    """Suppress cuEquivariance's import-time torch.jit.script deprecation."""
    with warnings.catch_warnings():
        showwarning = warnings.showwarning

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            if (
                issubclass(category, DeprecationWarning | FutureWarning)
                and str(message) == _TORCH_JIT_SCRIPT_DEPRECATION
            ):
                return
            showwarning(message, category, filename, lineno, file=file, line=line)

        warnings.showwarning = _showwarning
        yield
