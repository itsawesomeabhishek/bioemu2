"""SwiGLU MLP with optional LayerNorm and hidden-dimension chunking."""

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwiGLU(nn.Module):
    """Activation gate: splits the last dimension in half and applies SiLU gating."""

    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


class Transition(nn.Module):
    """SwiGLU-style two-layer MLP.

    The pre-LayerNorm is optional: pass ``use_norm=False`` when the surrounding
    block already normalises the input (e.g. the dual-stream diffusion stack).
    When disabled, no ``self.norm`` submodule is created so the ``state_dict``
    has no ``norm.*`` keys.

    Args:
        dim: Input (and default output) dimension.
        hidden: Hidden dimension of the gated MLP.
        out_dim: Output dimension; defaults to *dim*.
        use_norm: If True, prepend a LayerNorm.
        chunk_size: If set, tile the hidden dimension during inference to
            reduce peak memory.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        out_dim: int | None = None,
        use_norm: bool = True,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.use_norm = use_norm
        self.chunk_size = chunk_size
        if use_norm:
            self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(dim, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, out_dim, bias=False)
        self.hidden = hidden

    def forward(self, x: Tensor) -> Tensor:
        """SwiGLU forward: ``fc3(silu(fc1(norm(x))) * fc2(norm(x)))``.

        When ``use_norm=False`` the pre-LayerNorm step is skipped and the
        forward becomes ``fc3(silu(fc1(x)) * fc2(x))``.

        ``self.chunk_size`` (set in ``__init__``) tiles the hidden dimension
        to reduce peak memory at inference time. Always bypassed during
        training to preserve the un-tiled baseline.

        Args:
            x: Input tensor ``[*, dim]`` (typically ``[B, N, N, dim]``).

        Returns:
            Output tensor with the same shape as *x*.
        """
        if self.use_norm:
            x = self.norm(x)
        if self.chunk_size is None or self.training:
            a = self.fc1(x)
            a = F.silu(a)
            b = self.fc2(x)
            b *= a
            b = self.fc3(b)
            return b
        # Chunked: iterate over hidden dimension to reduce peak memory
        out_dim = self.fc3.weight.shape[0]
        x_out = x.new_empty(*x.shape[:-1], out_dim)
        for i in range(0, self.hidden, self.chunk_size):
            a = x @ self.fc1.weight[i : i + self.chunk_size].T
            a = F.silu(a, inplace=True)
            a *= x @ self.fc2.weight[i : i + self.chunk_size].T
            proj = a @ self.fc3.weight[:, i : i + self.chunk_size].T
            del a
            if i == 0:
                x_out.copy_(proj)
            else:
                x_out += proj
            del proj
        return x_out
