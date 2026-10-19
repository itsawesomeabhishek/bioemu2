"""Full and sliding-window attention with additive pair bias.

(1) Full attention — trunk + per-token structure module
    (``AttentionPairBias``).

        out_{b,m,i,h} = Σ_j  softmax_j(  Q_{b,m,i,h} · K_{b,m,j,h} / √d
                                       + B_{b,h,i,j}  )  ·  V_{b,m,j,h}

    Shapes (M = multiplicity, H = heads, d = head_dim; pair rep B is M-invariant):

        Q, K, V   : [B*M, N, H, d]
        B (bias)  : [B,   H, N, N]                 (no M; broadcast over M)
        output    : [B*M, N, H, d]

(2) Sliding-window attention — atom transformer
    (``AttentionPairBiasSlidingWindow``).

        out_{b,m,k,i,h} = Σ_{j∈keys(k)}  softmax_j(  Q · K / √d + B )  ·  V

    Shapes (NW = num_windows, batch axes folded row-major (b·M + m)·NW + k):

        Q          : [B*M*NW, W,      H, d]
        K, V       : [B*M*NW, H_keys, H, d]
        B (bias)   : [B,      H, NW, W, H_keys]    (no M; broadcast over M)
        output     : [B*M*NW, W,      H, d]

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _at_least_float32(x: Tensor) -> Tensor:
    """Upcast to float32 if needed (float16/bfloat16), but preserve float64."""
    if x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    return x


class AttentionPairBias(nn.Module):
    """Full self-attention with additive pair bias and gated output projection.

    Computes multi-head attention where pair-representation features are projected
    to per-head biases added to the QK logits before softmax. Supports multiplicity
    broadcasting (pair bias is M-invariant) and optional query-tiling for memory-bound
    regimes.

    References: Jumper et al. 2021 — AlphaFold2, Evoformer attention with pair bias.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
        tile_size: int | None = 512,
        zero_init_output: bool = False,
        use_bias_in_layer_norm: bool = False,
        use_efficient_fusion: bool = False,
    ) -> None:
        super().__init__()
        assert c_s % num_heads == 0
        # The efficient fusion folds the LayerNorm gain γ into the projection weight, which
        # only works when the LayerNorm has no bias/offset term (see _proj_z_fused).
        if use_efficient_fusion and use_bias_in_layer_norm:
            raise ValueError(
                "use_efficient_fusion=True is incompatible with use_bias_in_layer_norm=True: "
                "the LayerNorm+Linear fold cannot absorb a LayerNorm bias."
            )
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.tile_size = tile_size
        self.use_bias_in_layer_norm = use_bias_in_layer_norm
        self.use_efficient_fusion = use_efficient_fusion

        self.initial_norm = initial_norm
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_qkv = nn.Linear(c_s, 3 * c_s, bias=False)
        self.proj_q_bias = nn.Parameter(torch.zeros(c_s))
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.norm_z = nn.LayerNorm(c_z, bias=use_bias_in_layer_norm)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        # adaLN-Zero style: start the attention residual branch at zero.
        if zero_init_output:
            nn.init.zeros_(self.proj_o.weight)

    def _proj_z_fused(self, z: Tensor) -> Tensor:
        """Project pair repr to attention bias, shape [B, H, N, N].

        When use_efficient_fusion=False (default):
          Standard LayerNorm → Linear → permute. This is numerically batch-invariant
          (a plain matmul) and works with or without a LayerNorm bias.

        When use_efficient_fusion=True (requires use_bias_in_layer_norm=False):
          Fuse LayerNorm(z) + LinearNoBias(z) into one F.conv2d.
          norm_z.weight is the γ (scale) parameter of LayerNorm (no bias/offset).
          proj_z.weight is [n_heads, c_z].
          Multiplying them folds γ into the linear weight: W_fused = W_linear * γ.
          The resulting [n_heads, c_z, 1, 1] kernel does per-pixel (per-pair) linear
          projection via F.conv2d, outputting [B, n_heads, N, N] directly — no permute
          needed on the output side. NB: cuDNN's 1×1 conv2d is not batch-invariant, so
          this path trades exact batch invariance for speed.
        """
        if not self.use_efficient_fusion:
            z = self.norm_z(z)  # [B, N, N, c_z]
            z = self.proj_z(z)  # [B, N, N, H]
            return z.permute(0, 3, 1, 2)  # [B, H, N, N]
        else:
            # Normalize without affine (strip γ — we fold it into the conv weight).
            z = F.layer_norm(z, (self.c_z,))
            z = z.permute(0, 3, 1, 2)  # [B, c_z, N, N]
            # Fuse γ into the linear weight → 1×1 conv kernel.
            weight = (self.proj_z.weight * self.norm_z.weight[None, :])[
                :, :, None, None
            ]  # [n_heads, c_z, 1, 1]
            return F.conv2d(z, weight)  # [B, H, N, N]

    def forward(
        self,
        s: Tensor,  # [B*M, N, c_s]
        z: Tensor,  # [B, N, N, c_z]
        mask: Tensor,  # [B, N] bool
        multiplicity: int = 1,
    ) -> Tensor:  # [B*M, N, c_s]
        """
        Args:
            s: single features, shape ``[B*M, N, c_s]``.
            z: pair features, shape ``[B, N, N, c_z]``
            mask: boolean mask for valid tokens, shape ``[B, N]``.
            multiplicity: number of copies of each sequence in the batch (M).

        Returns:
            Updated single representation, shape ``[B*M, N, c_s]``.
        """

        assert s.ndim == 3 and z.ndim == 4 and mask.ndim == 2, f"{s.shape}, {z.shape}, {mask.shape}"
        assert s.shape[0] == z.shape[0] * multiplicity, (
            f"{s.shape}, {z.shape}, multiplicity={multiplicity}"
        )

        BM = s.shape[0]  # B*M
        if self.initial_norm:
            s = self.norm_s(s)
        N = s.shape[1]

        qkv = self.proj_qkv(s)
        q, k, v = qkv.chunk(3, dim=-1)
        q = (q + self.proj_q_bias).view(BM, N, self.num_heads, self.head_dim)
        k = k.view(BM, N, self.num_heads, self.head_dim)
        v = v.view(BM, N, self.num_heads, self.head_dim)
        g = self.proj_g(s).sigmoid()

        z_proj = self._proj_z_fused(z)  # [B, H, N, N]

        with torch.autocast("cuda", enabled=False):
            q, k, v = _at_least_float32(q), _at_least_float32(k), _at_least_float32(v)
            z_f = _at_least_float32(z_proj)
            N_q = q.shape[1]
            B_real = z_f.shape[0]

            chunk = self.tile_size
            if chunk is None or chunk >= N_q:
                attn = torch.einsum("bihd,bjhd->bhij", q, k)
                attn /= self.head_dim**0.5
                # pair bias [B, H, N, N] → broadcast over M
                if attn.shape == z_f.shape:
                    attn += z_f
                else:
                    assert z_f.shape == torch.Size([B_real, self.num_heads, N_q, N_q]), (
                        f"{z_f.shape}, expected {[B_real, self.num_heads, N_q, N_q]}"
                    )
                    attn.view(B_real, multiplicity, *z_f.shape[1:]).add_(z_f.unsqueeze(1))
                # mask [B, 1, 1, N]; broadcast over M via view (no copy)
                mask_view = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
                if multiplicity > 1:
                    attn.view(B_real, multiplicity, *attn.shape[1:]).masked_fill_(
                        ~mask_view.unsqueeze(1), -self.inf
                    )
                else:
                    attn.masked_fill_(~mask_view, -self.inf)
                attn = attn.softmax(dim=-1)
                o = torch.einsum("bhij,bjhd->bihd", attn, v)
            else:
                # mask [B, 1, 1, N]; broadcast over M via view (no copy)
                mask_view = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
                o = torch.zeros(
                    BM,
                    N_q,
                    self.num_heads,
                    self.head_dim,
                    device=q.device,
                    dtype=q.dtype,
                )
                for qs in range(0, N_q, chunk):
                    qe = min(qs + chunk, N_q)
                    attn = torch.einsum("bchd,bnhd->bhcn", q[:, qs:qe], k)
                    attn /= self.head_dim**0.5
                    z_chunk = z_f[..., qs:qe, :]
                    if attn.shape == z_chunk.shape:
                        attn += z_chunk
                    else:
                        attn.view(B_real, multiplicity, *z_chunk.shape[1:]).add_(
                            z_chunk.unsqueeze(1)
                        )
                    if multiplicity > 1:
                        attn.view(B_real, multiplicity, *attn.shape[1:]).masked_fill_(
                            ~mask_view.unsqueeze(1), -self.inf
                        )
                    else:
                        attn.masked_fill_(~mask_view, -self.inf)
                    attn = attn.softmax(dim=-1)
                    o[:, qs:qe] = torch.einsum("bhcn,bnhd->bchd", attn, v)

        o = o.to(g.dtype).reshape(BM, -1, self.c_s)
        return self.proj_o(g * o)


class AttentionPairBiasSlidingWindow(AttentionPairBias):
    """Sliding-window attention with pair bias for atom-level transformers.

    Each query window of size ``W`` attends to a key window of size ``H`` (centred
    on the query window with symmetric context padding). Pair bias is shaped
    ``[B, H_heads, NW, W, H_keys]`` and broadcast over the multiplicity axis.

    Inherits projections from :class:`AttentionPairBias`; overrides ``forward``
    with zero-padded unfold key-gather and windowed pair-bias addition.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        window_queries: int = 32,
        window_keys: int = 128,
        inf: float = 1e6,
        initial_norm: bool = True,
        tile_size: int | None = 512,
        zero_init_output: bool = False,
        use_bias_in_layer_norm: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        assert c_s % num_heads == 0
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.tile_size = tile_size
        self.window_queries = window_queries
        self.window_keys = window_keys

        self.initial_norm = initial_norm
        if initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_qkv = nn.Linear(c_s, 3 * c_s, bias=False)
        self.proj_q_bias = nn.Parameter(torch.zeros(c_s))
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.norm_z = nn.LayerNorm(c_z, bias=use_bias_in_layer_norm)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        # adaLN-Zero style: start the attention residual branch at zero.
        if zero_init_output:
            nn.init.zeros_(self.proj_o.weight)

    def forward(
        self,
        s: Tensor,  # [B*M*NW, W, c_s]
        z: Tensor,  # [B, NW, W, H_keys, c_z]
        mask: Tensor,  # [B*NW, W] bool (no M axis)
        multiplicity: int = 1,
    ) -> Tensor:  # [B*M*NW, W, c_s]
        num_heads, head_dim, c_s = self.num_heads, self.head_dim, self.c_s
        M = multiplicity
        B, NW, W, H = z.shape[0], z.shape[1], z.shape[2], z.shape[3]

        assert s.ndim == 3 and z.ndim == 5 and mask.ndim == 2

        folded = s.shape[0]  # B * M * NW
        BM = folded // NW  # B * M
        context = (H - W) // 2

        if self.initial_norm:
            s = self.norm_s(s)

        qkv = self.proj_qkv(s)
        q_raw, kv_raw = qkv.split([c_s, 2 * c_s], dim=-1)
        q = (q_raw + self.proj_q_bias).view(folded, W, num_heads, head_dim)
        g = self.proj_g(s).sigmoid()

        # Fused window-gather for k and v (bias-free linear: zero pad stays zero).
        kv = (
            F.pad(kv_raw.reshape(BM, NW * W, 2 * c_s), (0, 0, context, context))
            .unfold(dimension=1, size=H, step=W)
            .transpose(-1, -2)
            .reshape(folded, H, 2, num_heads, head_dim)
        )
        k, v = kv.unbind(dim=2)

        # mask is [B*NW, W] (no M axis) — gather to [B*NW, H] then broadcast over M
        B_NW = B * NW
        mask_keys = (
            F.pad(mask.reshape(B, NW * W), (context, context))
            .unfold(dimension=1, size=H, step=W)
            .reshape(B_NW, H)
        )

        z_bias = self.proj_z(self.norm_z(z)).permute(0, 1, 4, 2, 3)  # [B, NW, num_heads, W, H]

        with torch.autocast("cuda", enabled=False):
            q, k, v = _at_least_float32(q), _at_least_float32(k), _at_least_float32(v)
            z_bias_f = _at_least_float32(z_bias)

            attn = torch.einsum("bihd,bjhd->bhij", q, k)  # [folded, num_heads, W, H]
            attn /= head_dim**0.5

            # pair bias [B, NW, num_heads, W, H] → broadcast over M
            bias_aligned = z_bias_f.unsqueeze(1)  # [B, 1, NW, num_heads, W, H]
            attn.view(B, M, NW, num_heads, W, H).add_(bias_aligned)

            # mask_keys [B*NW, H] → [B, 1, NW, 1, 1, H]; True=valid, False=pad→-inf
            mask_view = mask_keys.view(B, NW, H)[:, None, :, None, None, :]
            attn.view(B, M, NW, num_heads, W, H).masked_fill_(~mask_view, -self.inf)

            attn = attn.softmax(dim=-1)
            o = torch.einsum("bhij,bjhd->bihd", attn, v)

        o = o.to(g.dtype).reshape(folded, W, c_s)
        return self.proj_o(g * o)


__all__ = [
    "AttentionPairBias",
    "AttentionPairBiasSlidingWindow",
]
