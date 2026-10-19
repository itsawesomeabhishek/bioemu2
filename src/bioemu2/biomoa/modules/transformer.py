# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""Transformer blocks for biomoa: token (full) and atom (sliding-window) attention.

Shape symbols used throughout this module:
    ``B``   batch size
    ``M``   multiplicity (noise replicates per batch element)
    ``T``   number of tokens (full / token attention)
    ``N``   number of atoms (windowed / atom attention), ``N = NW * W``
    ``NW``  number of windows, ``NW = N // W``
    ``W``   query atoms per window
    ``H``   key atoms per window
    ``dim`` / ``dim_pairwise`` / ``dim_cond``: single / pair / conditioning feature dims

Token attention is always full; atom attention is always sliding-window.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bioemu2.biomoa.ops.attention_pair_bias import AttentionPairBias, AttentionPairBiasSlidingWindow
from bioemu2.biomoa.ops.transition import Transition


def ada_ln(x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer normalization: ``LN(x) * (1 + alpha) + beta``.

    The ``(1 + alpha)`` shift is baked into this function so that callers
    can pass the raw conditioning output directly.

    Args:
        x: The input tensor of shape ``(..., dim)``.
        alpha: The scaling factor of shape ``(..., dim)``.
        beta: The shifting factor of shape ``(..., dim)``.

    Returns:
        The normalized and adapted tensor of shape ``(..., dim)``.
    """
    dim = x.shape[-1]
    return F.layer_norm(x, (dim,), weight=None, bias=None) * (1 + alpha) + beta


def ada_scale(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """Apply adaptive scaling.

    Args:
        x: The input tensor of shape ``(..., dim)``.
        gamma: The scaling factor of shape ``(..., dim)``.

    Returns:
        The scaled tensor of shape ``(..., dim)``.
    """
    return gamma * x


class AttentionLayer(nn.Module):
    """Attention Layer."""

    def __init__(
        self,
        heads: int,
        dim: int = 384,
        dim_pairwise: int = 128,
        attn_pair_bias_tile_size: int | None = None,
        is_sliding_window: bool = False,
        window_queries: int | None = None,
        window_keys: int | None = None,
        zero_init_output: bool = False,
    ) -> None:
        """Initialize the attention layer.

        Args:
            heads: The number of heads.
            dim: The single representation dimension.
            dim_pairwise: The pairwise representation dimension.
            attn_pair_bias_tile_size: The attention pair bias tile size.
            is_sliding_window: If True, the pair-bias module operates in
                sliding-window attention mode and expects rank-5 z
                ``[B, NW, W, H, dim_pairwise]``; trunk/per-token callers leave it False.
            window_queries: Atoms per query window (W). Required when
                ``is_sliding_window`` is True; ignored otherwise.
            window_keys: Atoms per key window (H). Required when
                ``is_sliding_window`` is True; ignored otherwise.
            zero_init_output: If True, zero-init the attention-pair-bias output
                projection (``proj_o``) weight so the attention residual starts at zero.
        """
        super().__init__()

        if is_sliding_window:
            # The sliding-window APB carries its own fixed window geometry and does not
            # support distance bias or tiling (each key window is already small).
            assert window_queries is not None and window_keys is not None, (
                "sliding-window AttentionLayer requires window_queries / window_keys."
            )
            self.pair_bias_attn: AttentionPairBias | AttentionPairBiasSlidingWindow = (
                AttentionPairBiasSlidingWindow(
                    c_s=dim,
                    c_z=dim_pairwise,
                    num_heads=heads,
                    window_queries=window_queries,
                    window_keys=window_keys,
                    initial_norm=False,
                    zero_init_output=zero_init_output,
                )
            )
        else:
            self.pair_bias_attn = AttentionPairBias(
                c_s=dim,
                c_z=dim_pairwise,
                num_heads=heads,
                initial_norm=False,
                zero_init_output=zero_init_output,
                tile_size=512 if attn_pair_bias_tile_size is None else attn_pair_bias_tile_size,
            )

    def forward(
        self,
        tokens: Tensor,  # full: (B * M, T, dim) | windowed: (B * M * NW, W, dim)
        pairs: Tensor,  # full: (B, T, T, dim_pairwise) | windowed: (B, NW, W, H, dim_pairwise)
        mask: Tensor | None = None,  # full: (B, T) | windowed: (B * NW, W)
        multiplicity: int = 1,
    ) -> Tensor:  # same shape as tokens
        """Apply attention with pair bias.

        Args:
            tokens: Single representation, ``(B * M, T, dim)`` full or
                ``(B * M * NW, W, dim)`` windowed.
            pairs: Pair representation, ``(B, T, T, dim_pairwise)`` full or
                ``(B, NW, W, H, dim_pairwise)`` windowed; for windowed attention
                ``AttentionPairBias`` gathers the key window itself.
            mask: Validity mask (1 valid, 0 padded), ``(B, T)`` full or ``(B * NW, W)`` windowed.
            multiplicity: Number of noise replicates per batch element (M).

        Returns:
            Updated single representation, same shape as ``tokens``.
        """
        s = self.pair_bias_attn(
            s=tokens,
            z=pairs,
            mask=mask,
            multiplicity=multiplicity,
        )

        return s


class SlidingWindowAttentionLayer(nn.Module):
    """Sliding-window attention layer. This is just a wrapper around the AttentionLayer to handle reshaping for windowing."""

    def __init__(
        self,
        attn_window_queries: int,
        attn_window_keys: int,
        **atom_attention_kwargs,
    ) -> None:
        """Initialize the atom attention layer.

        Args:
            attn_window_queries: The attention window queries (W).
            attn_window_keys: The attention window keys (H).
            **atom_attention_kwargs: Forwarded to the underlying ``AttentionLayer``.
        """
        super().__init__()
        self.attn_window_queries = attn_window_queries
        self.attn_window_keys = attn_window_keys
        assert attn_window_queries > 1, (
            f"atten_window_queris has to be >1. get {attn_window_queries} instead."
        )
        assert attn_window_keys > 1, (
            f"attn_window_keys has to be >1. get {attn_window_keys} instead."
        )

        self.attention_layer = AttentionLayer(
            **atom_attention_kwargs,
            is_sliding_window=True,
            window_queries=attn_window_queries,
            window_keys=attn_window_keys,
        )

    def forward(
        self,
        tokens: Tensor,  # (B * M, N, atom_s)
        pairs: Tensor,  # pre-windowed (B, NW, W, H, atom_z), or flat (B, N, H, atom_z)
        mask: Tensor | None = None,  # (B, N)
        multiplicity: int = 1,
    ) -> Tensor:  # (B * M, N, atom_s)
        """Apply windowed attention with pair bias over atoms.

        Args:
            tokens: Atom single representation, ``(B * M, N, atom_s)``.
            pairs: Atom pair representation, pre-windowed ``(B, NW, W, H, atom_z)`` (as
                produced by the atom pair embedder) or the equivalent flat
                ``(B, N, H, atom_z)``; both fold to ``(B, NW, W, H, atom_z)``.
            mask: Atom validity mask (1 valid, 0 padded), ``(B, N)``.
            multiplicity: Number of noise replicates per batch element (M).

        Returns:
            Updated atom single representation, ``(B * M, N, atom_s)``.
        """
        W = self.attn_window_queries
        H = self.attn_window_keys

        B, N, D = tokens.shape
        NW = N // W

        # fold atoms into query windows
        tokens = tokens.view((B * NW, W, -1))  # (B * NW, W, atom_s)

        # mask has no M axis; fold into windows
        if mask is not None:
            B_real = pairs.shape[0]
            mask = mask.reshape(B_real * NW, W)  # (B_real * NW, W)

        # fold pair bias into query windows so AttentionPairBias gathers its own key window
        pairs = pairs.view((pairs.shape[0], NW, W, H, -1))  # (B, NW, W, H, atom_z)

        # main transformer
        tokens = self.attention_layer(
            tokens=tokens,
            pairs=pairs,
            mask=mask,
            multiplicity=multiplicity,
        )

        tokens = tokens.view((B, NW * W, D))  # (B * multiplicity, N, atom_s)

        return tokens


class ConditionedTransformerLayer(nn.Module):
    """Residual transformer block driven by externally-computed DiT conditioning vectors.

    For both an attention and a transition (MLP) residual sub-block, this applies the
    pattern ``x = x + gamma * op(lns_scale * ((1 + alpha) * LN(x) + beta))``. The
    attention op is full (:class:`AttentionLayer`) or windowed
    (:class:`SlidingWindowAttentionLayer`); the transition is a bare 2-projection SwiGLU
    ``Transition`` (``use_norm=False``, so the pre-norm here is the only LN).

    The conditioning vectors (attention/MLP ``alpha``/``beta``/``gamma``) are produced
    *inside* the layer from the conditioning activations ``c`` via its own
    ``LayerNorm -> SiLU -> Linear`` producer. This lets the atom
    encoder/decoder and the diffusion module stack the same class directly by passing
    ``c``, without a separate wrapper to bundle a producer with the block. Keeping the
    producer inside also means activation checkpointing recomputes the conditioning
    vectors in the backward pass.

    References:
        - Peebles & Xie 2023 - Scalable Diffusion Models with Transformers (DiT / adaLN-Zero)
        - arXiv:2502.05795 - LayerNorm Scaling (the ``lns_scale`` factor)
    """

    def __init__(
        self,
        dim: int,
        dim_pairwise: int,
        dim_cond: int,
        heads: int,
        transition_hidden: int,
        is_sliding_window: bool = False,
        window_queries: int | None = None,
        window_keys: int | None = None,
        attn_pair_bias_tile_size: int | None = None,
        lns_scale: float = 1.0,
    ) -> None:
        """Initialize the conditioned transformer layer.

        Args:
            dim: The single representation dimension.
            dim_pairwise: The pairwise representation dimension.
            dim_cond: Dimension of the conditioning input ``c``.
            heads: The number of attention heads.
            transition_hidden: Hidden dimension of the transition (MLP) sub-block.
            is_sliding_window: If True, use windowed atom attention
                (:class:`SlidingWindowAttentionLayer`); otherwise full attention
                (:class:`AttentionLayer`).
            window_queries: Atoms per query window (W). Required when
                ``is_sliding_window`` is True.
            window_keys: Atoms per key window (H). Required when
                ``is_sliding_window`` is True.
            attn_pair_bias_tile_size: The attention pair bias tile size (full attention only).
            lns_scale: LayerNorm-Scaling factor multiplying each ``ada_ln`` output
                (arXiv:2502.05795); ``1.0`` disables it.
        """
        super().__init__()
        self.lns_scale = lns_scale

        # Produces 2 x (alpha, beta, gamma): attention then MLP conditioning.
        self.cond = nn.Sequential(nn.LayerNorm(dim_cond), nn.SiLU(), nn.Linear(dim_cond, 6 * dim))

        if is_sliding_window:
            assert window_queries is not None and window_keys is not None, (
                "sliding-window ConditionedTransformerLayer requires window_queries / window_keys."
            )
            self.attention: nn.Module = SlidingWindowAttentionLayer(
                attn_window_queries=window_queries,
                attn_window_keys=window_keys,
                heads=heads,
                dim=dim,
                dim_pairwise=dim_pairwise,
                attn_pair_bias_tile_size=attn_pair_bias_tile_size,
            )
        else:
            self.attention = AttentionLayer(
                heads=heads,
                dim=dim,
                dim_pairwise=dim_pairwise,
                attn_pair_bias_tile_size=attn_pair_bias_tile_size,
            )

        self.transition = Transition(
            dim=dim,
            hidden=transition_hidden,
            use_norm=False,
        )

    def forward(
        self,
        x: Tensor,  # full: (B * M, T, dim) | windowed: (B * M, N, dim)
        pairs: Tensor,  # full: (B, T, T, dim_pairwise) | windowed: (B, NW, W, H, dim_pairwise)
        c: Tensor,  # full: (B * M, T, dim_cond) | windowed: (B * M, N, dim_cond)
        mask: Tensor | None = None,  # full: (B, T) | windowed: (B, N)
        multiplicity: int = 1,
    ) -> Tensor:  # same shape as x
        """Produce the conditioned vectors from ``c`` then apply attention and transition.

        Args:
            x: Single representation, ``(B * M, T, dim)`` full or ``(B * M, N, dim)`` windowed.
            pairs: Pair representation, ``(B, T, T, dim_pairwise)`` full or pre-windowed
                ``(B, NW, W, H, dim_pairwise)`` windowed.
            c: Conditioning activations, ``(B * M, T, dim_cond)`` full or
                ``(B * M, N, dim_cond)`` windowed. Drives adaLN and is never reshaped.
            mask: Validity mask, ``(B, T)`` full or ``(B, N)`` windowed; folds to
                ``(B * NW, W)`` inside windowed attention.
            multiplicity: Number of noise replicates per batch element (M).

        Returns:
            Updated single representation, same shape as ``x``.
        """
        attn_alpha, attn_beta, attn_gamma, mlp_alpha, mlp_beta, mlp_gamma = self.cond(c).chunk(
            6, dim=-1
        )
        h = self.attention(
            tokens=self.lns_scale * ada_ln(x, attn_alpha, attn_beta),
            pairs=pairs,
            mask=mask,
            multiplicity=multiplicity,
        )
        x = x + ada_scale(h, attn_gamma)

        h = self.transition(self.lns_scale * ada_ln(x, mlp_alpha, mlp_beta))
        x = x + ada_scale(h, mlp_gamma)
        return x


class ConditionedSlidingWindowTransformerLayer(nn.Module):
    """Windowed-atom-attention wrapper around :class:`ConditionedTransformerLayer`.

    Mirrors how :class:`SlidingWindowAttentionLayer` wraps :class:`AttentionLayer`: it
    builds a :class:`ConditionedTransformerLayer` in sliding-window mode (whose attention
    is a :class:`SlidingWindowAttentionLayer`) and delegates the forward pass. Consumers
    that only ever want windowed atom attention (the atom encoder/decoder and the
    diffusion atom stream) stack this class instead of passing ``is_sliding_window=True``
    explicitly.
    """

    def __init__(
        self,
        window_queries: int,
        window_keys: int,
        **conditioned_layer_kwargs,
    ) -> None:
        """Initialize the windowed conditioned transformer layer.

        Args:
            window_queries: Atoms per query window (W).
            window_keys: Atoms per key window (H).
            **conditioned_layer_kwargs: Forwarded to the underlying
                :class:`ConditionedTransformerLayer` (e.g. ``dim``, ``dim_pairwise``,
                ``dim_cond``, ``heads``, ``transition_hidden``).
        """
        super().__init__()
        self.transformer_layer = ConditionedTransformerLayer(
            **conditioned_layer_kwargs,
            is_sliding_window=True,
            window_queries=window_queries,
            window_keys=window_keys,
        )

    def forward(
        self,
        x: Tensor,  # (B * M, N, dim)
        pairs: Tensor,  # pre-windowed (B, NW, W, H, dim_pairwise)
        c: Tensor,  # (B * M, N, dim_cond)
        mask: Tensor | None = None,  # (B, N)
        multiplicity: int = 1,
    ) -> Tensor:  # (B * M, N, dim)
        """Delegate to the underlying windowed :class:`ConditionedTransformerLayer`.

        Args:
            x: Atom single representation, ``(B * M, N, dim)``.
            pairs: Atom pair conditioning, pre-windowed ``(B, NW, W, H, dim_pairwise)``.
            c: Conditioning activations, ``(B * M, N, dim_cond)``. Drives adaLN, never windowed.
            mask: Atom validity mask (1 valid, 0 padded), ``(B, N)``.
            multiplicity: Number of noise replicates per batch element (M).

        Returns:
            Updated atom single representation, ``(B * M, N, dim)``.
        """
        return self.transformer_layer(x, pairs, c, mask=mask, multiplicity=multiplicity)
