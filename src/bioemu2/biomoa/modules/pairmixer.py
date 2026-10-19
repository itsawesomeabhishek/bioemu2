# Portions adapted from Boltz. See NOTICES.md and licenses/LICENSE.BOLTZ-MIT.

"""PairMixer trunk modules: sequence embedder, pair stack, and MSA module."""

import torch
import torch.nn as nn

from bioemu2.biomoa.modules.conditioning import ContactConditioning, MethodConditioning
from bioemu2.biomoa.modules.relative_position_encoder import FusedRelativePositionEncoder
from bioemu2.biomoa.ops.dropout import apply_axiswise_dropout
from bioemu2.biomoa.ops.outer_product_mean import REGISTRY as OPM_REGISTRY
from bioemu2.biomoa.ops.outer_product_mean import OPMVariant
from bioemu2.biomoa.ops.pair_weighted_averaging import PairWeightedAveraging
from bioemu2.biomoa.ops.transition import Transition
from bioemu2.biomoa.ops.triangle_multiplication import REGISTRY as TRIMUL_REGISTRY
from bioemu2.biomoa.ops.triangle_multiplication import TriMulVariant
from bioemu2.biomoa.utils.precision import resolve_precision_cfg
from bioemu2.biomoa.utils.strategy import make
from bioemu2.biomoa.utils.tiling import resolve_tiling_cfg
from bioemu2.data_interface.const import NUM_TOKENS


class PairMixerSequenceEmbedder(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        s_input_dim: int,
        msa_s: int = 64,
        msa_blocks: int = 4,
        msa_dropout: float = 0.15,
        z_dropout: float = 0.25,
        num_blocks: int = 48,
        dropout: float = 0.0,  # 0.25 in the original PairMixer paper
        trimul_variant: TriMulVariant | str = TriMulVariant.CUEQ,
        opm_variant: OPMVariant | str = OPMVariant.VANILLA,
        tiling_cfg: dict | None = None,
        relpos_cfg: dict | None = None,
        precision_cfg: dict | None = None,
        use_contact_conditioning: bool = True,
        use_method_conditioning: bool = True,
        embed_method_into_pairs: bool = False,
        use_paired_feature: bool = True,
        use_unconditional_branch: bool = False,
        num_tokens: int = NUM_TOKENS,
    ):
        """
        End-to-end sequence embedder using an MSA module and PairMixer stack as
        described in [#pairmixer2025_1]_.

        Args:
            token_s: Number of token node features.
            token_z: Number of token pair features.
            s_input_dim: Dimension of the input single representation from the
                ``InputEmbedder`` (``token_s + num_token_types + num_msa_tokens
                + 1 + num_pocket_classes``).
            msa_s: Number of MSA features.
            msa_blocks: Number of MSA blocks.
            msa_dropout: Dropout used for MSA module.
            z_dropout: Dropout used on pairwise MSA features.
            num_blocks: Number of PairMixer blocks.
            dropout: Dropout used for PairMixer module.
            trimul_variant: Triangle multiplication variant ("full", "cueq", "cueq_tiled").
            opm_variant: Outer product mean variant ("vanilla" or "triton_memory_saving"). Default "vanilla".
            tiling_cfg: Optional tile / chunk sizes for inference. Recognised keys:
                ``tile_size`` (global default for any unset ``*_tile_size`` key),
                ``chunk_size`` (global default for any unset ``*_chunk_size`` key),
                ``opm_tile_size``, ``transition_chunk_size``, ``trimul_tile_size``,
                ``pwa_chunk_heads``. Tile/chunk sizes are bypassed during
                training (the un-tiled baseline is used for gradient
                computation).
            relpos_cfg: Optional configuration for the relative position encoder. Keys are
                ``r_max`` and ``s_max``. If None, the constructor's
                defaults will be used.
            precision_cfg: Optional matmul-precision overrides for this trunk's
                ``msa_module`` / ``pairmixer_module`` sub-trees (no APB knob).
                See :mod:`bioemu2.biomoa.utils.precision`.
            use_contact_conditioning: Add contact-dependent pair biases.
            use_method_conditioning: Add a method-dependent bias sourced from
                ``feats["method_feature"]`` to the single representation.
            embed_method_into_pairs: Also add method conditioning to the pair representation.
            use_paired_feature: Use feature from paired MSAs.
            use_unconditional_branch: Whether to create modules for an unconditional
                branch through the sequence trunk that skips the MSA module and all
                the PairMixer layers and just runs one Transition.
            num_tokens: number of tokens passed to MSA submodule

        References
        ----------
        .. [#pairmixer2025_1] Ouyang-Zhang, J., et al.
            Triangle Multiplication Is All You Need For Biomolecular Structure Representations.
            arXiv preprint arXiv:2510.18870. 2025.
        """
        super().__init__()

        # Resolve the per-module tile/chunk sizes from `tiling_cfg` and switch
        # any sub-module variant that is not tiling-capable to a preferred one
        # (a warning is logged for each swap).
        resolved, resolved_variants = resolve_tiling_cfg(
            tiling_cfg,
            module_name=type(self).__name__,
            variants={
                "opm_variant": opm_variant,
                "trimul_variant": trimul_variant,
            },
        )
        opm_tile_size = resolved["opm_tile_size"]
        transition_chunk_size = resolved["transition_chunk_size"]
        trimul_tile_size = resolved["trimul_tile_size"]
        pwa_chunk_heads = resolved["pwa_chunk_heads"]
        opm_variant = resolved_variants["opm_variant"]
        trimul_variant = resolved_variants["trimul_variant"]

        # Input projections — s_inputs is a concatenation of atom encoding and
        # per-token features (one-hot residue type, MSA profile, deletion mean,
        # pocket one-hot).
        self.s_init = torch.nn.Linear(s_input_dim, token_s, bias=False)
        self.z_init_1 = torch.nn.Linear(s_input_dim, token_z, bias=False)
        self.z_init_2 = torch.nn.Linear(s_input_dim, token_z, bias=False)

        self.token_bonds = torch.nn.Linear(1, token_z, bias=False)
        # The sequence trunk and structure model use separate relative-position
        # encoders so the trunk can be reused with different structure modules.
        if relpos_cfg is None:
            relpos_cfg = {}
        self.rel_pos = FusedRelativePositionEncoder(token_z=token_z, **relpos_cfg)

        self.use_contact_conditioning = use_contact_conditioning
        if use_contact_conditioning:
            self.contact_conditioning = ContactConditioning(token_z=token_z)

        self.use_method_conditioning = use_method_conditioning
        if embed_method_into_pairs and not use_method_conditioning:
            raise ValueError("embed_method_into_pairs requires use_method_conditioning=True")
        if use_method_conditioning:
            self.method_conditioning = MethodConditioning(
                token_s=token_s,
                token_z=token_z,
                embed_into_pairs=embed_method_into_pairs,
            )

        self.msa_module = PairMixerMSAModule(
            s_input_dim=s_input_dim,
            token_s=token_s,
            token_z=token_z,
            msa_s=msa_s,
            msa_blocks=msa_blocks,
            msa_dropout=msa_dropout,
            z_dropout=z_dropout,
            trimul_variant=trimul_variant,
            trimul_tile_size=trimul_tile_size,
            transition_chunk_size=transition_chunk_size,
            opm_variant=opm_variant,
            opm_tile_size=opm_tile_size,
            pwa_chunk_heads=pwa_chunk_heads,
            num_tokens=num_tokens,
            opm_matmul_precision=resolve_precision_cfg(
                precision_cfg,
                site="msa_module.opm_matmul_precision",
                variant=opm_variant,
            ),
            trimul_matmul_precision=resolve_precision_cfg(
                precision_cfg,
                site="msa_module.trimul_matmul_precision",
                variant=trimul_variant,
            ),
            use_paired_feature=use_paired_feature,
        )

        self.pairmixer_module = PairMixerModule(
            token_s=token_s,
            token_z=token_z,
            num_blocks=num_blocks,
            dropout=dropout,
            trimul_variant=trimul_variant,
            trimul_tile_size=trimul_tile_size,
            transition_chunk_size=transition_chunk_size,
            trimul_matmul_precision=resolve_precision_cfg(
                precision_cfg,
                site="pairmixer_module.trimul_matmul_precision",
                variant=trimul_variant,
            ),
        )

        # Recycling projections
        self.s_recycle = torch.nn.Linear(token_s, token_s, bias=False)
        self.z_recycle = torch.nn.Linear(token_z, token_z, bias=False)

        # Normalization layers
        self.s_norm = torch.nn.LayerNorm(token_s)
        self.z_norm = torch.nn.LayerNorm(token_z)

        # Unconditional layer
        self.use_unconditional_branch = use_unconditional_branch
        if use_unconditional_branch:
            self.uncond_transition = Transition(
                dim=token_z,
                hidden=token_z * 4,
                chunk_size=transition_chunk_size,
            )

    def forward(
        self,
        s_inputs: torch.Tensor,
        feats: dict[str, torch.Tensor],
        num_steps_recycle: int = 0,
        unconditional: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute end-to-end embeddings and distogram loss.

        Args:
            s_inputs: Initial input features.
            feats: Batched model features.
            num_steps_recycle: Number of recycling steps. This argument is
                ignored when ``unconditional=True`` is set.
            unconditional: Whether to bypass the main sequence
                trunk and use only the lightweight unconditional form
                of sequence encoding. This is used in classifier free
                guidance inference.

        Returns:
            Node features [B x T x Fn], pair features [B x T x T x Fp] and
            [B x T x T x Nbins], all in dense batch format.
        """

        if not unconditional and num_steps_recycle < 0:
            raise ValueError("num_steps_recycle must be non-negative; use unconditional=True.")

        # Initialize the sequence and pairwise embeddings.
        s_init: torch.Tensor = self.s_init(s_inputs)
        z_init: torch.Tensor = (
            self.z_init_1(s_inputs)[:, :, None] + self.z_init_2(s_inputs)[:, None, :]
        )
        if self.use_method_conditioning:
            single_method, pair_method = self.method_conditioning(
                method_feature=feats["method_feature"],
                token_pad_mask=feats["token_pad_mask"],
            )
            s_init = s_init + single_method
            if pair_method is not None:
                z_init = z_init + pair_method
        z_init = z_init + self.rel_pos(
            asym_id=feats["asym_id"],
            entity_id=feats["entity_id"],
            residue_index=feats["residue_index"],
            sym_id=feats["sym_id"],
        )
        z_init = z_init + self.token_bonds(feats["token_bonds"])
        if self.use_contact_conditioning:
            z_init = z_init + self.contact_conditioning(
                contact_conditioning=feats["contact_conditioning"],
                contact_threshold=feats["contact_threshold"],
                token_pad_mask=feats["token_pad_mask"],
            )

        s: torch.Tensor = torch.zeros_like(s_init)
        z: torch.Tensor = torch.zeros_like(z_init)

        is_grad_enabled = torch.is_grad_enabled()

        # Compute pairwise mask
        mask = feats["token_pad_mask"]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)

        if unconditional:
            if not self.use_unconditional_branch:
                raise ValueError(
                    "Passing unconditional=True at inference time requires use_unconditional_branch=True at __init__"
                )
            s = s_init
            z = z_init + self.uncond_transition(self.z_norm(z_init))
            return s, z

        for i in range(num_steps_recycle + 1):
            with torch.set_grad_enabled(is_grad_enabled and (i == num_steps_recycle)):
                # Clear the autocast cache to avoid retaining unused parameter casts.
                if self.training and (i == num_steps_recycle) and torch.is_autocast_enabled():
                    torch.clear_autocast_cache()

                s = s_init + self.s_recycle(self.s_norm(s))
                z = z_init + self.z_recycle(self.z_norm(z))

                _, z = self.msa_module(
                    s=None,
                    z=z,
                    emb=s_inputs,
                    msa=feats["msa"],
                    msa_mask=feats["msa_mask"],
                    has_deletion=feats["has_deletion"],
                    deletion_value=feats["deletion_value"],
                    msa_paired=feats["msa_paired"],
                    token_pad_mask=feats["token_pad_mask"],
                )

                s, z = self.pairmixer_module(s, z, pair_mask=pair_mask)

        return s, z


class PairMixerModule(nn.Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        num_blocks: int,
        dropout: float = 0.25,
        trimul_variant: TriMulVariant | str = TriMulVariant.CUEQ,
        trimul_tile_size: int | None = None,
        transition_chunk_size: int | None = None,
        trimul_matmul_precision: str | None = None,
    ) -> None:
        """Initialize the PairMixer module.

        Args:
            token_s: Single representation dimension.
            token_z: Pair representation dimension.
            num_blocks: Number of stacked PairMixer layers.
            dropout: Dropout rate applied inside each layer.
            trimul_variant: Triangle multiplication variant (``"full"``, ``"cueq"``,
                ``"cueq_tiled"``).
            trimul_tile_size: Tile size for the ``cueq_tiled`` variant
                (ignored by ``full`` and ``cueq``). ``None`` uses the variant's
                default tile size.
            transition_chunk_size: Hidden-dim chunk size baked into every
                ``Transition`` at construction time. ``None`` disables chunking.
                Bypassed during training.
            trimul_matmul_precision: Matmul precision override forwarded to TriMul
                kernel variants that support it. ``None`` preserves each kernel's own
                default.
        """
        super().__init__()
        self.token_z = token_z
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                PairMixerLayer(
                    token_s,
                    token_z,
                    dropout,
                    trimul_variant=trimul_variant,
                    trimul_tile_size=trimul_tile_size,
                    transition_chunk_size=transition_chunk_size,
                    trimul_matmul_precision=trimul_matmul_precision,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run all PairMixer layers sequentially.

        Args:
            s: Single representation ``[B, T, token_s]``.
            z: Pair representation ``[B, T, T, token_z]``.
            pair_mask: Pair mask ``[B, T, T]``.

        Returns:
            Updated ``(s, z)`` after ``num_blocks`` layers.
        """
        for layer in self.layers:
            s, z = layer(s, z, pair_mask)
        return s, z


class PairMixerLayer(nn.Module):
    """PairMixer layer with triangle multiplication and transition on pair representation."""

    def __init__(
        self,
        token_s: int,
        token_z: int,
        dropout: float = 0.25,
        trimul_variant: TriMulVariant | str = TriMulVariant.CUEQ,
        trimul_tile_size: int | None = None,
        transition_chunk_size: int | None = None,
        trimul_matmul_precision: str | None = None,
    ) -> None:
        """Initialize a PairMixer layer.

        Args:
            token_s: Single representation dimension.
            token_z: Pair representation dimension.
            dropout: Dropout rate for the pair update path.
            trimul_variant: Triangle multiplication variant. Defaults to the fused
                CuEQ Triton kernel; pass ``"full"`` to use PyTorch einsum.
            trimul_tile_size: Tile size for the ``cueq_tiled`` variant
                (ignored by ``full`` and ``cueq``). ``None`` uses the variant
                default.
            transition_chunk_size: Hidden-dim chunk size baked into the
                pair-side ``transition_z``. ``None`` disables chunking.
                Bypassed during training.
            trimul_matmul_precision: Matmul precision override forwarded to TriMul
                kernel variants that support it. ``None`` preserves each kernel's own
                default.
        """
        super().__init__()
        self.token_z = token_z
        self.dropout = dropout
        trimul_variant = TriMulVariant(trimul_variant)
        self.tri_mul_out = make(
            TRIMUL_REGISTRY,
            trimul_variant,
            dim=token_z,
            mode="outgoing",
            tile_size=trimul_tile_size,
            matmul_precision=trimul_matmul_precision,
        )
        self.tri_mul_in = make(
            TRIMUL_REGISTRY,
            trimul_variant,
            dim=token_z,
            mode="incoming",
            tile_size=trimul_tile_size,
            matmul_precision=trimul_matmul_precision,
        )
        self.transition_z = Transition(
            dim=token_z,
            hidden=token_z * 4,
            chunk_size=transition_chunk_size,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply triangle multiplication and transition.

        Args:
            s: Single representation ``[B, T, token_s]`` (passed through unchanged).
            z: Pair representation ``[B, T, T, token_z]``.
            pair_mask: Pair mask ``[B, T, T]``.

        Returns:
            Updated ``(s, z)``.
        """
        # Compute pairwise stack
        z = z + apply_axiswise_dropout(
            self.dropout, self.tri_mul_out(z, mask=pair_mask), self.training
        )
        z = z + apply_axiswise_dropout(
            self.dropout, self.tri_mul_in(z, mask=pair_mask), self.training
        )

        z = z + self.transition_z(z)

        return s, z


class PairMixerMSAModule(nn.Module):
    """
    PairMixer variant of the MSA module removing all triangular attention operations.
    """

    def __init__(
        self,
        token_s: int,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        use_paired_feature: bool = False,
        trimul_variant: TriMulVariant | str = TriMulVariant.CUEQ,
        trimul_tile_size: int | None = None,
        transition_chunk_size: int | None = None,
        opm_variant: OPMVariant | str = OPMVariant.VANILLA,
        opm_tile_size: int | None = None,
        pwa_chunk_heads: bool = False,
        opm_matmul_precision: str | None = None,
        trimul_matmul_precision: str | None = None,
        num_tokens: int = NUM_TOKENS,
    ) -> None:
        """Initialize the MSA module.

        Tile/chunk sizes are stored on the layers at construction time and
        bypassed automatically during training (see :meth:`forward`).

        Args:
            token_s: Single representation dimension.
            msa_s: MSA embedding size.
            token_z: Pair representation dimension.
            s_input_dim: Input sequence feature dimension fed to ``s_proj``.
            msa_blocks: Number of stacked MSA layers.
            msa_dropout: Dropout applied to the MSA representation.
            z_dropout: Dropout applied to the pair representation.
            use_paired_feature: Concatenate the MSA-paired indicator into the MSA
                input embedding.
            trimul_variant: Triangle multiplication variant (``"full"``,
                ``"cueq"``, ``"cueq_tiled"``).
            trimul_tile_size: Tile size for the ``cueq_tiled`` triangle
                multiplication variant. ``None`` uses the variant default;
                ignored by other variants.
            transition_chunk_size: Hidden-dim chunk size baked into every MSA-side
                ``Transition``. ``None`` disables chunking. Bypassed during
                training.
            opm_variant: Outer-product-mean variant (``"vanilla"`` or
                ``"triton_memory_saving"``).
            opm_tile_size: Spatial tile size for the triton_memory_saving OPM kernel
                (ignored by ``vanilla``).
            pwa_chunk_heads: Chunk pair-weighted-averaging across heads to reduce
                peak memory at inference. Bypassed during training.
            opm_matmul_precision: Matmul precision override forwarded to OPM /
                TriMul kernel variants that support it. ``None`` preserves each
                kernel's own default.
            trimul_matmul_precision: Matmul precision override forwarded to OPM /
                TriMul kernel variants that support it. ``None`` preserves each
                kernel's own default.
        """
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.pwa_chunk_heads = pwa_chunk_heads

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        self.msa_proj = nn.Linear(
            num_tokens + 2 + int(use_paired_feature),
            msa_s,
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                PairMixerMSALayer(
                    token_s,
                    msa_s,
                    token_z,
                    msa_dropout,
                    z_dropout,
                    trimul_variant=trimul_variant,
                    trimul_tile_size=trimul_tile_size,
                    transition_chunk_size=transition_chunk_size,
                    opm_variant=opm_variant,
                    opm_tile_size=opm_tile_size,
                    opm_matmul_precision=opm_matmul_precision,
                    trimul_matmul_precision=trimul_matmul_precision,
                )
                for _ in range(msa_blocks)
            ]
        )

    def forward(
        self,
        s: torch.Tensor | None,  # (B, T, token_s) or None
        z: torch.Tensor,  # (B, T, T, token_z)
        emb: torch.Tensor,  # (B, T, s_input_dim)
        msa: torch.Tensor,  # (B, R, T, msa_feat)
        msa_mask: torch.Tensor,  # (B, R, T)
        has_deletion: torch.Tensor,  # (B, R, T)
        deletion_value: torch.Tensor,  # (B, R, T)
        msa_paired: torch.Tensor,  # (B, R, T)
        token_pad_mask: torch.Tensor,  # (B, T)
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Run all MSA blocks, mixing single, pair, and MSA representations.

        Args:
            s: Single representation, ``(B, T, token_s)``, or ``None`` when
                single-representation updates are disabled.
            z: Pair representation, ``(B, T, T, token_z)``.
            emb: Input single-token embedding, ``(B, T, s_input_dim)``.
            msa: MSA features, ``(B, R, T, msa_feat)``.
            msa_mask: MSA row validity mask, ``(B, R, T)``.
            has_deletion: Per-position deletion flag, ``(B, R, T)``.
            deletion_value: Per-position deletion value, ``(B, R, T)``.
            msa_paired: Per-position paired flag, ``(B, R, T)``.
            token_pad_mask: Token validity mask, ``(B, T)``.

        Returns:
            Updated single and pair representations.
        """
        # Tiling for pair-weighted-averaging head chunking is bypassed during training.
        chunk_heads_pwa = False if self.training else self.pwa_chunk_heads

        # Expand per-position features to include a trailing dimension.
        has_del = has_deletion.unsqueeze(-1)  # (B, R, T, 1)
        del_val = deletion_value.unsqueeze(-1)  # (B, R, T, 1)
        is_paired = msa_paired.unsqueeze(-1)  # (B, R, T, 1)
        token_mask = token_pad_mask.unsqueeze(-1) & token_pad_mask.unsqueeze(-2)

        # Compute MSA embeddings
        if self.use_paired_feature:
            m = torch.cat([msa, has_del, del_val, is_paired], dim=-1)
        else:
            m = torch.cat([msa, has_del, del_val], dim=-1)

        # Compute input projections
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)

        # Perform MSA blocks
        for i in range(self.msa_blocks):
            s, z, m = self.layers[i](
                s,
                z,
                m,
                token_mask,
                msa_mask,
                chunk_heads_pwa,
            )
        return s, z


class PairMixerMSALayer(nn.Module):
    """
    PairMixer variant of basic MSA module removing triangular attention updates.
    """

    def __init__(
        self,
        token_s: int,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        trimul_variant: TriMulVariant | str = TriMulVariant.CUEQ,
        trimul_tile_size: int | None = None,
        transition_chunk_size: int | None = None,
        opm_variant: OPMVariant | str = OPMVariant.VANILLA,
        opm_tile_size: int | None = None,
        opm_matmul_precision: str | None = None,
        trimul_matmul_precision: str | None = None,
    ) -> None:
        """Initialize the MSA layer.

        Tile/chunk sizes are stored at construction time and bypassed
        automatically during training.

        Args:
            token_s: Single representation dimension.
            msa_s: MSA embedding size.
            token_z: Pair representation dimension.
            msa_dropout: Dropout applied to the MSA representation.
            z_dropout: Dropout applied to the pair representation.
            trimul_variant: Triangle multiplication variant (``"full"``,
                ``"cueq"``, ``"cueq_tiled"``).
            trimul_tile_size: Tile size for the ``cueq_tiled`` triangle
                multiplication variant. ``None`` uses the variant default;
                ignored by other variants.
            transition_chunk_size: Hidden-dim chunk size baked into the MSA and
                pair ``Transition`` modules. ``None`` disables chunking.
            opm_variant: Outer-product-mean variant (``"vanilla"`` or
                ``"triton_memory_saving"``).
            opm_tile_size: Spatial tile for the triton_memory_saving OPM kernel.
                When ``None`` the kernel default is used. Ignored by
                ``vanilla``.
            opm_matmul_precision: Matmul precision override forwarded to OPM /
                TriMul kernel variants that support it. ``None`` preserves each
                kernel's own default.
            trimul_matmul_precision: Matmul precision override forwarded to OPM /
                TriMul kernel variants that support it. ``None`` preserves each
                kernel's own default.
        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.msa_transition = Transition(
            dim=msa_s,
            hidden=msa_s * 4,
            chunk_size=transition_chunk_size,
        )
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )
        trimul_variant = TriMulVariant(trimul_variant)
        self.tri_mul_out = make(
            TRIMUL_REGISTRY,
            trimul_variant,
            mode="outgoing",
            dim=token_z,
            tile_size=trimul_tile_size,
            matmul_precision=trimul_matmul_precision,
        )
        self.tri_mul_in = make(
            TRIMUL_REGISTRY,
            trimul_variant,
            mode="incoming",
            dim=token_z,
            tile_size=trimul_tile_size,
            matmul_precision=trimul_matmul_precision,
        )
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
            chunk_size=transition_chunk_size,
        )
        opm_variant = OPMVariant(opm_variant)
        self.outer_product_mean = make(
            OPM_REGISTRY,
            opm_variant,
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
            tile_size=opm_tile_size,
            matmul_precision=opm_matmul_precision,
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        m: torch.Tensor,
        token_mask: torch.Tensor,
        msa_mask: torch.Tensor,
        chunk_heads_pwa: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mix MSA, pair, and (optionally) single representations.

        Args:
            s: Single representation ``[B, T, token_s]``.
            z: Pair representation ``[B, T, T, token_z]``.
            m: MSA representation ``[B, S, T, msa_s]``.
            token_mask: Token-pair mask ``[B, T, T]``.
            msa_mask: MSA mask ``[B, S, T]``.
            chunk_heads_pwa: When ``True``, chunk the pair-weighted-averaging
                head dimension to reduce peak memory at inference. Resolved by
                the parent module from ``self.pwa_chunk_heads`` and the training
                flag, so callers do not need to set this directly.

        Returns:
            Updated ``(s, z, m)``.
        """
        # Communication to MSA stack
        m = m + apply_axiswise_dropout(
            self.msa_dropout,
            self.pair_weighted_averaging(m, z, token_mask, chunk_heads_pwa),
            self.training,
        )
        m = m + self.msa_transition(m)

        # Communication to pairwise stack
        z = z + self.outer_product_mean(m, msa_mask)

        # Compute pairwise stack
        z = z + apply_axiswise_dropout(
            self.z_dropout, self.tri_mul_out(z, mask=token_mask), self.training
        )
        z = z + apply_axiswise_dropout(
            self.z_dropout, self.tri_mul_in(z, mask=token_mask), self.training
        )

        z = z + self.z_transition(z)

        return s, z, m
