import logging
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch import nn

from bioemu2.biomoa.models.base import ScoreModelOutput
from bioemu2.biomoa.models.edm_schedule import EDMSchedule
from bioemu2.biomoa.modules.diffusion import (
    DualStreamDiffusionModule,
    StaticContext,
)
from bioemu2.biomoa.modules.embedders import DistogramModule, InputEmbedder
from bioemu2.biomoa.modules.pairmixer import PairMixerSequenceEmbedder
from bioemu2.biomoa.modules.relative_position_encoder import (
    FusedRelativePositionEncoder,
)
from bioemu2.biomoa.utils.feature_converter import convert_densebatch_to_dictionary
from bioemu2.biomoa.utils.tiling import resolve_tiling_cfg
from bioemu2.precision import matmul_precision_ctx
from bioemu2.structure import BatchAllAtomStructure

logger = logging.getLogger(__name__)

_EMBEDDINGS_FILE = "embeddings.pt"


@dataclass
class TrunkEmbeddings:
    """Sequence-trunk embeddings and derived outputs reused during denoising.

    Attributes:
        s_inputs: Input embeddings, shape ``(B, N_token, s_input_dim)``.
        single: Trunk single representations, shape ``(B, N_token, token_s)``.
        pair: Trunk pair representations, shape
            ``(B, N_token, N_token, token_z)``.
        relative_position_encoding: Pairwise relative-position features, shape
            ``(B, N_token, N_token, token_z)``.
        distogram: Predicted distance-bin logits, shape
            ``(B, N_token, N_token, num_bins)``.
    """

    s_inputs: torch.Tensor
    single: torch.Tensor
    pair: torch.Tensor
    relative_position_encoding: torch.Tensor
    distogram: torch.Tensor

    def save(self, path: Path) -> None:
        """Write these embeddings to the single file ``path`` for user-managed reuse.

        This handles only the embedding tensors; any companion metadata is the
        caller's concern. Refuse to overwrite an existing file so a saved
        embedding is never silently clobbered; pass a new path.
        """
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file {path}; pass a new path.")
        path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {f.name: getattr(self, f.name).detach().cpu() for f in fields(self)}
        torch.save(tensors, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "TrunkEmbeddings":
        """Load embeddings written by :meth:`save` from the file ``path`` onto ``device``.

        The loaded embeddings are not validated against any inputs, so load ones
        produced for the same system, MSA, contacts, mode, and model in use.
        """
        tensors = torch.load(path, map_location="cpu", weights_only=True)
        names = {f.name for f in fields(cls)}
        if (
            not isinstance(tensors, dict)
            or set(tensors) != names
            or any(not isinstance(value, torch.Tensor) for value in tensors.values())
        ):
            raise ValueError(f"Invalid trunk embeddings at {path}")
        return cls(**{name: tensors[name].to(device) for name in names})


@dataclass
class DenoiseOutput:
    """Score-model outputs produced by ``BioMoaCModel.denoise``.

    Attributes:
        score: Predicted coordinate update, shape ``(B * multiplicity, N, 3)``.
        denoised_coords: Reconstructed coordinates, shape
            ``(B * multiplicity, N, 3)``.
    """

    score: torch.Tensor
    denoised_coords: torch.Tensor


class BioMoaCModel(nn.Module):
    """BioMoa score model combining the sequence trunk with the diffusion module."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atom_attention_heads: int,
        sequence_embedder: PairMixerSequenceEmbedder,
        s_input_dim: int,
        num_bins_distogram: int,
        edm_schedule: EDMSchedule,
        a_s: int | None = None,
        a_z: int | None = None,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        token_attention_heads: int = 16,  # Note that this is only half the size of original AF3 since we run structure module with token_s as well
        max_atoms: int | None = None,
        compile_score: bool = False,
        dim_fourier: int = 256,
        diffusion_transformer_depth: int = 16,  # Note that this is only half the size of original AF3 since we do one atom and one token transformer per layer
        conditioning_transition_layers: int = 2,
        tiling_cfg: dict | None = None,
        relpos_cfg: dict | None = None,
        use_lns: bool = True,
        num_token_types: int = 33,
        num_msa_tokens: int = 33,
        num_pocket_classes: int = 4,
        use_dynamics_conditioning: bool = False,
        downscale_condition_contrib: bool = False,
        dt_embed_num_timescales: int = 32,
    ):
        """Score model using the dual-stream diffusion module with interleaved atom/token
        attention layers and an pairmixer for sequence encoding.

        Atom feature embedding is handled inside the dual stream diffusion module directly
        (no separate atom encoder).

        Args:
            atom_s: Number of atom node features.
            atom_z: Number of atom pair features.
            token_s: Number of token node features.
            token_z: Number of token pair features.
            a_s: Internal single representation dimension for the diffusion layers.
                Defaults to token_s when None.
            a_z: Internal pair representation dimension for the diffusion layers.
                Defaults to token_z when None.
            atom_attention_heads: Number of heads in the atom attention layers.
            sequence_embedder: Module for end-to-end sequence embedding.
            s_input_dim: Dimension of the ``InputEmbedder`` output
                (``token_s + num_token_types + num_msa_tokens + 1 + num_pocket_classes``).
                Set via ``${eval:...}`` in the Hydra config so all consumers agree.
            num_bins_distogram: Number of bins used for distogram.
            edm_schedule: ``EDMSchedule`` class for controlling EDM settings.
            atoms_per_window_queries: Number of atoms considered in the query window of the atom
              level transformers.
            atoms_per_window_keys: Number of atoms considered in the key window of the atom level
              transformers.
            token_attention_heads: Number of heads in the token attention layers.
            max_atoms: Optional maximum length of atom dimension. Should be larger than the number
              of atoms encountered in a batch and a multiple of ``atoms_per_window_queries``.
            compile_score: Compile the score model.
            dim_fourier: Number of features used for encoding diffusion times.
            diffusion_transformer_depth: Number of interleaved atom/token attention layers.
            conditioning_transition_layers: Number of transition layers for conditioning.
            tiling_cfg: Optional tile / chunk sizes for inference. Recognised keys:
                ``tile_size`` / ``chunk_size`` (global defaults overruling any unset
                per-key entry), ``attn_pair_bias_tile_size``, ``transition_chunk_size``.
                Independent of the sequence embedder's own ``tiling_cfg``. Tiling is
                bypassed during training.
            relpos_cfg: Optional configuration for the relative position encoder. Keys are
                ``r_max`` and ``s_max``. If None, the constructor's
                defaults will be used.
            use_lns: Whether to use LayerNorm Scaling (arXiv:2502.05795) in the structure module.
            num_token_types: Number of token types used in InputEmbedder
            num_msa_tokens: Number of MSA tokens used in InputEmbedder
            num_pocket_classes: Number of pocket classes used in InputEmbedder
            use_dynamics_conditioning: Enable MD-sequential (ITO) conditioning. The structure
                module is conditioned on a companion frame ``coords_cond`` and the time lag
                ``dt``, both gated per-sample by ``is_dynamics_conditioned``. The added conditioning
                projections are initialized to output zeros, so a fresh model matches the baseline.
            downscale_condition_contrib: Gate the per-atom ``coords_cond`` contribution by a
                learned, monotonically decreasing function of the lag ``dt``.
            dt_embed_num_timescales: Number of log-spaced timescales in the ``dt`` embeddings.
        """
        super().__init__()
        self.sequence_embedder = sequence_embedder

        resolved, _ = resolve_tiling_cfg(
            tiling_cfg,
            module_name=type(self).__name__,
        )
        attn_pair_bias_tile_size = resolved["attn_pair_bias_tile_size"]
        transition_chunk_size = resolved["transition_chunk_size"]

        # Input embeddings.
        self.input_embedder = InputEmbedder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_encoder_depth=3,
            atom_encoder_heads=4,
            no_atom_encoder=False,
            num_token_types=num_token_types,
            num_msa_tokens=num_msa_tokens,
            num_pocket_classes=num_pocket_classes,
        )
        if relpos_cfg is None:
            relpos_cfg = {}
        self.rel_pos = FusedRelativePositionEncoder(token_z=token_z, **relpos_cfg)
        self.token_bonds = nn.Linear(1, token_z, bias=False)

        # Module for distogram computation from sequence pair embeddings.
        self.distogram_module = DistogramModule(token_z, num_bins_distogram)

        # Dual stream diffusion module with interleaved atom/token attention layers.
        self.structure_module = DualStreamDiffusionModule(
            token_s=token_s,
            token_z=token_z,
            atom_s=atom_s,
            atom_z=atom_z,
            s_input_dim=s_input_dim,
            a_s=a_s,
            a_z=a_z,
            num_layers=diffusion_transformer_depth,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            sigma_data=16.0,
            dim_fourier=dim_fourier,
            atom_attention_heads=atom_attention_heads,
            token_attention_heads=token_attention_heads,
            conditioning_transition_layers=conditioning_transition_layers,
            attn_pair_bias_tile_size=attn_pair_bias_tile_size,
            transition_chunk_size=transition_chunk_size,
            use_lns=use_lns,
            use_dynamics_conditioning=use_dynamics_conditioning,
            downscale_condition_contrib=downscale_condition_contrib,
            dt_embed_num_timescales=dt_embed_num_timescales,
        )
        if compile_score:
            self.structure_module = torch.compile(
                self.structure_module, dynamic=False, fullgraph=False
            )

        self.max_atoms = max_atoms
        self.atoms_per_window_queries = atoms_per_window_queries
        self.edm_schedule = edm_schedule
        self.supports_propagation = use_dynamics_conditioning

    def forward(
        self,
        batch: BatchAllAtomStructure,
        sigmas: torch.Tensor,
        num_steps_recycle: int = 0,
    ) -> ScoreModelOutput:
        """
        EDM wrapper for model forward. See:

        Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of
          diffusion-based generative models. Advances in neural information processing systems, 35,
          26565-26577.
        """
        trunk_embeddings = self.compute_trunk_embeddings(batch, num_steps_recycle)
        context = self.compute_static_context(batch, trunk_embeddings)
        output = self.denoise(context, trunk_embeddings, batch.coords, sigmas)
        return ScoreModelOutput(
            score=output.score,
            distogram=trunk_embeddings.distogram,
            denoised_coords=output.denoised_coords,
        )

    def compute_trunk_embeddings(
        self,
        batch: BatchAllAtomStructure,
        num_steps_recycle: int = 0,
    ) -> TrunkEmbeddings:
        feats = convert_densebatch_to_dictionary(batch)
        # Create initial input embeddings.
        s_inputs = self.input_embedder(
            res_type=feats["res_type"],
            profile=feats["profile"],
            deletion_mean=feats["deletion_mean"],
            pocket_feature=feats["pocket_feature"],
            ref_pos=feats["template_coords"],
            ref_charge=feats["charge"],
            ref_element=feats["element"],
            ref_atom_name_chars=feats["name_encoding"],
            ref_space_uid=feats["ref_space_uid"],
            atom_pad_mask=feats["atom_pad_mask"],
            atom_to_token_ids=feats["token_idx_at_atom"],
            token_pad_mask=feats["token_pad_mask"],
        )
        # Compute single and pair embeddings.
        s, z = self.sequence_embedder(s_inputs, feats, num_steps_recycle=num_steps_recycle)

        # The pairwise encoding is needed only after the trunk; retaining it during
        # recycling raises peak memory substantially for large systems.
        relative_position_encoding = self.rel_pos(
            asym_id=feats["asym_id"],
            entity_id=feats["entity_id"],
            residue_index=feats["residue_index"],
            sym_id=feats["sym_id"],
        )

        # Compute the distogram based on pair embeddings.
        distogram: torch.Tensor = self.distogram_module(z)
        return TrunkEmbeddings(
            s_inputs=s_inputs,
            single=s,
            pair=z,
            relative_position_encoding=relative_position_encoding,
            distogram=distogram,
        )

    def compute_static_context(
        self,
        batch: BatchAllAtomStructure,
        trunk_embeddings: TrunkEmbeddings,
    ) -> StaticContext:
        feats = convert_densebatch_to_dictionary(batch)
        with torch.autocast("cuda", enabled=False):
            context = self.structure_module.compute_static_context(
                s_trunk=trunk_embeddings.single.float(),
                z_trunk=trunk_embeddings.pair.float(),
                feats=feats,
                relative_position_encoding=trunk_embeddings.relative_position_encoding.float(),
            )
        return context

    def denoise(
        self,
        context: StaticContext,
        trunk_embeddings: TrunkEmbeddings,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
        matmul_precision: str | None = None,
    ) -> DenoiseOutput:
        batch_size = context.token_pad_mask.shape[0]
        if coords.shape[0] % batch_size:
            raise ValueError(
                f"Coordinate batch size {coords.shape[0]} is not divisible by "
                f"context batch size {batch_size}."
            )
        if sigmas.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Sigma batch size {sigmas.shape[0]} does not match coordinate "
                f"batch size {coords.shape[0]}."
            )
        multiplicity = coords.shape[0] // batch_size
        # Unit contract: ``coords`` arrives in angstroms. EDM preconditioning makes
        # the structure-module coordinates unitless; downstream real-space
        # featurization restores the model's angstrom scale.
        preconditioned_coords = self.edm_schedule.c_in(sigmas)[:, None, None] * coords
        times = self.edm_schedule.c_noise(sigmas)

        with torch.autocast("cuda", enabled=False), matmul_precision_ctx(matmul_precision):
            score = self.structure_module.denoise(
                context=context,
                r_noisy=preconditioned_coords.float(),
                s_trunk=trunk_embeddings.single.float(),
                s_inputs=trunk_embeddings.s_inputs.float(),
                times=times.float(),
                multiplicity=multiplicity,
            )["r_update"]

        # Revert multiple-of-32 padding, i.e. 251 -pad-> 256 -reverse pad-> 251.
        score = score[:, : coords.shape[1]]

        # Reconstruct denoised coordinates according to schedule.
        denoised_coords = (
            self.edm_schedule.c_skip(sigmas)[:, None, None] * coords
            + self.edm_schedule.c_out(sigmas)[:, None, None] * score
        )
        return DenoiseOutput(score=score, denoised_coords=denoised_coords)
