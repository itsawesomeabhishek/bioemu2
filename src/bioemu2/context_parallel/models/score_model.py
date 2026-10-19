"""Canonical context-parallel BioMoaC score model."""

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.models.base import ScoreModelOutput
from bioemu2.biomoa.models.score_model import BioMoaCModel, DenoiseOutput
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.layers.relative_position_encoder import (
    DistributedRelativePositionEncoder,
)
from bioemu2.context_parallel.modules.diffusion import (
    CPStaticContext,
    DistributedDualStreamDiffusionModule,
)
from bioemu2.context_parallel.modules.embedders import DistributedDistogramWrap
from bioemu2.context_parallel.modules.pairmixer import (
    PairMixerSequenceEmbedderDistributed,
)
from bioemu2.context_parallel.ops.distributed_primitives.linear import LinearParamsReplicated
from bioemu2.context_parallel.utils import (
    gather_along_atom,
)


@dataclass
class CPTrunkEmbeddings:
    """Sequence-trunk embeddings and derived outputs reused during denoising.

    Attributes:
        s_inputs: Input embeddings, shape ``(B, N_token, s_input_dim)``.
        single: Trunk single representations, shape ``(B, N_token, token_s)``.
        pair: Sharded trunk pair representations, global shape
            ``(B, N_token, N_token, token_z)``.
        relative_position_encoding: Sharded pairwise relative-position features,
            global shape ``(B, N_token, N_token, token_z)``.
        distogram: Sharded predicted distance-bin logits, global shape
            ``(B, N_token, N_token, num_bins)``.
    """

    s_inputs: torch.Tensor
    single: torch.Tensor
    pair: DTensor
    relative_position_encoding: DTensor
    distogram: DTensor

    def save(self, path: Path) -> None:
        raise NotImplementedError(
            "Embedding save/load is not supported for context-parallel sampling."
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "CPTrunkEmbeddings":
        raise NotImplementedError(
            "Embedding save/load is not supported for context-parallel sampling."
        )


class BioMoaCDistributed(nn.Module):
    def __init__(
        self,
        serial: BioMoaCModel,
        dist_manager: DistributedManager,
        *,
        sequence_embedder: PairMixerSequenceEmbedderDistributed,
    ) -> None:
        nn.Module.__init__(self)

        mesh = dist_manager.device_mesh_subgroups
        device = next(serial.parameters()).device

        self.sequence_embedder = sequence_embedder
        self.input_embedder = serial.input_embedder
        self.rel_pos = DistributedRelativePositionEncoder(serial.rel_pos, mesh).to(device=device)
        self.token_bonds = LinearParamsReplicated(serial.token_bonds, mesh)
        self.distogram_module = DistributedDistogramWrap(serial.distogram_module, dist_manager).to(
            device=device
        )
        self.structure_module = DistributedDualStreamDiffusionModule(
            serial.structure_module, dist_manager
        )
        self.max_atoms = serial.max_atoms
        self.atoms_per_window_queries = serial.atoms_per_window_queries
        self.edm_schedule = serial.edm_schedule

    def compute_trunk_embeddings(
        self,
        *,
        res_type: torch.Tensor,
        profile: torch.Tensor,
        deletion_mean: torch.Tensor,
        pocket_feature: torch.Tensor,
        asym_id: torch.Tensor,
        entity_id: torch.Tensor,
        residue_index: torch.Tensor,
        sym_id: torch.Tensor,
        token_pad_mask: torch.Tensor,
        token_bonds: DTensor,
        template_coords: torch.Tensor,
        charge: torch.Tensor,
        element: torch.Tensor,
        name_encoding: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        token_idx_at_atom: torch.Tensor,
        msa: DTensor,  # PL_MSA; (B, S, T, c_msa)
        has_deletion: DTensor,  # PL_MSA; (B, S, T)
        deletion_value: DTensor,  # PL_MSA; (B, S, T)
        msa_mask: DTensor,  # PL_MSA; (B, S, T)
        msa_paired: DTensor | None = None,  # PL_MSA; (B, S, T)
        method_feature: torch.Tensor | None = None,
        contact_conditioning: DTensor | None = None,
        contact_threshold: DTensor | None = None,
        num_steps_recycle: int = 0,
    ) -> CPTrunkEmbeddings:
        s_inputs = self.input_embedder(
            res_type=res_type,
            profile=profile,
            deletion_mean=deletion_mean,
            pocket_feature=pocket_feature,
            ref_pos=template_coords,
            ref_charge=charge,
            ref_element=element,
            ref_atom_name_chars=name_encoding,
            ref_space_uid=ref_space_uid,
            atom_pad_mask=atom_pad_mask,
            atom_to_token_ids=token_idx_at_atom,
            token_pad_mask=token_pad_mask,
        )
        single, pair = self.sequence_embedder(
            s_inputs,
            asym_id=asym_id,
            entity_id=entity_id,
            residue_index=residue_index,
            sym_id=sym_id,
            token_pad_mask=token_pad_mask,
            token_bonds=token_bonds,
            msa=msa,
            has_deletion=has_deletion,
            deletion_value=deletion_value,
            msa_mask=msa_mask,
            msa_paired=msa_paired,
            method_feature=method_feature,
            contact_conditioning=contact_conditioning,
            contact_threshold=contact_threshold,
            num_steps_recycle=num_steps_recycle,
        )
        # Avoid retaining relative-position features while computing single and pair embeddings.
        relative_position_encoding = self.rel_pos(
            asym_id=asym_id,
            entity_id=entity_id,
            residue_index=residue_index,
            sym_id=sym_id,
        )
        return CPTrunkEmbeddings(
            s_inputs=s_inputs,
            single=single,
            pair=pair,
            relative_position_encoding=relative_position_encoding,
            distogram=self.distogram_module(pair),
        )

    def compute_static_context(
        self,
        trunk_embeddings: CPTrunkEmbeddings,
        *,
        template_coords: torch.Tensor,
        charge: torch.Tensor,
        element: torch.Tensor,
        name_encoding: torch.Tensor,
        ref_space_uid: torch.Tensor,
        atom_pad_mask: torch.Tensor,
        atom_mask: torch.Tensor,
        token_idx_at_atom: torch.Tensor,
        token_pad_mask: torch.Tensor,
        token_center_atom_index: torch.Tensor,
        multiplicity: int,
        coords_cond: torch.Tensor | None = None,
        is_dynamics_conditioned: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
    ) -> CPStaticContext:
        # structure model is always with fp32. no casting
        with torch.autocast("cuda", enabled=False):
            results = self.structure_module.compute_static_context(
                s_trunk=trunk_embeddings.single.float(),
                z_trunk=trunk_embeddings.pair.float(),
                relative_position_encoding=trunk_embeddings.relative_position_encoding.float(),
                template_coords=template_coords,
                charge=charge,
                element=element,
                name_encoding=name_encoding,
                ref_space_uid=ref_space_uid,
                atom_pad_mask=atom_pad_mask,
                atom_mask=atom_mask,
                token_idx_at_atom=token_idx_at_atom,
                token_pad_mask=token_pad_mask,
                token_center_atom_index=token_center_atom_index,
                multiplicity=multiplicity,
                coords_cond=coords_cond,
                is_dynamics_conditioned=is_dynamics_conditioned,
                dt=dt,
            )
        return results

    def denoise(
        self,
        context: CPStaticContext,
        trunk_embeddings: CPTrunkEmbeddings,
        coords: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> DenoiseOutput:
        batch_size = context.token_pad_mask.shape[0]
        if coords.shape[0] != batch_size * context.multiplicity:
            raise ValueError(
                f"Coordinate batch size {coords.shape[0]} does not match prepared "
                f"batch size {batch_size} * multiplicity {context.multiplicity}."
            )
        if sigmas.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Sigma batch size {sigmas.shape[0]} does not match coordinate "
                f"batch size {coords.shape[0]}."
            )
        preconditioned_coords = self.edm_schedule.c_in(sigmas)[:, None, None] * coords
        with torch.autocast("cuda", enabled=False):
            score = self.structure_module.denoise(
                context=context,
                s_inputs=trunk_embeddings.s_inputs.float(),
                s_trunk=trunk_embeddings.single.float(),
                r_noisy=preconditioned_coords.float(),
                times=self.edm_schedule.c_noise(sigmas).float(),
                multiplicity=context.multiplicity,
            )["r_update"]
        score = gather_along_atom([score])[0]
        denoised_padded = (
            self.edm_schedule.c_skip(sigmas)[:, None, None] * coords
            + self.edm_schedule.c_out(sigmas)[:, None, None] * score
        )
        return DenoiseOutput(
            score=score,
            denoised_coords=denoised_padded,
        )

    def forward(
        self,
        padded: dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        num_steps_recycle: int = 0,
        multiplicity: int = 1,
    ) -> ScoreModelOutput:
        trunk_embeddings = self.compute_trunk_embeddings(
            res_type=padded["res_type"],
            profile=padded["profile"],
            deletion_mean=padded["deletion_mean"],
            pocket_feature=padded["pocket_feature"],
            asym_id=padded["asym_id"],
            entity_id=padded["entity_id"],
            residue_index=padded["residue_index"],
            sym_id=padded["sym_id"],
            token_pad_mask=padded["token_pad_mask"],
            token_bonds=padded["token_bonds"],
            template_coords=padded["template_coords"],
            charge=padded["charge"],
            element=padded["element"],
            name_encoding=padded["name_encoding"],
            ref_space_uid=padded["ref_space_uid"],
            atom_pad_mask=padded["atom_pad_mask"],
            token_idx_at_atom=padded["token_idx_at_atom"],
            msa=padded["msa"],
            has_deletion=padded["has_deletion"],
            deletion_value=padded["deletion_value"],
            msa_mask=padded["msa_mask"],
            msa_paired=padded.get("msa_paired"),
            method_feature=padded.get("method_feature"),
            contact_conditioning=padded.get("contact_conditioning"),
            contact_threshold=padded.get("contact_threshold"),
            num_steps_recycle=num_steps_recycle,
        )
        context = self.compute_static_context(
            trunk_embeddings,
            template_coords=padded["template_coords"],
            charge=padded["charge"],
            element=padded["element"],
            name_encoding=padded["name_encoding"],
            ref_space_uid=padded["ref_space_uid"],
            atom_pad_mask=padded["atom_pad_mask"],
            atom_mask=padded["atom_mask"],
            token_idx_at_atom=padded["token_idx_at_atom"],
            token_pad_mask=padded["token_pad_mask"],
            token_center_atom_index=padded["token_center_atom_index"],
            multiplicity=multiplicity,
            coords_cond=padded.get("coords_cond"),
            is_dynamics_conditioned=padded.get("is_dynamics_conditioned"),
            dt=padded.get("dt"),
        )
        output = self.denoise(context, trunk_embeddings, padded["coords"], sigmas)
        return ScoreModelOutput(
            score=output.score,
            distogram=trunk_embeddings.distogram,
            denoised_coords=output.denoised_coords,
        )


__all__ = [
    "BioMoaCDistributed",
    "CPTrunkEmbeddings",
]
