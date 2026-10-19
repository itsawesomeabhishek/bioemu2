"""Canonical BioMoaC score-model CP parity tests."""

import copy
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from bioemu2.biomoa.models.score_model import BioMoaCModel
from bioemu2.context_parallel.distributed.comm import Ring2DComm
from bioemu2.context_parallel.distributed.manager import DistributedManager
from bioemu2.context_parallel.features import (
    build_shard_msa_feats,
    build_shard_pair_feats,
    pad_feats_for_cp,
)
from bioemu2.context_parallel.models import wrap_score_model_for_cp
from bioemu2.context_parallel.models.score_model import (
    BioMoaCDistributed,
    CPTrunkEmbeddings,
)
from bioemu2.context_parallel.modules.diffusion import CPStaticContext
from bioemu2.context_parallel.utils import CPPadSizes, make_cp_pad_sizes
from bioemu2.context_parallel.utils.dtensor import PL_PAIR
from bioemu2.data_interface import Contact, ContactToken, System
from bioemu2.msa.alignment import MSA
from bioemu2.structure.batch import BatchAllAtomStructure, batch_collate
from bioemu2.structure.preparation import prepare_structure
from tests.context_parallel.helpers import (
    assert_output_placement,
    assert_replicate_values_against_rank0,
    gather_n_assert_parity_at_rank0,
    spawn_cp_test,
    spawn_cpu_cp_test,
)

# One MSA block, one PairMixer block, and one structure block accumulate up to
# two float32 ULPs.
CPU_ATOL = 2e-6
# CUDA kernels reorder float32 reductions relative to the serial PyTorch path.
GPU_ATOL = 8e-5
CPU_MESHES = pytest.mark.parametrize(
    ("dp_size", "cp_size", "batch_size", "local_world_size"),
    [(1, 2, 1, 1), (2, 2, 2, 2), (2, 2, 4, 2)],
)


def _cp_trunk(
    model: BioMoaCDistributed, padded: dict, num_steps_recycle: int = 0
) -> CPTrunkEmbeddings:
    return model.compute_trunk_embeddings(
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


def _cp_static_context(
    model: BioMoaCDistributed,
    trunk: CPTrunkEmbeddings,
    padded: dict,
    multiplicity: int,
) -> CPStaticContext:
    return model.compute_static_context(
        trunk,
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


def _build_batch(
    batch_size: int,
    device: torch.device,
    mesh: DeviceMesh,
    atoms_per_window: int,
) -> tuple[BatchAllAtomStructure, dict, CPPadSizes]:
    """Collate real structures identically on every rank and pad them for CP."""
    torch.manual_seed(42)
    structures = []
    for index, sequence in enumerate(("YDPE", "AGA", "LYSP", "VV")[:batch_size]):
        system = System.from_description(proteins=[sequence])
        chain_id = system.chain_ids[0]
        structure = prepare_structure(
            system,
            msa=MSA({chain_id: f">query\n{sequence}\n" * (index + 2)}),
            mode="structure" if index % 2 == 0 else "equilibrium",
            contacts=[
                Contact(
                    token1=ContactToken(asym_id=chain_id, res_id=1),
                    token2=ContactToken(asym_id=chain_id, res_id=2),
                    max_distance=4.0 + index,
                )
            ],
        )
        structures.append(structure)
    sizes = make_cp_pad_sizes(
        n_token=max(s.num_tokens for s in structures),
        n_atom=max(s.num_nodes for s in structures),
        n_msa=max(s.msa.shape[1] for s in structures),
        atoms_per_window=atoms_per_window,
        mesh=mesh,
    )
    batch = batch_collate(
        structures,
        window_size=sizes.n_atom_final,
    ).to(device)
    batch.coords = torch.randn_like(batch.coords)
    # Condition alternate samples on a companion frame to exercise dynamics gating.
    batch.coords_cond = torch.randn_like(batch.coords)
    batch.dt = torch.rand_like(batch.dt)
    batch.is_dynamics_conditioned = (
        torch.arange(batch.batch_size, device=device) % 2 == 0
    ).reshape_as(batch.is_dynamics_conditioned)
    for value in batch.to_dict().values():
        if isinstance(value, torch.Tensor):
            dist.broadcast(value, src=0)
    padded = pad_feats_for_cp(batch, sizes)
    padded.update(build_shard_pair_feats(batch, sizes, mesh, device=device))
    padded.update(build_shard_msa_feats(batch, sizes, mesh, device=device))
    assert batch.coords.shape[1] == sizes.n_atom_final
    return batch, padded, sizes


def run_with_context_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    batch_size: int,
    atol: float,
    serial_model: BioMoaCModel,
    serial_pair_model: BioMoaCModel,
) -> None:
    """Compare trunk, static context and reused-context denoising stage by stage."""
    serial_model = serial_model.to(device)
    serial_pair_model = serial_pair_model.to(device)
    cp_model = wrap_score_model_for_cp(copy.deepcopy(serial_model), dist_manager)
    batch, padded, sizes = _build_batch(batch_size, device, mesh, cp_model.atoms_per_window_queries)

    with torch.no_grad():
        serial_trunk = serial_model.compute_trunk_embeddings(batch)
        cp_trunk = _cp_trunk(cp_model, padded)

    assert isinstance(cp_trunk, CPTrunkEmbeddings)
    torch.testing.assert_close(cp_trunk.s_inputs, serial_trunk.s_inputs, atol=atol, rtol=0)
    torch.testing.assert_close(cp_trunk.single, serial_trunk.single, atol=atol, rtol=0)

    for key, serial_result, cp_result in [
        ("pair", serial_trunk.pair, cp_trunk.pair),
        (
            "relative_position_encoding",
            serial_trunk.relative_position_encoding,
            cp_trunk.relative_position_encoding,
        ),
        ("distogram", serial_trunk.distogram, cp_trunk.distogram),
    ]:
        assert_output_placement(cp_result, expected_placements=PL_PAIR, label=f"trunk/{key}")
        gather_n_assert_parity_at_rank0(serial_result, cp_result, atol=atol, label=f"trunk/{key}")
        assert_replicate_values_against_rank0(cp_result)

    with torch.no_grad():
        serial_context = serial_model.compute_static_context(batch, serial_trunk)
        cp_context = _cp_static_context(cp_model, cp_trunk, padded, multiplicity=1)
        cp_context_mult = _cp_static_context(cp_model, cp_trunk, padded, multiplicity=2)

    assert isinstance(cp_context, CPStaticContext)
    gather_n_assert_parity_at_rank0(
        serial_context.z, cp_context.z, atol=atol, label="static_context/z"
    )
    assert_replicate_values_against_rank0(cp_context.z)

    torch.testing.assert_close(
        cp_context.atom_encoder_context.q,
        serial_context.atom_encoder_context.q,
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        cp_context.atom_encoder_context.c,
        serial_context.atom_encoder_context.c,
        atol=atol,
        rtol=0,
    )

    gather_n_assert_parity_at_rank0(
        serial_context.atom_encoder_context.p,
        cp_context.atom_encoder_context.p,
        atol=atol,
        label="static_context/p",
    )
    assert_replicate_values_against_rank0(cp_context.atom_encoder_context.p, atol=atol)

    for prepared, multiplicity in ((cp_context, 1), (cp_context_mult, 2)):
        assert prepared.multiplicity == multiplicity
        assert isinstance(prepared.token_mask, DTensor)
        assert isinstance(prepared.token_attention_mask, DTensor)
        assert not isinstance(prepared.atom_mask, DTensor)
        assert prepared.atom_mask is padded["atom_mask"]
        assert prepared.atom_mask.shape == (batch.batch_size, sizes.n_atom_final)
        assert prepared.atom_mask.dtype == prepared.token_mask.dtype == torch.bool
        assert prepared.atom_ids_window.shape[0] == batch.batch_size * multiplicity
        assert prepared.token_mask.shape == (
            batch.batch_size * multiplicity,
            sizes.n_token_final,
        )
        gather_n_assert_parity_at_rank0(
            padded["token_idx_at_atom"]
            .repeat_interleave(multiplicity, 0)
            .unflatten(1, (-1, cp_model.atoms_per_window_queries)),
            prepared.atom_ids_window,
            atol=0,
            label="static_context/atom_ids_window",
        )
        assert_replicate_values_against_rank0(prepared.atom_ids_window)

        # Like a sampler, prepare the context once and reuse it for consecutive
        # denoising steps whose coordinates and noise levels change.
        atom_mask = batch.atom_pad_mask.repeat_interleave(multiplicity, 0).unsqueeze(-1)
        coords = batch.coords.repeat_interleave(multiplicity, 0)
        for sigma in (0.34, 0.17):
            sigmas = torch.full(
                (batch.batch_size * multiplicity,),
                sigma,
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                serial_out = serial_model.denoise(serial_context, serial_trunk, coords, sigmas)
                cp_out = cp_model.denoise(prepared, cp_trunk, coords, sigmas)
            assert cp_out.score.shape[1] == sizes.n_atom_final

            serial_score = serial_out.score * atom_mask
            assert serial_score.abs().max().item() > 0, "serial score is trivially zero"
            torch.testing.assert_close(cp_out.score * atom_mask, serial_score, atol=atol, rtol=0)
            torch.testing.assert_close(
                cp_out.denoised_coords * atom_mask,
                serial_out.denoised_coords * atom_mask,
                atol=atol,
                rtol=0,
            )
            # Both models take the same next input so parity tolerances do not compound.
            coords = serial_out.denoised_coords

    cp_pair_model = wrap_score_model_for_cp(copy.deepcopy(serial_pair_model), dist_manager)
    # Also cover sequence-embedder recycling with method conditioning embedded into pairs.
    with torch.no_grad():
        serial_pair_trunk = serial_pair_model.compute_trunk_embeddings(batch, num_steps_recycle=1)
        cp_pair_trunk = _cp_trunk(cp_pair_model, padded, num_steps_recycle=1)
    assert serial_pair_trunk.pair.abs().max().item() > 0, "serial pair is trivially zero"
    assert_output_placement(
        cp_pair_trunk.pair, expected_placements=PL_PAIR, label="method_pair_trunk/pair"
    )
    torch.testing.assert_close(cp_pair_trunk.single, serial_pair_trunk.single, atol=atol, rtol=0)
    gather_n_assert_parity_at_rank0(
        serial_pair_trunk.pair, cp_pair_trunk.pair, atol=atol, label="method_pair_trunk/pair"
    )
    assert_replicate_values_against_rank0(cp_pair_trunk.pair)


def run_full_forward_parity(
    rank: int,
    device: torch.device,
    mesh: DeviceMesh,
    ring_comm: Ring2DComm,
    dist_manager: DistributedManager,
    *,
    batch_size: int,
    atol: float,
    serial_model: BioMoaCModel,
) -> None:
    """Compare one end-to-end ``forward`` call on padded CP features."""
    serial_model = serial_model.to(device)
    cp_model = wrap_score_model_for_cp(copy.deepcopy(serial_model), dist_manager)
    batch, padded, _ = _build_batch(batch_size, device, mesh, cp_model.atoms_per_window_queries)

    sigmas = torch.full((batch.batch_size,), 0.34, dtype=torch.float32, device=device)
    with torch.no_grad():
        serial_forward = serial_model(batch.clone(), sigmas)
        cp_forward = cp_model(padded, sigmas)
    atom_mask = batch.atom_pad_mask.unsqueeze(-1)
    torch.testing.assert_close(
        cp_forward.score * atom_mask,
        serial_forward.score * atom_mask,
        atol=atol,
        rtol=0,
    )
    torch.testing.assert_close(
        cp_forward.denoised_coords * atom_mask,
        serial_forward.denoised_coords * atom_mask,
        atol=atol,
        rtol=0,
    )
    assert cp_forward.distogram is not None
    assert serial_forward.distogram is not None
    gather_n_assert_parity_at_rank0(
        serial_forward.distogram, cp_forward.distogram, atol=atol, label="forward/distogram"
    )
    assert_replicate_values_against_rank0(cp_forward.distogram)


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_with_context_parity_on_gpu(
    score_model: BioMoaCModel, method_pair_score_model: BioMoaCModel
) -> None:
    spawn_cp_test(
        partial(
            run_with_context_parity,
            batch_size=2,
            atol=GPU_ATOL,
            serial_model=score_model,
            serial_pair_model=method_pair_score_model,
        )
    )


@pytest.mark.gpu
@pytest.mark.requires_4gpu
def test_full_forward_parity_on_gpu(score_model: BioMoaCModel) -> None:
    spawn_cp_test(
        partial(run_full_forward_parity, batch_size=2, atol=GPU_ATOL, serial_model=score_model)
    )


@pytest.mark.serial
@CPU_MESHES
def test_with_context_parity_on_cpu(
    free_tcp_port: int,
    dp_size: int,
    cp_size: int,
    batch_size: int,
    local_world_size: int,
    score_model: BioMoaCModel,
    method_pair_score_model: BioMoaCModel,
) -> None:
    spawn_cpu_cp_test(
        partial(
            run_with_context_parity,
            batch_size=batch_size,
            atol=CPU_ATOL,
            serial_model=score_model,
            serial_pair_model=method_pair_score_model,
        ),
        dp_size=dp_size,
        cp_size=cp_size,
        local_world_size=local_world_size,
        master_port=str(free_tcp_port),
    )


@pytest.mark.serial
@CPU_MESHES
def test_full_forward_parity_on_cpu(
    free_tcp_port: int,
    dp_size: int,
    cp_size: int,
    batch_size: int,
    local_world_size: int,
    score_model: BioMoaCModel,
) -> None:
    spawn_cpu_cp_test(
        partial(
            run_full_forward_parity,
            batch_size=batch_size,
            atol=CPU_ATOL,
            serial_model=score_model,
        ),
        dp_size=dp_size,
        cp_size=cp_size,
        local_world_size=local_world_size,
        master_port=str(free_tcp_port),
    )
