"""Trajectory simulation by iterative propagation with the dynamics model."""

import logging
import math
from collections import Counter
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import mdtraj
import numpy as np
import torch
from tqdm.auto import trange

from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.data_interface import Contact, System
from bioemu2.denoising.rigid_align import weighted_rigid_align
from bioemu2.denoising.samplers import EDMSampler, GuidedFKSampler, Sampler
from bioemu2.model import InferenceModel
from bioemu2.msa.alignment import get_msa
from bioemu2.physicality import (
    PhysicalityFilterMode,
    _select_expected_extra_bonds,
    check_allatom_physicality,
)
from bioemu2.sampling import (
    SamplerPreset,
    expected_extra_bonds_from_system,
    generate_samples,
    rng_seed,
    sampler_from_preset,
)
from bioemu2.structure.preparation import prepare_sampling_inputs
from bioemu2.trajectory_io import (
    coords_angstrom_to_trajectory,
    map_cif_atom_order,
    trajectory_to_coords_angstrom,
)

logger = logging.getLogger(__name__)


@cache
def _warn_custom_step_scale(step_scale: float) -> None:
    logger.warning(
        "Propagation normally uses step_scale=1.0; using the supplied "
        "step_scale=%s unchanged. This warning will not be shown again for this scale.",
        step_scale,
    )


def simulate_trajectory(
    system: System | None = None,
    *,
    model: InferenceModel[TrunkEmbeddings],
    x0: mdtraj.Trajectory | str | Path,
    lag_time_ns: float,
    num_trajectory_steps: int,
    contacts: list[Contact] | None = None,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    seed: int | None = None,
    display_propagation_progress: bool = True,
) -> list[mdtraj.Trajectory]:
    """Simulate time-ordered trajectories with a loaded model and acquired MSA.

    Each starting frame produces an independent trajectory including its
    initial frame. Pass a CIF path as ``x0`` to infer the system, or pass a
    ``System`` with an MDTraj trajectory. The CIF must contain all system atoms;
    inferred systems are intended for simple BioEmu2 IID sampling outputs.
    ColabFold MSA results are cached on disk. To customize the sampler or
    supply an explicit MSA, embed the system and use ``simulate_trajectory_with_embeddings``.

    Args:
        system: Molecular system; inferred from a CIF path when omitted.
        model: Loaded inference model with propagation support.
        x0: Starting frames with the system's topology, or a CIF path.
        lag_time_ns: Positive time lag in nanoseconds.
        num_trajectory_steps: Positive number of propagation steps.
        contacts: Optional contact conditioning.
        batch_size: Maximum trajectories per denoiser call.
        sampler: Sampling preset; ``"edm"`` or ``"edm_gd"``. Resampling
            presets are not supported for multi-step trajectories.
        seed: Optional Torch random seed for the complete rollout.
        display_propagation_progress: Whether to display propagation progress.

    Returns:
        One time-ordered MDTraj trajectory per starting frame, in input order.
    """
    if num_trajectory_steps < 1:
        raise ValueError("num_trajectory_steps must be a positive integer.")
    configured_sampler = sampler_from_preset(sampler, mode="equilibrium")
    if (
        getattr(configured_sampler, "resampling_interval", None) is not None
        or getattr(configured_sampler, "ess_threshold", None) is not None
    ):
        raise ValueError(
            "Multi-step trajectory simulation does not support samplers with resampling enabled."
        )
    if isinstance(x0, str | Path):
        if system is None:
            system = System.from_cif(x0)
        x0 = map_cif_atom_order(system, x0)
    elif system is None:
        raise ValueError("system is required when x0 is a trajectory; pass a CIF path to infer it.")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    msa = get_msa(system)
    trunk_embeddings = model.embed(system, msa=msa, mode="equilibrium", contacts=contacts)
    return simulate_trajectory_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=lag_time_ns,
        num_trajectory_steps=num_trajectory_steps,
        model=model,
        batch_size=batch_size,
        sampler=configured_sampler,
        seed=seed,
        display_propagation_progress=display_propagation_progress,
    )


def simulate_trajectory_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    num_trajectory_steps: int,
    model: InferenceModel[TrunkEmbeddings],
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    display_propagation_progress: bool = True,
) -> list[mdtraj.Trajectory]:
    """Simulate independent time-lagged trajectories with reusable embeddings.

    Each frame of ``x0`` starts one independent trajectory. Every endpoint is
    aligned to its own preceding frame, and the returned trajectories include
    their starting frame at time zero. ``lag_time_ns`` is specified in
    nanoseconds because trajectory lags are typically on that scale; returned
    frame times follow MDTraj's convention and are stored in picoseconds.
    Each returned ``mdtraj.Trajectory`` contains frames in time order, rather
    than an unordered collection of samples.

    At the end of the rollout, all frames are checked against the
    ``"high_quality"`` physicality thresholds. The result is logged, but no
    frames are removed and physicality failures do not interrupt propagation.

    Args:
        system: Molecular system.
        trunk_embeddings: Reusable output from ``model.embed`` for the system.
        x0: One or more starting conformations with the system's topology.
        lag_time_ns: Positive time lag between consecutive frames in nanoseconds.
            We recommend 1 to 1000 (1 ns to 1 µs), the range the model mostly
            saw during training. Lags longer than a few milliseconds effectively
            disable conditioning on ``x0``; use ``sample_equilibrium()`` instead.
        num_trajectory_steps: Positive number of time-lagged propagation steps.
        model: Loaded inference model.
        batch_size: Maximum trajectories per denoiser call.
        sampler: Preconfigured sampling algorithm, including its step scale.
            Samplers with resampling enabled are not supported.
        seed: Optional Torch random seed for the complete rollout.
        display_propagation_progress: Whether to display propagation progress.

    Returns:
        One time-ordered MDTraj trajectory per starting frame, in input order.

    Raises:
        ValueError: If ``num_trajectory_steps`` is not positive or propagation
            inputs are invalid.
    """
    if num_trajectory_steps < 1:
        raise ValueError("num_trajectory_steps must be a positive integer.")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    if (
        getattr(sampler, "resampling_interval", None) is not None
        or getattr(sampler, "ess_threshold", None) is not None
    ):
        raise ValueError(
            "Multi-step trajectory simulation does not support samplers with resampling enabled."
        )
    # Each step holds one frame per starting configuration in the same order.
    step_frames: list[mdtraj.Trajectory] = [x0]
    with rng_seed(seed):
        for _ in trange(
            num_trajectory_steps, desc="Propagating", disable=not display_propagation_progress
        ):
            step_frames.append(
                propagate_with_embeddings(
                    system,
                    trunk_embeddings,
                    x0=step_frames[-1],
                    lag_time_ns=lag_time_ns,
                    model=model,
                    batch_size=batch_size,
                    sampler=sampler,
                    seed=None,
                    progress=False,
                )
            )

    trajectories = join_trajectory_steps(step_frames, lag_time_ns=lag_time_ns)
    _report_trajectory_physicality(trajectories, system)
    return trajectories


def join_trajectory_steps(
    step_frames: Sequence[mdtraj.Trajectory],
    *,
    lag_time_ns: float,
) -> list[mdtraj.Trajectory]:
    """Reorganize step-batched frames into individual time-ordered trajectories.

    Input is ordered as ``[step][trajectory]``: each step contains one frame
    per trajectory in the same order, starting with the initial frames.
    Output is ordered as ``[trajectory][step]``.

    Args:
        step_frames: Starting frames followed by endpoint frames for each step.
        lag_time_ns: Positive time between consecutive steps in nanoseconds.

    Returns:
        One trajectory per frame in each step. MDTraj frame times are in
        picoseconds.

    Raises:
        ValueError: If no frames are provided, step sizes differ, or the lag is
            not finite and positive.
    """
    if len(step_frames) == 0 or step_frames[0].n_frames == 0:
        raise ValueError("step_frames must contain at least one frame.")
    num_trajectories = step_frames[0].n_frames
    if any(frames.n_frames != num_trajectories for frames in step_frames):
        counts = Counter(frames.n_frames for frames in step_frames)
        raise ValueError(
            f"Every element of step_frames must contain {num_trajectories} frames; "
            f"observed counts: {dict(counts)}."
        )
    if not math.isfinite(lag_time_ns) or lag_time_ns <= 0:
        raise ValueError("lag_time_ns must be finite and positive.")

    times_ps = 1000 * lag_time_ns * np.arange(len(step_frames))
    trajectories = [
        mdtraj.join([frames[i] for frames in step_frames]) for i in range(num_trajectories)
    ]
    for trajectory in trajectories:
        trajectory.time = times_ps
    return trajectories


def _report_trajectory_physicality(trajectories: list[mdtraj.Trajectory], system: System) -> None:
    topology = trajectories[0].topology
    protein_atom_indices = topology.select("protein")
    if len(protein_atom_indices) == 0:
        logger.info(
            "Trajectory physicality report skipped because the system has no protein atoms."
        )
        return
    expected_extra_bonds = _select_expected_extra_bonds(
        expected_extra_bonds_from_system(system, topology), protein_atom_indices, topology.n_atoms
    )

    num_physical = 0
    num_frames = 0
    violation_totals: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    for trajectory in trajectories:
        protein_trajectory = trajectory.atom_slice(protein_atom_indices)
        protein_trajectory.topology.create_standard_bonds()
        is_physical, stats, reasons = check_allatom_physicality(
            protein_trajectory,
            mode=PhysicalityFilterMode.HIGH_QUALITY,
            expected_extra_bonds=expected_extra_bonds,
        )
        num_physical += int(is_physical.sum())
        num_frames += len(trajectory)
        violation_totals.update(
            {name: int(values.sum()) for name, values in stats.items() if values.any()}
        )
        failure_reasons.update(reason for reason in reasons if reason is not None)

    message = (
        f"Trajectory physicality report: {num_physical}/{num_frames} frames passed "
        f"high_quality thresholds; violation totals: {dict(violation_totals)}"
    )
    if num_physical == num_frames:
        logger.info(message)
    else:
        logger.warning(
            f"{message}; first failure reasons: {dict(failure_reasons)}. All frames are retained."
        )


def propagate(
    system: System | None = None,
    *,
    model: InferenceModel[TrunkEmbeddings],
    x0: mdtraj.Trajectory | str | Path,
    lag_time_ns: float,
    num_samples: int | None = None,
    contacts: list[Contact] | None = None,
    batch_size: int | None = None,
    sampler: SamplerPreset = "edm",
    seed: int | None = None,
    progress: bool = True,
) -> mdtraj.Trajectory:
    """Sample time-lagged endpoints with a loaded model and acquired MSA.

    Pass a CIF path as ``x0`` to load and map its coordinates into the system's
    atom order. If ``system`` is omitted, it is inferred with ``System.from_cif``.
    CIF-only inference is intended for simple BioEmu2 IID sampling outputs,
    not arbitrary PDB structures or MD frames. Supply an explicit system for
    modified residues, SMILES ligands, branched entities, or explicit bonds,
    and pass contacts separately. Every atom required by the system must have
    coordinates; supplying a system does not repair an incomplete CIF.

    ``x0`` is the conditioning set: each endpoint is sampled conditional on a
    starting frame and the lag, then rigidly superposed on that frame. A CIF
    path loads one starting conformation. A trajectory produces one endpoint
    per frame. For a CIF path or a single-frame trajectory, pass ``num_samples``
    to draw several endpoints from that frame. Embed the system and use
    ``propagate_with_embeddings`` to configure the sampler or reuse embeddings.

    Args:
        system: Molecular system. Inferred from a CIF path when omitted;
            required when ``x0`` is an MDTraj trajectory.
        model: Loaded inference model with propagation support.
        x0: Starting conformations with the system's topology, or a CIF path.
        lag_time_ns: Positive time lag in nanoseconds. We recommend 1 to 1000
            (1 ns to 1 µs), the range the model mostly saw during training.
            Lags longer than a few milliseconds effectively disable conditioning
            on ``x0``; use ``sample_equilibrium()`` instead.
        num_samples: Number of endpoints to draw from a CIF path or single-frame
            trajectory. Omit for one endpoint per starting frame.
        contacts: Optional contact conditioning.
        batch_size: Maximum frames per denoiser call.
        sampler: Sampling preset (``"edm"``, ``"edm_gd"``, ``"edm_fk"``, or
            ``"edm_fkc"``). Resampling presets require identical starting frames.
        seed: Optional Torch random seed.
        progress: Whether to display sampling progress.

    Returns:
        One endpoint per starting frame, or ``num_samples`` endpoints from a
        single starting frame, as an MDTraj trajectory in nanometers.
    """
    if num_samples is not None and num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if isinstance(x0, str | Path):
        if system is None:
            system = System.from_cif(x0)
        x0 = map_cif_atom_order(system, x0)
    elif system is None:
        raise ValueError("system is required when x0 is a trajectory; pass a CIF path to infer it.")
    if num_samples is not None:
        if x0.n_frames != 1:
            raise ValueError("num_samples requires a CIF path or a single-frame trajectory.")
        x0 = x0[[0] * num_samples]
    configured_sampler = sampler_from_preset(sampler, mode="equilibrium")
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    msa = get_msa(system)
    trunk_embeddings = model.embed(system, msa=msa, mode="equilibrium", contacts=contacts)
    return propagate_with_embeddings(
        system,
        trunk_embeddings,
        x0=x0,
        lag_time_ns=lag_time_ns,
        model=model,
        batch_size=batch_size,
        sampler=configured_sampler,
        seed=seed,
        progress=progress,
    )


def propagate_with_embeddings(
    system: System,
    trunk_embeddings: TrunkEmbeddings,
    *,
    x0: mdtraj.Trajectory,
    lag_time_ns: float,
    model: InferenceModel[TrunkEmbeddings],
    batch_size: int | None = None,
    sampler: Sampler,
    seed: int | None = None,
    progress: bool = True,
) -> mdtraj.Trajectory:
    """Propagate starting conformations using precomputed embeddings.

    Use this operation when reusing one embedding across many propagation calls.
    ``propagate`` is the simpler entry point when embedding reuse is unnecessary.
    Each endpoint is aligned to its corresponding ``x0`` frame. Resampling
    requires identical frames within each resampling group (all frames if the
    group size is ``None``). Batch size must align with group boundaries.
    """
    if not model.supports_propagation:
        raise ValueError("The loaded checkpoint does not support propagation.")
    topology, batch = prepare_sampling_inputs(system, device=model.device)
    if len(x0) == 0:
        raise ValueError("x0 must be a non-empty mdtraj.Trajectory.")
    if x0.topology != topology:
        raise ValueError("x0 topology must exactly match the embedded system topology.")
    if not math.isfinite(lag_time_ns) or lag_time_ns <= 0:
        raise ValueError("lag_time_ns must be finite and positive.")
    input_coords = trajectory_to_coords_angstrom(x0, device=model.device, dtype=torch.float32)
    if not torch.isfinite(input_coords).all():
        raise ValueError("x0 coordinates must be finite.")
    if lag_time_ns * 1000.0 < torch.finfo(torch.float32).tiny:
        raise ValueError("lag_time_ns underflows the model's float32 lag.")
    if batch_size is None:
        batch_size = len(x0)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(sampler, EDMSampler | GuidedFKSampler) and sampler.step_scale != 1.0:
        _warn_custom_step_scale(sampler.step_scale)
    if (
        getattr(sampler, "resampling_interval", None) is not None
        or getattr(sampler, "ess_threshold", None) is not None
    ):
        # Resampling copies particles within groups of `resampling_group_size`
        # consecutive samples; every group must share one conditioning frame.
        # None resamples a whole chunk, so then all of x0 must be identical.
        group = getattr(sampler, "resampling_group_size", None) or len(x0)
        if group < len(x0) and batch_size < len(x0) and batch_size % group:
            raise ValueError(
                f"batch_size={batch_size} must be a multiple of resampling_group_size={group}."
            )
        # Check that the conditions in ``x0`` are identical and thus safe for group resampling
        for start in range(0, len(x0), group):
            block = input_coords[start : start + group]
            if not torch.equal(block, block[:1].expand_as(block)):
                raise ValueError(
                    f"x0 frames {start}:{start + len(block)} differ; with resampling every "
                    f"block of {group} consecutive frames must be identical "
                    "(all frames when resampling_group_size is None)."
                )

    # Conditions live on the sample axis: pad each frame to the model's atom dimension.
    coords_cond = input_coords.new_zeros(len(input_coords), batch.atom_pad_mask.shape[1], 3)
    coords_cond[:, : topology.n_atoms] = input_coords - input_coords.mean(dim=1, keepdim=True)

    chunks: list[mdtraj.Trajectory] = []
    with rng_seed(seed):
        for start in range(0, len(x0), batch_size):
            chunk = coords_cond[start : start + batch_size]
            n = len(chunk)
            chunk_batch = batch.replace(
                coords_cond=chunk,
                is_dynamics_conditioned=torch.ones((1, n), dtype=torch.bool, device=chunk.device),
                dt=torch.full((1, n), lag_time_ns * 1000.0, device=chunk.device),
            )
            coords = generate_samples(
                num_samples=n,
                batch_size=n,
                score_model=model,
                sampler=sampler,
                num_atoms=topology.n_atoms,
                system=system,
                batch=chunk_batch,
                trunk_embeddings=trunk_embeddings,
                progress=progress,
            )
            chunks.append(coords_angstrom_to_trajectory(coords, topology))
    trajectory = mdtraj.join(chunks)

    # Superpose each endpoint on its own input frame.
    mask = torch.ones(x0.xyz.shape[:2])
    aligned, _, _ = weighted_rigid_align(
        torch.as_tensor(trajectory.xyz), torch.as_tensor(x0.xyz), mask, mask
    )
    trajectory.xyz = aligned.numpy()
    return trajectory
