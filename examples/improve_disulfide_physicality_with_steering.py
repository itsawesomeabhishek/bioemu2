# %% [markdown]
# # Guide sampling toward physically realistic structures
#
# Sampled structures can contain unrealistic geometry, such as stretched or
# compressed bonds and overlapping atoms. BioEmu2 can steer sampling away from
# that geometry with physicality potentials. This notebook samples the
# C-terminal NTR domain of mouse Meteorin-like protein (PDB 7TLW), first without
# steering and then with steering. Without steering, most samples fail the
# physicality checks because their disulfide bonds are too short; the steered
# samples pass. Measuring how the two helices joined by the disulfides pack
# shows that steering corrects the disulfides without changing the fold.
#
# A separate example, `tilt_a_sample_distribution_with_steering.py`, measures how well
# steering moves a sampled distribution toward a chosen target.
#
# See the [steering guide](https://github.com/microsoft/bioemu2/blob/main/docs/steering.md#choose-a-sampler)
# to choose a sampler and the [potentials guide](https://github.com/microsoft/bioemu2/blob/main/docs/potentials.md) for the default physicality
# potentials.
#
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np

from bioemu2 import (
    EDMSampler,
    EmuSession,
    GuidedFKSampler,
    PhysicalityFilterMode,
    PhysicalityReport,
    System,
    check_physicality,
    load_model,
    save_frames,
)
from bioemu2.paths import EXAMPLES_DIR
from bioemu2.physicality import DISULFIDE_BOND_WINDOW
from bioemu2.visualization import view_trajectory

# %% [markdown]
# ## Load the system
#
# The domain has two disulfide bonds, Cys20–Cys113 and Cys30–Cys135. PDB import
# does not read disulfides, so the system file declares them as bonds. Declared
# bonds are part of the sampled topology, the bond-length steering potential,
# and the physicality checks.

# %%
meteorin_like = System.from_yaml(EXAMPLES_DIR / "systems/meteorin_like_7tlw.yaml")
print(meteorin_like)

# %%
emu = EmuSession(meteorin_like, model=load_model("bioemu2-preview"), mode="equilibrium")

num_samples = 10
seed = 0
physicality_mode = PhysicalityFilterMode.NO_VIOLATIONS_ALLOWED


# %% [markdown]
# `check_physicality` checks protein bond lengths, including declared
# disulfides, steric clashes within and between chains, valence, isolated atoms,
# and duplicate atoms. It does not check ligands or remove frames; it reports
# whether each frame passed and, for failed frames, the first check that
# exceeded its threshold. This example uses
# the strictest preset, `no_violations`, which rejects a frame with any
# violation. The default `high_quality` preset tolerates some other violation
# types, but also rejects every disulfide outside the accepted S–S range.


# %%
def summarize(report: PhysicalityReport) -> list[dict[str, object]]:
    print(f"{int(report.passed.sum())}/{len(report.passed)} samples pass the physicality checks")
    return [
        {"frame": frame, "passes": bool(passed), "failure": reason}
        for frame, (passed, reason) in enumerate(zip(report.passed, report.reasons, strict=True))
    ]


def disulfide_lengths_angstrom(samples: mdtraj.Trajectory) -> np.ndarray:
    """Return SG–SG distances of the disulfide bonds, shape [num_frames, num_bonds]."""
    pairs = [
        [first.index, second.index]
        for first, second in samples.topology.bonds
        if first.name == second.name == "SG"
    ]
    return mdtraj.compute_distances(samples, pairs) * 10.0


# %% [markdown]
# ## Sample without steering
#
# `EDMSampler` runs the reverse diffusion without any potentials.

# %%
unsteered_samples: mdtraj.Trajectory = emu.sample(
    num_samples=num_samples, sampler=EDMSampler(), seed=seed
)
unsteered_report = check_physicality(unsteered_samples, physicality_mode, system=meteorin_like)
summarize(unsteered_report)

# %% [markdown]
# A disulfide S–S bond is about 2.03 Å long. Without steering, many sampled
# disulfides are much shorter, which the checks report as bad sidechain bonds or
# as clashes between the two cysteines.

# %%
disulfide_lengths_angstrom(unsteered_samples).round(2)

# %% [markdown]
# ## Sample with physicality steering
#
# `GuidedFKSampler` uses the default physicality potentials unless you pass
# `potentials`. They penalize atom overlaps within and between chains, bond
# lengths and angles outside their reference ranges, and ligand geometry errors.
# `resampling_interval=None` disables Feynman-Kac resampling, so each sample is
# steered only by gradient guidance; this matches the `"edm_gd"` sampler preset.
# Guidance evaluates the potentials and their gradients during denoising, so
# steered sampling takes longer than sampling without steering.

# %%
physicality_sampler = GuidedFKSampler(resampling_interval=None)
[type(potential).__name__ for potential in physicality_sampler.potentials]

# %%
steered_samples: mdtraj.Trajectory = emu.sample(
    num_samples=num_samples, sampler=physicality_sampler, seed=seed
)
steered_report = check_physicality(steered_samples, physicality_mode, system=meteorin_like)
summarize(steered_report)

# %%
disulfide_lengths_angstrom(steered_samples).round(2)

# %% [markdown]
# Steering reduces unrealistic geometry, but it does not guarantee that every
# sample passes. To keep only samples that pass, select frames with the report:
# `steered_samples[steered_report.passed]`.

# %% [markdown]
# ## Compare the samples
#
# The plot shows every disulfide length in both sets of samples. Filled markers
# are samples that pass the physicality checks; open markers are samples that fail.
# The dashed line marks the mean S–S length and the shaded band marks the range
# of lengths the checks accept. A sample whose disulfides fall inside the band
# can still fail another check, such as a clash.

# %%
reference_disulfide_angstrom = 2.033
fig, ax = plt.subplots(figsize=(6, 4))
for row, (samples, report) in enumerate(
    [(unsteered_samples, unsteered_report), (steered_samples, steered_report)]
):
    lengths = disulfide_lengths_angstrom(samples)
    for passed in (True, False):
        frames = report.passed == passed
        values = lengths[frames].ravel()
        jitter = np.random.default_rng(row).uniform(-0.15, 0.15, size=values.shape)
        ax.scatter(
            values,
            row + jitter,
            facecolors=f"C{row}" if passed else "none",
            edgecolors=f"C{row}",
        )
# The y-axis already names each row, so the legend only explains the marker fill
# and reference lines, using gray stand-in markers.
ax.scatter([], [], facecolors="0.4", edgecolors="0.4", label="passes")
ax.scatter([], [], facecolors="none", edgecolors="0.4", label="fails")
ax.axvspan(*DISULFIDE_BOND_WINDOW, color="0.9", zorder=0, label="accepted S–S range")
ax.axvline(reference_disulfide_angstrom, color="black", linestyle="--", label="mean S–S")
ax.set_yticks([0, 1], ["without steering", "with steering"])
ax.set_xlabel("disulfide S–S length (Å)")
ax.set_title("Meteorin-like NTR domain (7TLW) disulfide lengths")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()

# %% [markdown]
# The 3D views show the sampled structures without and with steering.

# %%
view_trajectory(unsteered_samples, max_frames=num_samples).show()

# %%
view_trajectory(steered_samples, max_frames=num_samples).show()

# %% [markdown]
# ## Check the packing of the helices
#
# Steering corrects the disulfide geometry. To check whether it also changes the
# fold, look at the two helices that the disulfides hold together. The
# disulfides join Cys20 and Cys30, at either end of the N-terminal helix
# (N-helix, residues 22–31), to Cys113 and Cys135, on either side of the
# C-terminal helix (C-helix, residues 117–129). Two quantities describe how the
# helices pack against each other:
#
# - the distance between the centers of their Cα atoms;
# - the angle between their axes, each pointing from the first to the last
#   residue of the helix.
#
# If steering moved the bonded cysteines by distorting the fold, both quantities
# would shift. The deposited 7TLW structure is shown for reference; it does not
# resolve the first 16 residues of the sampled sequence, so its residues are
# matched to the sampled sequence before measuring.

# %%
n_terminal_helix = range(22, 32)
c_terminal_helix = range(117, 130)


def helix_packing(
    structures: mdtraj.Trajectory, first_residue: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the helix center distance in Å and the interhelix angle in degrees.

    `first_residue` is the 1-based sampled-sequence number of the first residue in
    `structures`.
    """

    def helix_ca(residues: range) -> np.ndarray:
        atoms = [
            structures.topology.select(f"name CA and resid {residue - first_residue}")[0]
            for residue in residues
        ]
        return 10.0 * structures.xyz[:, atoms]

    def helix_axis(ca: np.ndarray) -> np.ndarray:
        # The first principal component of the Cα atoms runs along the helix.
        centered = ca - ca.mean(axis=1, keepdims=True)
        axis = np.linalg.svd(centered, full_matrices=False).Vh[:, 0]
        return axis * np.sign(np.sum(axis * (ca[:, -1] - ca[:, 0]), axis=-1, keepdims=True))

    first, second = helix_ca(n_terminal_helix), helix_ca(c_terminal_helix)
    distance = np.linalg.norm(first.mean(axis=1) - second.mean(axis=1), axis=-1)
    cosine = np.sum(helix_axis(first) * helix_axis(second), axis=-1)
    return distance, np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


crystal = mdtraj.load_pdb("https://files.rcsb.org/download/7TLW.pdb")
crystal_ca = crystal.atom_slice(crystal.topology.select("protein and name CA"))
crystal_sequence = "".join(residue.code for residue in crystal_ca.topology.residues)
sampled_sequence = "".join(residue.code for residue in unsteered_samples.topology.residues)
first_resolved = sampled_sequence.index(crystal_sequence) + 1

packing = {
    "without steering": helix_packing(unsteered_samples, 1),
    "with steering": helix_packing(steered_samples, 1),
    "7TLW": helix_packing(crystal_ca, first_resolved),
}
for source, (distance, angle) in packing.items():
    print(
        f"{source}: median distance {np.median(distance):.1f} Å, median angle {np.median(angle):.0f}°"
    )

# %%
packing_fig, ax = plt.subplots(figsize=(6, 4))
for source, (distance, angle) in packing.items():
    style = {"color": "black", "marker": "*", "s": 250} if source == "7TLW" else {"alpha": 0.7}
    ax.scatter(distance, angle, label=source, **style)
ax.set_xlabel("N-helix to C-helix distance (Å)")
ax.set_ylabel("angle between N-helix and C-helix axes (°)")
ax.set_title("Meteorin-like NTR domain (7TLW) helix packing")
ax.legend(fontsize=8)
packing_fig.tight_layout()

# %% [markdown]
# Both sets of samples spread over the same range of distances and angles,
# around the values of the crystal structure: steering shortens the disulfides to
# their bonded length without changing how the helices pack.

# %% [markdown]
# ## Save the samples
#
# `save_frames` writes every frame together with the physicality report.

# %%
output_dir = Path("outputs/improve_disulfide_physicality_with_steering")
for name, samples, report in [
    ("unsteered", unsteered_samples, unsteered_report),
    ("steered", steered_samples, steered_report),
]:
    output_path = save_frames(samples, output_dir / name, physicality=report)
    print(f"saved {name} samples to {output_path}")
fig.savefig(output_dir / "disulfide_lengths.png", dpi=150)
packing_fig.savefig(output_dir / "helix_packing.png", dpi=150)
