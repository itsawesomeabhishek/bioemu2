# %% [markdown]
# # Sample a protein complex with a phosphorylated residue
#
# BioEmu2 can sample proteins that carry post-translational modifications. This
# notebook builds the Grb2 SH2 domain bound to a phosphotyrosine (pTyr) peptide
# from amino-acid sequences, marks the phosphorylated tyrosine as the CCD
# component `PTR`, samples equilibrium structures, and checks the recognition
# features known from the crystal structure, PDB
# [1JYR](https://www.rcsb.org/structure/1JYR).
#
# SH2 domains bind peptides only after a tyrosine has been phosphorylated. The
# phosphate forms salt bridges with two conserved arginines, and Grb2 also
# selects for an asparagine two residues after the pTyr, which makes the
# peptide fold into a β-turn.
#
# Related guides: [systems](https://github.com/microsoft/bioemu2/blob/main/docs/system.md) and [sampling](https://github.com/microsoft/bioemu2/blob/main/docs/sampling.md).
#
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np

from bioemu2 import (
    EmuSession,
    ModifiedResidue,
    Protein,
    System,
    load_model,
    save_frames,
)
from bioemu2.visualization import view_trajectory

# %% [markdown]
# ## Build the system from sequences
#
# The system has two chains: the human Grb2 SH2 domain construct from 1JYR
# (residues 56–152) and the Shc-derived peptide `APS(pY)VNVQN`. Pass a `ModifiedResidue` to the
# peptide's `mods` to replace residue 4 with phosphotyrosine (`PTR`). The
# sequence keeps the parent letter `Y` at that position, so the MSA search
# uses the unmodified sequence.

# %%
SH2_SEQUENCE = (
    "GSMAWFFGKIPRAKAEEMLSKQRHDGAFLIRESESAPGDFSLSVKFGNDVQHFKVLRDGAGKYFLWVVKFNSLNELVDYHRSTSVSRNQQIFLRDI"
)
PEPTIDE_SEQUENCE = "APSYVNVQN"
PTYR_POSITION = 4  # 1-based position in the peptide

system = System(
    entities=[
        Protein(sequence=SH2_SEQUENCE, asym_ids=["A"]),
        Protein(
            sequence=PEPTIDE_SEQUENCE,
            asym_ids=["B"],
            mods=[ModifiedResidue(res_id=PTYR_POSITION, ccd_code="PTR")],
        ),
    ]
)
print(system)

# %% [markdown]
# ## Sample equilibrium structures

# %%
emu = EmuSession(system, model=load_model(), mode="equilibrium")
samples: mdtraj.Trajectory = emu.sample(num_samples=10, seed=0)
samples

# %%
view = view_trajectory(samples, max_frames=8)
view.show()

# %% [markdown]
# ## Load the crystal structure for comparison
#
# The deposited 1JYR model contains the same two chains. Waters are removed so
# that its chains match the sampled topology.

# %%
reference = mdtraj.load_pdb("https://files.rcsb.org/download/1JYR.pdb")
reference = reference.atom_slice(reference.topology.select("chainid 0 1 and not water"))
reference

# %% [markdown]
# ## Check pTyr recognition
#
# Two geometric features summarize how the SH2 domain recognizes the
# phosphopeptide:
#
# - salt bridges between the pTyr phosphate and the conserved arginines Arg67
#   (αA2) and Arg86 (βB5). Each salt bridge is reported as the shortest
#   distance between a phosphate oxygen (`O1P`, `O2P`, `O3P`) and an arginine
#   side-chain nitrogen (`NE`, `NH1`, `NH2`). A distance below 4.0 Å counts as
#   a formed salt bridge, so the fraction of samples below that threshold is the
#   salt-bridge occupancy.
# - the Cα–Cα distance between pTyr and the residue three positions later.
#   Values below 7 Å indicate the β-turn that Grb2 selects.
#
# SH2 residue numbers follow 1JYR, whose SH2 chain starts at residue 56.

# %%
FIRST_SH2_RESIDUE = 56
SALT_BRIDGE_ARGININES = {"Arg67 (αA2)": 67, "Arg86 (βB5)": 86}
PHOSPHATE_OXYGENS = ("O1P", "O2P", "O3P")
GUANIDINIUM_NITROGENS = ("NE", "NH1", "NH2")


def recognition_distances_angstrom(trajectory: mdtraj.Trajectory) -> dict[str, np.ndarray]:
    """Return per-frame recognition distances in Å."""
    sh2, peptide = trajectory.topology.chain(0), trajectory.topology.chain(1)
    ptyr = peptide.residue(PTYR_POSITION - 1)
    phosphate = [ptyr.atom(name).index for name in PHOSPHATE_OXYGENS]

    distances = {}
    for label, residue_number in SALT_BRIDGE_ARGININES.items():
        arginine = sh2.residue(residue_number - FIRST_SH2_RESIDUE)
        nitrogens = [arginine.atom(name).index for name in GUANIDINIUM_NITROGENS]
        pairs = [[oxygen, nitrogen] for oxygen in phosphate for nitrogen in nitrogens]
        distances[f"pTyr phosphate – {label} salt bridge"] = (
            mdtraj.compute_distances(trajectory, pairs).min(axis=1) * 10.0
        )

    turn_pair = [[ptyr.atom("CA").index, peptide.residue(PTYR_POSITION + 2).atom("CA").index]]
    distances["pTyr Cα – pTyr+3 Cα"] = mdtraj.compute_distances(trajectory, turn_pair)[:, 0] * 10.0
    return distances


sampled = recognition_distances_angstrom(samples)
crystal = recognition_distances_angstrom(reference)
thresholds = {
    name: 7.0 if name == "pTyr Cα – pTyr+3 Cα" else 4.0 for name in sampled
}

[
    {
        "feature": name,
        "crystal (Å)": round(float(crystal[name][0]), 1),
        "samples median (Å)": round(float(np.median(values)), 1),
        "threshold (Å)": thresholds[name],
        "fraction of samples below threshold": round(float(np.mean(values < thresholds[name])), 2),
    }
    for name, values in sampled.items()
]

# %% [markdown]
# ## Compare the peptide with the crystal structure
#
# Superpose each sample onto the SH2 Cα atoms of 1JYR, then measure how far each
# peptide Cα lies from its crystal position. The peptide is not used for the
# superposition, so these deviations include both its placement in the binding
# site and its internal conformation.


# %%
def ca_indices(trajectory: mdtraj.Trajectory, chain_index: int) -> list[int]:
    return [atom.index for atom in trajectory.topology.chain(chain_index).atoms if atom.name == "CA"]


aligned = samples.superpose(
    reference,
    atom_indices=ca_indices(samples, 0),
    ref_atom_indices=ca_indices(reference, 0),
)
peptide_deviation = 10.0 * np.linalg.norm(
    aligned.xyz[:, ca_indices(aligned, 1)] - reference.xyz[0, ca_indices(reference, 1)],
    axis=-1,
)

labels = list(PEPTIDE_SEQUENCE)
labels[PTYR_POSITION - 1] = "pY"
positions = np.arange(len(labels))
plt.figure(figsize=(6, 3))
plt.boxplot(peptide_deviation, positions=positions)
plt.xticks(positions, labels)
plt.xlabel("Peptide residue")
plt.ylabel("Cα deviation from 1JYR (Å)")
plt.title("Peptide placement after SH2 superposition")
plt.tight_layout()
plt.show()

# %% [markdown]
# The phosphate–arginine salt bridges and the β-turn should appear in nearly
# every sample, and the residues around pTyr should stay close to their crystal
# positions. Larger deviations at the peptide termini reflect residues that
# make fewer contacts with the SH2 domain.

# %% [markdown]
# ## Save the samples

# %%
output_path = save_frames(samples, Path("outputs/sample_a_phosphorylated_complex"))
print(output_path)
