# %% [markdown]
# # Sample a glycoprotein with branched N-glycans
#
# This notebook builds a glycosylated protein from its amino-acid sequence,
# attaches branched glycans with explicit glycosidic bonds, samples equilibrium
# conformations, and compares them with a crystal structure.
#
# The example uses the Fc fragment of human immunoglobulin G1 (IgG1), a
# homodimer of two 223-residue chains. Each chain carries an N-linked glycan at
# Asn297 that fills the space between the two CH2 domains. The crystal
# structure, PDB [1H3X](https://www.rcsb.org/structure/1H3X), resolves both
# glycans in their core-fucosylated, agalactosylated form (G0F):
#
# ```text
#   GlcNAc(β1-2)Man(α1-6)
#                        \
#                         Man(β1-4)GlcNAc(β1-4)GlcNAc(β1-N)Asn297
#                        /                       |
#   GlcNAc(β1-2)Man(α1-3)                Fuc(α1-6)
# ```
#
# The glycan branches at the β-mannose into the α1-3 and α1-6 arms, and at the
# first GlcNAc, which carries the core fucose.
#
# > [!NOTE]
# > Branched entities are absent from the current model training data, so the
# > generated glycan conformations are extrapolations. Check their geometry, as
# > shown in this notebook, before relying on them.
#
# Related guides: [systems](https://github.com/microsoft/bioemu2/blob/main/docs/system.md) and [sampling](https://github.com/microsoft/bioemu2/blob/main/docs/sampling.md).

# %% [markdown]
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %%
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np

from bioemu2 import (
    Atom,
    Bond,
    Branched,
    EmuSession,
    ModifiedResidue,
    Protein,
    System,
    load_model,
    save_frames,
)
from bioemu2.visualization import view_trajectory

# Missing optional tuning caches use cuEquivariance's default kernel settings.
logging.getLogger("cuequivariance_ops.triton.cache_manager").addFilter(
    lambda record: not (
        record.levelno == logging.WARNING
        and record.getMessage().startswith("Error reading system-wide triton tuning cache file:")
        and record.getMessage().count("[Errno 2] No such file or directory:") == 2
    )
)

# %% [markdown]
# ## Build the glycoprotein
#
# The Fc sequence below covers residues 225–447 of the IgG1 heavy chain (UniProt
# [P01857](https://www.uniprot.org/uniprotkb/P01857)) in EU numbering, the
# convention used for antibody constant domains. BioEmu2 numbers residues from 1,
# so EU residue `n` has `res_id` `n - 224`. Both chains share one `Protein`
# entity.

# %%
fc_sequence = (
    "TCPPCPAPELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPQVKFNWYVDGVQVHNAKTKPREQQYNSTYRVVS"
    "VLTVLHQNWLDGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSREEMTKNQVSLTCLVKGFYPSDIAVEWESN"
    "GQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMHEALHNHYTQKSLSLSPGK"
)
EU_OFFSET = 224


def eu(residue_number: int) -> int:
    """Convert an EU residue number to a 1-based BioEmu2 `res_id`."""
    return residue_number - EU_OFFSET


fc = Protein(sequence=fc_sequence, asym_ids=["A", "B"])

# %% [markdown]
# A `Branched` entity lists its monomers by Chemical Component Dictionary (CCD)
# code. The order of `mon_seq` defines the 1-based `res_id` used in the bonds
# below; it follows the monomer numbering in PDB entry 1H3X. Glycan `C` is
# attached to Fc chain `A`, and glycan `D` to chain `B`.

# %%
glycan_monomers = ["NAG", "NAG", "BMA", "MAN", "NAG", "MAN", "NAG", "FUC"]
glycans = Branched(
    mon_seq=[ModifiedResidue(ccd_code=code) for code in glycan_monomers],
    asym_ids=["C", "D"],
)

# %% [markdown]
# Unlike peptide bonds, glycosidic bonds and disulfides are not inferred from
# the sequence. Each glycosidic `Bond` connects the anomeric carbon `C1` of one
# sugar to an atom of the residue it is attached to. BioEmu2 removes the
# hydroxyl that each bonded `C1` loses on glycosidic bond formation.

# %%
# (acceptor oxygen, acceptor monomer, donor monomer) for each glycosidic bond.
glycosidic_links = [
    ("O4", 1, 2),  # GlcNAc(β1-4)GlcNAc
    ("O4", 2, 3),  # Man(β1-4)GlcNAc
    ("O3", 3, 4),  # Man(α1-3)Man
    ("O2", 4, 5),  # GlcNAc(β1-2)Man, α1-3 arm
    ("O6", 3, 6),  # Man(α1-6)Man
    ("O2", 6, 7),  # GlcNAc(β1-2)Man, α1-6 arm
    ("O6", 1, 8),  # Fuc(α1-6)GlcNAc
]


def atom(name: str, res_name: str, res_id: int, asym_id: str) -> Atom:
    return Atom(name=name, res_name=res_name, res_id=res_id, asym_id=asym_id)


def glycan_bonds(protein_chain: str, glycan_chain: str) -> list[Bond]:
    """Return the Asn297 attachment and the glycosidic bonds of one glycan."""
    bonds = [
        Bond(
            atom1=atom("ND2", "ASN", eu(297), protein_chain),
            atom2=atom("C1", "NAG", 1, glycan_chain),
        )
    ]
    for oxygen, acceptor, donor in glycosidic_links:
        bonds.append(
            Bond(
                atom1=atom(oxygen, glycan_monomers[acceptor - 1], acceptor, glycan_chain),
                atom2=atom("C1", glycan_monomers[donor - 1], donor, glycan_chain),
            )
        )
    return bonds


def disulfide(chain1: str, residue1: int, chain2: str, residue2: int) -> Bond:
    return Bond(
        atom1=atom("SG", "CYS", eu(residue1), chain1),
        atom2=atom("SG", "CYS", eu(residue2), chain2),
    )


glycosylation = glycan_bonds("A", "C") + glycan_bonds("B", "D")
disulfides = [
    # Intra-chain disulfides in the CH2 and CH3 domains.
    *(disulfide(chain, 261, chain, 321) for chain in "AB"),
    *(disulfide(chain, 367, chain, 425) for chain in "AB"),
    # Inter-chain disulfides in the hinge.
    disulfide("A", 226, "B", 226),
    disulfide("A", 229, "B", 229),
]

system = System(entities=[fc, glycans], bonds=glycosylation + disulfides)
print(system.topology)

# %% [markdown]
# ## Sample equilibrium conformations
#
# The session computes the protein MSA on first use. The glycans do not need
# an MSA.

# %%
model = load_model()
session = EmuSession(system, model=model, mode="equilibrium")
samples: mdtraj.Trajectory = session.sample(num_samples=10, seed=0)
samples

# %% [markdown]
# ### Visualize the samples
#
# The protein is shown as a cartoon and the glycans as sticks.

# %%
view = view_trajectory(samples, max_frames=8)
view.show()

# %% [markdown]
# ## Compare with the crystal structure
#
# Load PDB entry 1H3X and pair heavy atoms by chain, residue, and atom name.
# The crystal structure uses EU numbering for the protein and lacks the
# disordered hinge residues 225–237.

# %%
reference = mdtraj.load_pdb("https://files.rcsb.org/download/1H3X.pdb")
NUM_PROTEIN_CHAINS = 2
NUM_CHAINS = 4


def heavy_atom_keys(
    topology: mdtraj.Topology, residue_id: Callable[[mdtraj.core.topology.Chain, mdtraj.core.topology.Residue, int], int]
) -> dict[tuple[int, int, str], int]:
    """Map (chain, BioEmu2 res_id, atom name) to atom index for the protein and glycans."""
    keys = {}
    for chain in list(topology.chains)[:NUM_CHAINS]:
        for position, residue in enumerate(chain.residues, start=1):
            for atom_ in residue.atoms:
                if atom_.element.symbol != "H":
                    keys[(chain.index, residue_id(chain, residue, position), atom_.name)] = atom_.index
    return keys


sample_keys = heavy_atom_keys(samples.topology, lambda chain, residue, position: position)
reference_keys = heavy_atom_keys(
    reference.topology,
    lambda chain, residue, position: (
        eu(residue.resSeq) if chain.index < NUM_PROTEIN_CHAINS else residue.resSeq
    ),
)
shared_keys = sorted(sample_keys.keys() & reference_keys.keys(), key=sample_keys.get)
matched_samples = samples.atom_slice([sample_keys[key] for key in shared_keys])
matched_reference = mdtraj.Trajectory(
    reference.xyz[:, [reference_keys[key] for key in shared_keys]],
    topology=matched_samples.topology,
)
matched_index = {key: index for index, key in enumerate(shared_keys)}
print(f"Matched {len(shared_keys)} heavy atoms.")


def select(chain: int, res_ids: range | list[int], names: set[str] | None = None) -> list[int]:
    """Return matched atom indices in one chain, optionally restricted to atom names."""
    wanted = set(res_ids)
    return [
        index
        for (key_chain, res_id, name), index in matched_index.items()
        if key_chain == chain and res_id in wanted and (names is None or name in names)
    ]


# %% [markdown]
# Superpose every sample on the crystal structure using protein Cα atoms. The
# protein Cα RMSD measures how closely the Fc fold matches the crystal.

# %%
ca_atoms = [
    index
    for chain in range(NUM_PROTEIN_CHAINS)
    for index in select(chain, range(1, len(fc_sequence) + 1), {"CA"})
]
matched_samples.superpose(matched_reference, atom_indices=ca_atoms)
protein_rmsd = 10 * mdtraj.rmsd(matched_samples, matched_reference, atom_indices=ca_atoms)
print(f"Median protein Cα RMSD: {np.median(protein_rmsd):.2f} Å")

# %% [markdown]
# ## Glycan–protein interactions
#
# For each protein residue, compute the fraction of samples in which any heavy
# atom of a glycan is within 4 Å of the residue. Each glycan can touch residues
# in either Fc chain, so the fraction is pooled over both chains: a value of 1
# means that the residue touches a glycan in every sample, in both chains. Each
# contact is attributed to the sugar closest to the residue.

# %%
CONTACT_CUTOFF_NM = 0.4
PROTEIN_CHAINS = (0, 1)
GLYCAN_CHAINS = (2, 3)
ASN297_EU = 297
sugar_ids = range(1, len(glycan_monomers) + 1)
protein_res_ids = range(1, len(fc_sequence) + 1)


def residue_index(chain: int, res_id: int) -> int:
    """Return the topology index of a residue in the sampled trajectory."""
    return samples.topology.chain(chain).residue(res_id - 1).index


pairs = [
    (residue_index(glycan_chain, sugar), residue_index(protein_chain, res_id))
    for protein_chain in PROTEIN_CHAINS
    for res_id in protein_res_ids
    for glycan_chain in GLYCAN_CHAINS
    for sugar in sugar_ids
]
distances, _ = mdtraj.compute_contacts(samples, contacts=pairs, scheme="closest-heavy")
distances = distances.reshape(
    samples.n_frames,
    len(PROTEIN_CHAINS),
    len(protein_res_ids),
    len(GLYCAN_CHAINS) * len(sugar_ids),
)
in_contact = distances.min(axis=-1) < CONTACT_CUTOFF_NM
nearest_sugar = distances.argmin(axis=-1) % len(sugar_ids)

# contact_fraction[s, r] is the fraction of (sample, chain) pairs in which residue
# r touches a glycan and sugar s is the nearest one. Summing over sugars gives the
# total contact fraction of the residue, which lies between 0 and 1.
contact_fraction = np.stack(
    [(in_contact & (nearest_sugar == s)).mean(axis=(0, 1)) for s in range(len(sugar_ids))]
)

# %% [markdown]
# Plot the contact fraction of every CH2 residue, coloured by the nearest sugar.
# Sugars in the α1-3 arm are blue, sugars in the α1-6 arm are orange, the core is
# grey, and the core fucose is green. The yellow triangle below the axis marks
# Asn297, where each glycan is attached.

# %%
sugar_colors = {
    1: "#3b3a39",  # NAG1, attached to Asn297
    2: "#797775",  # NAG2
    3: "#b3b0ad",  # BMA3
    4: "#005a9e",  # MAN4, α1-3 arm
    5: "#6cb8f6",  # NAG5, α1-3 arm
    6: "#ca5010",  # MAN6, α1-6 arm
    7: "#f7a17a",  # NAG7, α1-6 arm
    8: "#498205",  # FUL8, core fucose
}
# The glycans sit between the two CH2 domains, so the plot shows only CH2.
CH2_EU_RANGE = (238, 340)
eu_numbers = np.array(protein_res_ids) + EU_OFFSET

fig, contact_axis = plt.subplots(figsize=(16, 5))
bottom = np.zeros(len(protein_res_ids))
for sugar in sugar_ids:
    contact_axis.bar(
        eu_numbers,
        contact_fraction[sugar - 1],
        bottom=bottom,
        width=1.0,
        color=sugar_colors[sugar],
        label=f"{glycan_monomers[sugar - 1]}{sugar}",
    )
    bottom += contact_fraction[sugar - 1]

contact_axis.plot(
    ASN297_EU,
    -0.03,
    marker="^",
    markersize=10,
    color="#fde300",
    markeredgecolor="#986f0b",
    linestyle="none",
    clip_on=False,
    zorder=5,
    label="Glycan anchor (N297)",
)
contact_axis.set_ylim(0, 1.05)
contact_axis.set_xlim(CH2_EU_RANGE[0] - 0.5, CH2_EU_RANGE[1] + 0.5)
contact_axis.set_xlabel("CH2 residue")
contact_axis.set_ylabel("Contact fraction")
contact_axis.set_title("Glycan–protein contacts (heavy atoms within 4 Å), coloured by nearest sugar")
contact_axis.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
fig.tight_layout()
plt.show()

# %% [markdown]
# Residues next to Asn297 touch the core GlcNAc in almost every sample. Contacts
# farther along the sequence show where the arms pack against the CH2 surface,
# for example the α1-6 arm (orange) around Phe241 and Phe243, as in crystal
# structures of IgG1 Fc. Low fractions indicate transient contacts of flexible
# glycan arms.

# %% [markdown]
# ## Check glycosidic bond lengths
#
# Glycan conformations are extrapolated, so confirm that every declared
# linkage stays covalently bonded. Glycosidic C–O and C–N bonds are about
# 1.4–1.5 Å long. Lengths from both glycans are pooled.


# %%
def atom_index(topology: mdtraj.Topology, bond_atom: Atom) -> int:
    """Return the topology index of a `bioemu2.Atom`."""
    chain = topology.chain(system.chain_ids.index(bond_atom.asym_id))
    residue = chain.residue(bond_atom.res_id - 1)
    return next(a.index for a in residue.atoms if a.name == bond_atom.name)


bond_pairs = [
    (atom_index(samples.topology, bond.atom1), atom_index(samples.topology, bond.atom2))
    for bond in glycosylation
]
bond_lengths = 10 * mdtraj.compute_distances(samples, bond_pairs)
links_per_glycan = len(glycosylation) // 2
for link, bond in enumerate(glycosylation[:links_per_glycan]):
    lengths = bond_lengths[:, [link, link + links_per_glycan]]
    label = (
        f"{bond.atom1.res_name}:{bond.atom1.name}"
        f"–{bond.atom2.res_name}{bond.atom2.res_id}:{bond.atom2.name}"
    )
    print(f"{label:<22} median {np.median(lengths):.2f} Å, max {lengths.max():.2f} Å")

# %% [markdown]
# ## Save the generated structures
#
# `save_frames` writes the generated coordinates to `samples.xtc` and the
# matching molecular topology, including the glycans, to `topology.cif`.

# %%
output_dir = Path("outputs/sample_a_glycoprotein")
output_path = save_frames(samples, output_dir)
fig.savefig(output_dir / "glycan_contacts.png", dpi=150, bbox_inches="tight")
print(output_path)
