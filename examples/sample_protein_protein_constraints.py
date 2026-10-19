# %% [markdown]
# # Using protein-protein contacts to bias complex predictions towards specific binding interfaces
#
# This example uses a binder for [1TEN](https://www.rcsb.org/structure/1TEN) generated in a short
# [ColabDesign](https://github.com/sokrypton/ColabDesign) run. The aim is to demonstrate
# contact conditioning, not to evaluate or optimize a binder.
#
# Epitope knowledge, existing structural templates, or cross-linking mass spectrometry data
# can guide the choice of residue pairs and distance bounds. We translate that information
# into pairwise residue contacts to bias BioEmu2's complex predictions towards a specific interface.
# Here, we use three contacts from the design pose and compare predictions with and without them.
#
# > **This demonstration uses query-only multiple sequence alignments (MSAs) to keep runtime short.**
# > Each chain has only its query sequence, with no homologues. For real applications,
# > use BioEmu2's default MSA workflow.
#
# Install the existing `viz` extra using the [installation guide](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart),
# then run the cells or
# `python examples/sample_protein_protein_constraints.py`.
# Run notebook cells from the repository root or `examples`.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mdtraj
import numpy as np

from bioemu2 import MSA, Contact, EDMSampler, EmuSession, System, load_model, save_frames
from bioemu2.data_interface import ContactToken

# %% [markdown]
# ## Set up the target, binder, and interface contacts
#
# The supplied `assets/1ten_protein_constraint.pdb` is a **design reference, not an experimental
# binder complex**. Chain A is the target and chain B is the binder. We read their sequences
# to define the system; reference coordinates are not supplied to the model.
#
# The three fixed contacts connect residues whose heavy atoms lie within 5 Å in the design.
# Their distance bounds are the corresponding Cα distances plus 1 Å, rounded up to 0.5 Å.
# When using other evidence, choose bounds appropriate to that evidence and the Cα representation.
# Contacts bias predictions; they do not enforce hard limits.

# %%
here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if (here / "examples").is_dir():
    here = here / "examples"
outputs = here / "binder_constraint" / "outputs" / "1TEN"
reference = mdtraj.load(str(here / "assets" / "1ten_protein_constraint.pdb"))
# Extract sequences only; keep the coordinates for comparison after prediction.
system = System.from_description(proteins=reference.topology.to_fasta())
# Each tuple is (1-based target A position, 1-based binder B position, maximum Cα–Cα distance in Å).
contact_specs = [(26, 7, 10.5), (22, 14, 10.5), (8, 18, 14.5)]
contacts = [
    Contact(
        token1=ContactToken(asym_id="A", res_id=target),
        token2=ContactToken(asym_id="B", res_id=binder), max_distance=distance,
    )
    for target, binder, distance in contact_specs
]
conditions = {"default": "Unconstrained", "three_contacts": "Constrained"}

# %% [markdown]
# ## Predict with and without the contacts
#
# Generate five structures per condition in structure-prediction mode. Each condition gets
# its own `EmuSession` so its embeddings include the chosen contacts. Keep the seed,
# batch size, and denoising settings the same.
#
# `load_model()` downloads the default checkpoint when needed. Saved predictions are reused;
# choose a fresh output directory after changing contacts or sampling settings.

# %%
def predict():
    # One query per chain skips homologous MSA search for this demonstration.
    msa = MSA({
        entity.asym_ids[0]: f">query_{entity.asym_ids[0]}\n{entity.sequence}\n"
        for entity in system.entities
    })
    predictions = {}
    model = None
    for condition in conditions:
        directory = outputs / condition
        if directory.exists():
            samples = mdtraj.load_xtc(str(directory / "samples.xtc"), top=str(directory / "topology.cif"))
        else:
            if model is None:
                model = load_model(device="cuda", precision="bf16-mixed")
            condition_contacts = contacts if condition == "three_contacts" else []
            session = EmuSession(system, model=model, mode="structure", msa=msa, contacts=condition_contacts)
            samples = session.sample(
                num_samples=5, batch_size=1, seed=42, sampler=EDMSampler(num_steps=200, step_scale=1.5),
            )
            save_frames(samples, directory)
            del session
        predictions[condition] = samples
    return predictions


predictions = predict()

# %% [markdown]
# ## Compare predictions with the design
#
# Fit each whole complex to the reference using Cα atoms and plot the five predictions per condition.
# Atoms match by chain and sequence order, not PDB residue numbers. Lower RMSD means closer
# agreement with this design; it does not establish experimental binding or design quality.

# %%
def ca_trace(trajectory: mdtraj.Trajectory) -> mdtraj.Trajectory:
    return trajectory.atom_slice(trajectory.topology.select("protein and name CA"))


def complex_rmsd(samples: mdtraj.Trajectory, reference: mdtraj.Trajectory) -> np.ndarray:
    # MDTraj reports distances in nanometres; display RMSD in angstroms.
    return 10 * mdtraj.rmsd(ca_trace(samples), ca_trace(reference))


fig, ax = plt.subplots(figsize=(4.5, 3), constrained_layout=True)
for column, condition in enumerate(conditions):
    rmsds = complex_rmsd(predictions[condition], reference)
    ax.scatter(column + np.linspace(-0.08, 0.08, len(rmsds)), rmsds, s=35)
ax.set(title="1TEN", ylabel="Complex Cα RMSD (Å)", xticks=[0, 1], xticklabels=list(conditions.values()))
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(outputs / "rmsd_comparison.png", dpi=160)
plt.show()

# %% [markdown]
# ## Inspect the first prediction from each condition
#
# For each condition, align the first prediction to the design reference using all complex
# Cα atoms. The legend identifies the reference and predicted chains.
# This static 3D plot uses Matplotlib, with no additional viewer dependencies.

# %%
fig = plt.figure(figsize=(9, 4), constrained_layout=True)
reference_ca = ca_trace(reference)
for column, (condition, label) in enumerate(conditions.items()):
    prediction = ca_trace(predictions[condition][:1]).superpose(reference_ca)
    ax = fig.add_subplot(1, 2, column + 1, projection="3d")
    for trace, colors, source in (
        (reference_ca, ("0.65", "black"), "Reference"),
        (prediction, ("tab:blue", "tab:orange"), "Predicted"),
    ):
        for chain, color, role in zip(trace.topology.chains, colors, ("target", "binder"), strict=True):
            xyz = trace.xyz[0, [atom.index for atom in chain.atoms]] * 10
            ax.plot(*xyz.T, color=color, label=f"{source} {role}")
    ax.set_title(label)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, frameon=False)
fig.savefig(outputs / "aligned_comparison.png", dpi=160)
plt.show()

# %% [markdown]
# ## Summary
#
# Protein-protein contacts let us use interface information from epitopes, structural templates,
# or cross-linking measurements to bias complex predictions towards a specific binding interface.
# This example compares predictions with and without three contacts for a 1TEN binder generated
# in a short ColabDesign run, using Cα RMSD and 3D overlays to assess agreement with the design.
