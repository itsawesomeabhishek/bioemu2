# %% [markdown]
# # Predict and sample protein structures
#
# BioEmu2 can predict a protein structure and generate a varied set of structures
# that the protein may adopt. This notebook demonstrates both capabilities with
# adenylate kinase (PDB 1AKE), a small protein available from the RCSB PDB.
#
# Related guides: [systems](https://github.com/microsoft/bioemu2/blob/main/docs/system.md),
# [MSAs](https://github.com/microsoft/bioemu2/blob/main/docs/msa.md), and
# [sampling](https://github.com/microsoft/bioemu2/blob/main/docs/sampling.md).

# %% [markdown]
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %%
from pathlib import Path

import mdtraj

from bioemu2 import (
    EmuSession,
    System,
    load_model,
    save_frames,
)
from bioemu2.visualization import view_trajectory

# %% [markdown]
# ## Load the molecular system
#
# The molecular composition is defined by a `System` instance. This notebook
# uses adenylate kinase from PDB entry 1AKE. See the
# [system guide](https://github.com/microsoft/bioemu2/blob/main/docs/system.md) to learn how to define other proteins and
# molecular systems.

# %%
system = System.from_pdb_id("1AKE")

print(system)

# %% [markdown]
# ## Predict or sample structures
#
# BioEmu2 was trained for two distinct tasks:
# structure prediction to generate accurate individual structures, and
# equilibrium sampling to represent a diverse conformational ensemble. A
# session fixes the task mode and computes the MSA on first use.
#
# Both modes use the same denoising algorithm, but structure prediction applies
# the heuristic sharpening factor `step_scale=1.5`. Equilibrium sampling uses
# `step_scale=1.0` so that it does not apply this extra sharpening.

# %%
model = load_model()
prediction = EmuSession(system, model=model, mode="structure")
prediction_samples: mdtraj.Trajectory = prediction.sample(num_samples=8, seed=0)
prediction_samples

# %% [markdown]
# ### Visualize the predicted structures
#
# The aligned viewer cycles through the generated structures and remains interactive.

# %%
prediction_view = view_trajectory(prediction_samples, max_frames=8)
prediction_view.show()

# %% [markdown]
# ## Sample equilibrium conformations
#
# Use a separate session to select equilibrium conditioning and its recommended
# denoising step scale.

# %%
equilibrium = EmuSession(system, model=model, mode="equilibrium", msa=prediction.msa)
equilibrium_samples: mdtraj.Trajectory = equilibrium.sample(num_samples=8, seed=0)
equilibrium_samples

# %% [markdown]
# ### Visualize the equilibrium conformations

# %%
equilibrium_view = view_trajectory(equilibrium_samples, max_frames=8)
equilibrium_view.show()

# %% [markdown]
# ## Save the generated structures
#
# `save_frames` writes the generated coordinates to `samples.xtc` and the
# matching molecular topology to `topology.cif`. You can read these files later
# with `mdtraj.load_xtc(...)`:
#
# ```python
# samples = mdtraj.load_xtc(
#     "outputs/sample_structures/prediction/samples.xtc",
#     top="outputs/sample_structures/prediction/topology.cif",
# )
# ```

# %%
output_dir = Path("outputs/sample_structures")
prediction_path = save_frames(prediction_samples, output_dir / "prediction")
equilibrium_path = save_frames(equilibrium_samples, output_dir / "equilibrium")

print(prediction_path)
print(equilibrium_path)
