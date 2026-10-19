# %% [markdown]
# # Simulate parallel trajectories for adenylate kinase
#
# This notebook loads adenylate kinase from the PDB, samples a starting structure,
# then propagates several independent trajectories from that structure.
#
# > [!NOTE]
# > Here each `mdtraj.Trajectory` is a time-ordered rollout, unlike the
# > unordered bag of samples stored in `mdtraj.Trajectory` by other BioEmu2
# > sampling APIs.
#
# For other propagation workflows, see the
# [propagation guide](https://github.com/microsoft/bioemu2/blob/main/docs/propagation.md).
#
# For installation and hardware requirements, see the
# [notebook quickstart](https://github.com/microsoft/bioemu2/blob/main/README.md#notebook-quickstart).

# %%
from pathlib import Path

import mdtraj
import matplotlib.pyplot as plt

from bioemu2 import (
    System,
    load_model,
    sample_equilibrium,
    save_frames,
    simulate_trajectory,
)
from bioemu2.visualization import view_trajectory

# %% [markdown]
# ## Prepare the system and model

# %%
system = System.from_pdb_id("1AKE")
model = load_model()

# %% [markdown]
# ## Sample a starting conformation

# %%
x0: mdtraj.Trajectory = sample_equilibrium(
    system,
    model=model,
    num_samples=1,
    batch_size=1,
    seed=0,
)

# %% [markdown]
# ## Propagate
#
# Repeat the initial conformation once for each independent trajectory.
# `simulate_trajectory()` computes one embedding, reuses it across every step,
# and returns one time-ordered trajectory per starting frame. Each trajectory
# includes the starting structure at time zero and one frame per propagation step.

# %%
lag_time_ns = 50
num_trajectory_steps = 20
num_trajectories = 4

trajectories = simulate_trajectory(
    system,
    model=model,
    x0=x0[[0] * num_trajectories],
    lag_time_ns=lag_time_ns,
    num_trajectory_steps=num_trajectory_steps,
    seed=1,
)
print(
    f"Generated {len(trajectories)} trajectories with "
    f"{trajectories[0].n_frames} frames each."
)

# %% [markdown]
# ## Visualize the propagated trajectory
#
# Each endpoint is aligned to its own previous frame. Animate the first trajectory;
# the distance plot below shows all trajectories.

# %%
view = view_trajectory(trajectories[0])
view.show()

# %% [markdown]
# ## Save the propagated trajectory

# %%
output_dir = Path("outputs/simulate_a_trajectory")
for i, trajectory in enumerate(trajectories, start=1):
    output_path = save_frames(trajectory, output_dir / f"trajectory_{i:02d}")
    print(f"saved trajectory {i} to {output_path}")

# %% [markdown]
# ## Compare the trajectories
#
# Plot the distance between the alpha carbons of residues 35 and 125 (1-based
# topology order). These residues lie in the NMP and LID domains, which move
# relative to each other as adenylate kinase opens and closes. Each line follows
# one trajectory across all propagation steps. MDTraj stores frame times in
# picoseconds; divide by 1000 to plot elapsed time in nanoseconds.

# %%
ca35 = system.topology.residue(34).atom("CA")
ca125 = system.topology.residue(124).atom("CA")
fig, ax = plt.subplots(figsize=(8, 4))
for i, trajectory in enumerate(trajectories, start=1):
    times_ns = trajectory.time / 1000
    distance_nm = mdtraj.compute_distances(trajectory, [[ca35.index, ca125.index]])[:, 0]
    ax.plot(times_ns, distance_nm, marker="o", markersize=3, label=f"Trajectory {i}")
ax.set(xlabel="Elapsed time (ns)", ylabel="CA 35-CA 125 distance (nm)")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "ca_distance.png", dpi=160)
plt.show()
