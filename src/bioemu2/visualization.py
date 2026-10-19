"""Interactive py3Dmol viewers for structures and trajectories."""

from pathlib import Path
from tempfile import TemporaryDirectory

import mdtraj
import numpy as np
import py3Dmol

from bioemu2.trajectory_io import save_mdtraj_cif_frame

_CHAIN_COLORS = ("#0078d4", "#49c5b1", "#8661c5", "#ffa38b", "#ffb900")


def view_structure(
    trajectory: mdtraj.Trajectory,
    *,
    frame: int = 0,
    width: int = 700,
    height: int = 500,
) -> py3Dmol.view:
    """Create a spinnable 3D viewer with one structure embedded in its HTML."""
    normalized_frame = frame if frame >= 0 else trajectory.n_frames + frame
    if not 0 <= normalized_frame < trajectory.n_frames:
        raise IndexError(
            f"Frame {frame} is outside a trajectory with {trajectory.n_frames} frames."
        )

    aligned = trajectory[normalized_frame]
    aligned.superpose(trajectory, frame=0)
    with TemporaryDirectory() as directory:
        cif_path = Path(directory) / "structure.cif"
        save_mdtraj_cif_frame(aligned, cif_path)
        cif_text = cif_path.read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(cif_text, "cif")
    view.setStyle({}, {"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": True}, {"stick": {}})
    view.zoomTo()
    return view


def view_trajectory(
    trajectory: mdtraj.Trajectory,
    *,
    max_frames: int = 20,
    interval: int = 800,
    width: int = 700,
    height: int = 500,
) -> py3Dmol.view:
    """Create an aligned, auto-playing 3D trajectory with a bounded number of frames."""
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")
    if interval <= 0:
        raise ValueError("interval must be positive.")

    frame_indices = np.linspace(
        0,
        trajectory.n_frames - 1,
        min(trajectory.n_frames, max_frames),
        dtype=int,
    )
    rendered = trajectory[frame_indices]
    rendered.superpose(trajectory, frame=0)
    non_protein_residues = sorted(
        {residue.name for residue in rendered.topology.residues if not residue.is_protein}
    )
    atoms = list(rendered.topology.atoms)
    serials = [atom.serial for atom in atoms]
    with TemporaryDirectory() as directory:
        pdb_path = Path(directory) / "trajectory.pdb"
        try:
            for serial, atom in enumerate(atoms, start=1):
                atom.serial = serial
            rendered.save_pdb(str(pdb_path))
        finally:
            for serial, atom in zip(serials, atoms, strict=True):
                atom.serial = serial
        pdb_text = pdb_path.read_text()

    view = py3Dmol.view(width=width, height=height)
    view.addModelsAsFrames(pdb_text, "pdb")
    view.setStyle({}, {"cartoon": {"color": "spectrum"}})
    if non_protein_residues:
        view.addStyle({"resn": non_protein_residues}, {"stick": {"radius": 0.18}})
    view.zoomTo()
    view.animate({"loop": "forward", "reps": 0, "interval": interval})
    return view


def view_structure_grid(
    trajectory: mdtraj.Trajectory,
    *,
    rows: int,
    columns: int,
    width: int = 1400,
    height: int = 700,
) -> py3Dmol.view:
    """Create a grid of structures aligned to the first frame."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive.")
    if trajectory.n_frames != rows * columns:
        raise ValueError(
            f"The {rows}x{columns} grid requires {rows * columns} frames, "
            f"but the trajectory has {trajectory.n_frames}."
        )

    aligned = trajectory[:]
    aligned.superpose(trajectory, frame=0)
    protein_chain_ids = [
        chain.chain_id or chr(ord("A") + chain.index)
        for chain in aligned.topology.chains
        if any(residue.is_protein for residue in chain.residues)
    ]
    non_protein_residues = sorted(
        {residue.name for residue in aligned.topology.residues if not residue.is_protein}
    )

    view = py3Dmol.view(
        width=width,
        height=height,
        viewergrid=(rows, columns),
        linked=True,
    )
    with TemporaryDirectory() as directory:
        cif_path = Path(directory) / "structure.cif"
        for frame_index in range(aligned.n_frames):
            save_mdtraj_cif_frame(aligned[frame_index], cif_path)
            viewer = divmod(frame_index, columns)
            view.addModel(cif_path.read_text(), "cif", viewer=viewer)
            for chain_index, chain_id in enumerate(protein_chain_ids):
                color = _CHAIN_COLORS[chain_index % len(_CHAIN_COLORS)]
                selection = {"chain": chain_id}
                view.setStyle(
                    selection,
                    {"cartoon": {"color": color}},
                    viewer=viewer,
                )
                view.addSurface(
                    py3Dmol.VDW,
                    {"color": color, "opacity": 0.5},
                    selection,
                    viewer=viewer,
                )
            if non_protein_residues:
                view.addStyle(
                    {"resn": non_protein_residues},
                    {"stick": {"radius": 0.18}},
                    viewer=viewer,
                )
            view.zoomTo({"hetflag": False}, viewer=viewer)
    return view
