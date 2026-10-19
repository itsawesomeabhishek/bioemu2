import mdtraj
import numpy as np
import pytest

from bioemu2.visualization import view_structure, view_structure_grid, view_trajectory


def test_view_structure_embeds_selected_frame() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(
        xyz=np.array([[[0.0, 0.0, 0.0]], [[0.1, 0.2, 0.3]]]),
        topology=topology,
    )

    html = view_structure(trajectory, frame=1)._make_html()

    assert "3Dmol.createViewer" in html
    assert "_atom_site.Cartn_x" in html
    assert html.count("ATOM") == 1


def test_view_structure_rejects_unknown_frame() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(xyz=np.zeros((1, 1, 3)), topology=topology)

    with pytest.raises(IndexError, match="outside a trajectory"):
        view_structure(trajectory, frame=1)


def test_view_trajectory_embeds_bounded_aligned_frames() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(
        xyz=np.array([[[offset, 0.0, 0.0]] for offset in range(4)]),
        topology=topology,
    )

    html = view_trajectory(trajectory, max_frames=3)._make_html()

    assert html.count("MODEL") == 3
    assert "animate" in html


def test_view_structure_grid_embeds_aligned_frames() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("N", mdtraj.element.nitrogen, residue)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    topology.add_atom("C", mdtraj.element.carbon, residue)
    base_coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ]
    )
    trajectory = mdtraj.Trajectory(
        xyz=np.array([base_coordinates + np.array([offset, 0.0, 0.0]) for offset in range(8)]),
        topology=topology,
    )

    html = view_structure_grid(trajectory, rows=2, columns=4)._make_html()

    assert "createViewerGrid" in html
    assert html.count("ATOM") == 24
    assert "70.000" not in html
    assert '"hetflag": false' in html
    assert html.count("addSurface") == 8
    assert '"color": "#0078d4"' in html
    assert '"opacity": 0.5' in html


def test_view_structure_grid_requires_one_frame_per_cell() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("GLY", chain)
    topology.add_atom("CA", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(xyz=np.zeros((2, 1, 3)), topology=topology)

    with pytest.raises(ValueError, match="2x4 grid requires 8 frames"):
        view_structure_grid(trajectory, rows=2, columns=4)
