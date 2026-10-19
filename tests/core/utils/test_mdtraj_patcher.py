import mdtraj
import numpy as np
import pytest


def test_patching_works():
    assert hasattr(mdtraj, "IS_PATCHED")
    assert mdtraj.IS_PATCHED


@pytest.mark.parametrize(
    ("resname", "rescode"),
    [("ASP", "D"), ("ASH", "D"), ("CYX", "C"), ("HIS", "H"), ("HIP", "H"), ("HIE", "H")],
)
def test_protonated_residue_to_fasta(resname, rescode):
    top = mdtraj.Topology()
    top.add_residue(resname, top.add_chain())

    fasta = top.to_fasta()
    assert len(fasta) == 1
    assert fasta[0] == rescode


@pytest.fixture
def traj_and_contacts_with_missing_ca(n_residues: int = 4) -> tuple[mdtraj.Trajectory, np.ndarray]:
    """Trajectory where one residue lacks a CA atom (e.g., a ACE cap at the N-terminus) — triggers
    the buggy code path.
    """
    top = mdtraj.Topology()
    chain = top.add_chain()
    if n_residues < 4:
        raise ValueError("Need at least 4 residues to test missing CA case")
    r = top.add_residue("ACE", chain)
    top.add_atom("CH3", mdtraj.element.carbon, r)
    top.add_atom("C", mdtraj.element.carbon, r)
    top.add_atom("O", mdtraj.element.oxygen, r)
    for i in range(n_residues - 1):
        r = top.add_residue("ALA", chain)
        top.add_atom("CA", mdtraj.element.carbon, r)
        top.add_atom("C", mdtraj.element.carbon, r)
        top.add_atom("N", mdtraj.element.nitrogen, r)
        top.add_atom("O", mdtraj.element.oxygen, r)
    coords = np.zeros((1, top.n_atoms, 3), dtype=np.float32)
    coords[0, :, 0] = np.arange(top.n_atoms) * 0.38
    # Request contacts that include residue 0 (no CA)
    contacts = np.array([[0, 2], [1, 3]])
    return mdtraj.Trajectory(coords, top), contacts


def test_compute_contacts_with_missing_ca(traj_and_contacts_with_missing_ca):
    traj, contacts = traj_and_contacts_with_missing_ca
    dists, contact_pairs = mdtraj.compute_contacts(traj, scheme="ca", contacts=contacts)
    assert np.allclose(dists, np.array([[3.04]], dtype=np.float32))
    assert isinstance(contact_pairs, np.ndarray)
    assert np.array_equal(contact_pairs, np.array([[1, 3]]))
