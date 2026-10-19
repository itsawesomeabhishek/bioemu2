import json
from pathlib import Path

import mdtraj
import numpy as np
import pytest

from bioemu2 import Atom, Bond, Ligand, Protein, System, physicality
from bioemu2.physicality import (
    PHYSICALITY_THRESHOLDS,
    PhysicalityCutoffs,
    PhysicalityFilterMode,
    _check_topology,
    _compute_geometry_stats,
    _select_expected_extra_bonds,
    compute_physicality_stats,
    get_protein_atom_indices,
    normalize_physicality_mode,
    remove_unphysical_frames,
)
from bioemu2.sampling import (
    apply_physicality_filter,
    check_physicality,
    expected_extra_bonds_from_system,
    save_topology_cif,
)
from bioemu2.structure import AllAtomStructure


@pytest.mark.parametrize("residue_name", ["LIG", "UNK", "MSE"])
def test_unsupported_residue_only_chain_is_not_checked(residue_name: str, tmp_path: Path) -> None:
    topology = mdtraj.Topology()
    residue = topology.add_residue(residue_name, topology.add_chain("X"))
    topology.add_atom("C1", mdtraj.element.carbon, residue)
    trajectory = mdtraj.Trajectory(np.zeros((2, 1, 3)), topology)
    assert get_protein_atom_indices(topology).size == 0
    with pytest.raises(ValueError, match="requires at least one protein atom"):
        remove_unphysical_frames(trajectory, PhysicalityFilterMode.HIGH_QUALITY)

    report = check_physicality(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    assert report.mode is PhysicalityFilterMode.HIGH_QUALITY
    assert not report.checked
    assert report.passed.tolist() == [True, True]
    assert report.reasons == [None, None]
    report.save(tmp_path / "physicality.json")
    assert json.loads((tmp_path / "physicality.json").read_text()) == [
        {"frame": index, "status": "not_checked", "reason": None} for index in range(2)
    ]

    kept, rejected = apply_physicality_filter(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    assert len(kept) == len(trajectory)
    assert len(rejected) == 0


@pytest.mark.parametrize("mode", [PhysicalityFilterMode.NO_FILTER, "no_filter"])
def test_no_physicality_filter(mode: PhysicalityFilterMode | str) -> None:
    trajectory = mdtraj.Trajectory(np.zeros((1, 0, 3)), mdtraj.Topology())

    assert remove_unphysical_frames(trajectory, mode) is trajectory


def test_none_is_not_a_physicality_mode() -> None:
    trajectory = mdtraj.Trajectory(np.zeros((1, 0, 3)), mdtraj.Topology())
    with pytest.raises(ValueError, match="not a valid PhysicalityFilterMode"):
        remove_unphysical_frames(
            trajectory,
            None,  # type: ignore[arg-type]
        )


def test_protein_atom_indices_ignore_unknown_ligand() -> None:
    system = System(
        entities=[Protein(sequence="AA", asym_ids=["A"]), Ligand(smiles="O", asym_ids=["B"])]
    )
    topology = system.topology
    assert topology.chain(1).residue(0).name == "UNK"
    assert topology.chain(1).residue(0).is_protein

    selected = get_protein_atom_indices(topology)
    assert selected.tolist() == [atom.index for atom in topology.chain(0).atoms]


def test_protein_atom_indices_keep_modified_residue_and_peptide_bonds() -> None:
    system = System(entities=[Protein(sequence="AAA", asym_ids=["A"])])
    topology = system.topology
    topology.residue(1).name = "UNK"
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    selected = get_protein_atom_indices(topology)
    protein_traj = trajectory.atom_slice(selected)
    assert selected.tolist() == list(range(topology.n_atoms))
    assert len(list(protein_traj.topology.bonds)) == len(list(topology.bonds))


def test_selector_retains_modified_protein_chain_without_system() -> None:
    system = System(
        entities=[
            Protein(sequence="AXA", asym_ids=["A"], mods=[{"res_id": 2, "ccd_code": "NLE"}]),
            Ligand(smiles="O", asym_ids=["B"]),
        ]
    )
    topology = system.topology
    protein_atoms = [atom.index for atom in topology.chain(0).atoms]

    assert get_protein_atom_indices(topology).tolist() == protein_atoms
    assert (
        _check_topology(
            mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology).atom_slice(
                protein_atoms
            )
        )["bonds_mismatch"]
        == 0
    )


def test_single_residue_amino_acid_named_ligand_is_not_selected() -> None:
    system = System(
        entities=[Protein(sequence="AA", asym_ids=["A"]), Ligand(smiles="O", asym_ids=["B"])]
    )
    topology = system.topology
    topology.chain(1).residue(0).name = "ALA"
    protein_atoms = [atom.index for atom in topology.chain(0).atoms]

    assert get_protein_atom_indices(topology).tolist() == protein_atoms


def test_single_residue_protein_is_not_selected() -> None:
    topology = System(entities=[Protein(sequence="A", asym_ids=["A"])]).topology
    assert get_protein_atom_indices(topology).size == 0
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
    with pytest.raises(ValueError, match="requires at least one protein atom"):
        remove_unphysical_frames(trajectory, PhysicalityFilterMode.HIGH_QUALITY)


@pytest.mark.parametrize("modified_name", ["UNK", "NLE"])
def test_topology_check_skips_modified_sidechain_but_checks_backbone(modified_name: str) -> None:
    topology = System(entities=[Protein(sequence="GGA", asym_ids=["A"])]).topology
    modified = topology.residue(1)
    modified.name = modified_name
    extra_atom = topology.add_atom("CX", mdtraj.element.carbon, modified)
    backbone_atom = next(atom for atom in modified.atoms if atom.name == "CA")
    topology.add_bond(backbone_atom, extra_atom)
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    assert _check_topology(trajectory)["bonds_mismatch"] == 0
    assert compute_physicality_stats(trajectory)["bad_bonds_backbone"].item() > 0


@pytest.mark.parametrize("modified_name", ["UNK", "NLE"])
def test_nonstandard_backbone_connections_are_checked_without_system(
    modified_name: str,
) -> None:
    system = System(
        entities=[
            Protein(
                sequence="AXA",
                asym_ids=["A"],
                mods=[{"res_id": 2, "ccd_code": "NLE"}],
            ),
            Ligand(smiles="O", asym_ids=["B"]),
        ]
    )
    topology = system.topology
    modified = topology.chain(0).residue(1)
    modified.name = modified_name
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
    selected = get_protein_atom_indices(topology)
    protein_traj = trajectory.atom_slice(selected)

    assert [res.name for res in protein_traj.topology.residues] == ["ALA", modified_name, "ALA"]
    assert _check_topology(protein_traj)["bonds_mismatch"] == 0

    first_peptide_bond = next(
        i
        for i, (a, b) in enumerate(protein_traj.topology.bonds)
        if a.residue.index == 0 and b.residue.index == 1
    )
    protein_traj.topology._bonds.pop(first_peptide_bond)
    assert _check_topology(protein_traj)["bonds_mismatch"] == 1


def test_nonbackbone_residue_does_not_link_its_neighbors() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain("A")
    for name, atom_names in (
        ("ALA", ("N", "CA", "C", "O", "CB")),
        ("CRO", ("C1", "O1")),
        ("ALA", ("N", "CA", "C", "O", "CB")),
    ):
        residue = topology.add_residue(name, chain)
        for atom_name in atom_names:
            element = (
                mdtraj.element.oxygen
                if atom_name.startswith("O")
                else (
                    mdtraj.element.nitrogen if atom_name.startswith("N") else mdtraj.element.carbon
                )
            )
            topology.add_atom(atom_name, element, residue)
    topology.create_standard_bonds()
    cro = topology.residue(1)
    topology.add_bond(*list(cro.atoms))
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
    selected = get_protein_atom_indices(topology)
    protein_traj = trajectory.atom_slice(selected)

    assert protein_traj.n_atoms == topology.n_atoms
    assert _check_topology(protein_traj)["bonds_mismatch"] == 0
    assert all(
        first.residue.index == second.residue.index for first, second in protein_traj.topology.bonds
    )
    stats = compute_physicality_stats(protein_traj)
    assert stats["bad_bonds_sidechain"].item() > 0
    assert stats["bonds_mismatch"].tolist() == [0]
    assert stats["same_chain_residue_clashes"].item() > 0


def test_modified_bond_without_standard_backbone_pair_uses_generic_distance() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain("A")
    topology.add_atom("CB", mdtraj.element.carbon, topology.add_residue("ALA", chain))
    residue = topology.add_residue("UNK", chain)
    nitrogen = topology.add_atom("N", mdtraj.element.nitrogen, residue)
    oxygen = topology.add_atom("O", mdtraj.element.oxygen, residue)
    topology.add_bond(nitrogen, oxygen)
    trajectory = mdtraj.Trajectory(np.array([[[1.0, 0, 0], [0, 0, 0], [0.3, 0, 0]]]), topology)

    stats = compute_physicality_stats(trajectory)
    assert stats["bad_bonds_backbone"].tolist() == [0]
    assert stats["bad_bonds_sidechain"].tolist() == [1]


@pytest.mark.parametrize("mode", list(PhysicalityFilterMode))
def test_physicality_preset_strings_resolve_to_modes(mode: PhysicalityFilterMode) -> None:
    assert normalize_physicality_mode(mode.value) is mode
    assert normalize_physicality_mode(mode) is mode


def test_topology_check_detects_duplicate_atoms_in_modified_residue() -> None:
    topology = System(entities=[Protein(sequence="AA", asym_ids=["A"])]).topology
    modified = topology.residue(1)
    modified.name = "NLE"
    topology.add_atom("CA", mdtraj.element.carbon, modified)
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    assert _check_topology(trajectory)["duplicate_atoms"] == 1


def test_declared_protein_crosslink_is_not_an_unexpected_bond() -> None:
    system = System(
        entities=[Protein(sequence="CC", asym_ids=["A"])],
        bonds=[
            Bond(
                atom1=Atom(name="SG", res_name="CYS", res_id=1, asym_id="A"),
                atom2=Atom(name="SG", res_name="CYS", res_id=2, asym_id="A"),
            )
        ],
    )
    topology = system.topology
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    extra_bonds = expected_extra_bonds_from_system(system, topology)
    assert compute_physicality_stats(trajectory, expected_extra_bonds=extra_bonds)[
        "bonds_mismatch"
    ].tolist() == [0]
    extra_bond_index = next(
        i for i, (a, b) in enumerate(topology.bonds) if a.name == b.name == "SG"
    )
    topology._bonds.pop(extra_bond_index)
    assert _check_topology(trajectory, expected_extra_bonds=extra_bonds)["bonds_mismatch"] == 1


def test_external_mdtraj_topology_accepts_expected_extra_bond_indices() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain("A")
    for _ in range(2):
        residue = topology.add_residue("CYS", chain)
        for name, element in (
            ("N", mdtraj.element.nitrogen),
            ("CA", mdtraj.element.carbon),
            ("C", mdtraj.element.carbon),
            ("O", mdtraj.element.oxygen),
            ("CB", mdtraj.element.carbon),
            ("SG", mdtraj.element.sulfur),
        ):
            topology.add_atom(name, element, residue)
    topology.create_standard_bonds()
    first = next(atom for atom in topology.residue(0).atoms if atom.name == "SG")
    second = next(atom for atom in topology.residue(1).atoms if atom.name == "SG")
    topology.add_bond(first, second)
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    assert _check_topology(trajectory)["bonds_mismatch"] == 1
    expected = {(first.index, second.index)}
    assert compute_physicality_stats(trajectory, expected_extra_bonds=expected)[
        "bonds_mismatch"
    ].tolist() == [0]


def test_disulfide_bond_length_uses_sulfur_limit() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain("A")
    sulfurs = []
    for _ in range(2):
        residue = topology.add_residue("CYS", chain)
        topology.add_atom("CB", mdtraj.element.carbon, residue)
        sulfurs.append(topology.add_atom("SG", mdtraj.element.sulfur, residue))
    topology.add_bond(*sulfurs)
    xyz = np.zeros((3, 4, 3))
    for frame, distance in enumerate((0.205, 0.27, 0.17)):
        xyz[frame, :, 0] = [-0.18, 0.0, distance + 0.18, distance]
    samples = mdtraj.Trajectory(xyz, topology)

    # A stretched or compressed disulfide fails both cysteines, and no mode tolerates it.
    stats = compute_physicality_stats(samples)
    assert stats["bad_bonds_sidechain"].tolist() == [0, 2, 2]
    assert stats["bad_bonds_sulfur"].tolist() == [0, 1, 1]
    assert all(t.bad_bonds_sulfur == 0 for t in PHYSICALITY_THRESHOLDS.values())


def test_extra_bond_leaving_checked_chains_warns(caplog: pytest.LogCaptureFixture) -> None:
    sulfurs = [Atom(name="SG", res_name="CYS", res_id=1, asym_id=chain) for chain in "AB"]
    system = System(
        entities=[Protein(sequence="CC", asym_ids=["A"]), Protein(sequence="C", asym_ids=["B"])],
        bonds=[Bond(atom1=sulfurs[0], atom2=sulfurs[1])],
    )
    topology = system.topology
    declared = expected_extra_bonds_from_system(system, topology)
    # Single-residue chain B is not checked, so its disulfide is dropped with a warning.
    assert not _select_expected_extra_bonds(
        declared, get_protein_atom_indices(topology), topology.n_atoms
    )
    assert "is not checked" in caplog.text


def test_extra_bond_indices_remap_after_ignoring_ligand_chain() -> None:
    system = System(
        entities=[
            Ligand(smiles="O", asym_ids=["L"]),
            Protein(sequence="CC", asym_ids=["A"]),
        ],
        bonds=[
            Bond(
                atom1=Atom(name="SG", res_name="CYS", res_id=1, asym_id="A"),
                atom2=Atom(name="SG", res_name="CYS", res_id=2, asym_id="A"),
            )
        ],
    )
    topology = system.topology
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)
    selected = get_protein_atom_indices(topology)
    declared = expected_extra_bonds_from_system(system, topology)
    protein_traj = trajectory.atom_slice(selected)
    mapped = _select_expected_extra_bonds(declared, selected, topology.n_atoms)

    assert selected[0] > 0
    assert _check_topology(protein_traj)["bonds_mismatch"] == 1
    assert _check_topology(protein_traj, expected_extra_bonds=mapped)["bonds_mismatch"] == 0
    with pytest.raises(ValueError, match="Invalid extra bond atom indices"):
        remove_unphysical_frames(
            trajectory, PhysicalityFilterMode.HIGH_QUALITY, expected_extra_bonds={(0, 99)}
        )


def _element(atom_name: str) -> mdtraj.element.Element:
    return mdtraj.element.Element.getBySymbol(atom_name[0])


@pytest.mark.parametrize(
    "atom_name_i,residue_j,atom_name_j,distance,n_bad",
    [
        pytest.param("C", 1, "N", 1.60, 0, id="peptide-accepted"),
        pytest.param("C", 1, "N", 1.66, 1, id="peptide-too-long"),
        pytest.param("C", 1, "N", 1.04, 1, id="peptide-too-short"),
        pytest.param("C", 0, "O", 1.49, 0, id="carbonyl-accepted"),
        pytest.param("C", 0, "O", 1.51, 1, id="carbonyl-too-long"),
        pytest.param("N", 0, "CA", 1.24, 1, id="n-ca-too-short"),
    ],
)
def test_backbone_bond_windows(
    atom_name_i: str, residue_j: int, atom_name_j: str, distance: float, n_bad: int
) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    first = topology.add_residue("ALA", chain)
    second = first if residue_j == 0 else topology.add_residue("ALA", chain)
    topology.add_atom(atom_name_i, _element(atom_name_i), first)
    topology.add_atom(atom_name_j, _element(atom_name_j), second)
    xyz = np.array([[[0.0, 0.0, 0.0], [distance / 10.0, 0.0, 0.0]]])

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    assert stats["bad_bonds_backbone"].tolist() == [n_bad]


@pytest.mark.parametrize("alpha_backbone", [True, False], ids=["alpha", "beta"])
def test_backbone_windows_apply_only_to_alpha_amino_acids(alpha_backbone: bool) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    topology.add_atom("CA", mdtraj.element.carbon, topology.add_residue("GLY", chain))
    # Beta-alanine (BAL) places CB between N and CA, so N and CA are 2.5 angstrom apart.
    residue = topology.add_residue("BAL", chain)
    atoms = {
        name: topology.add_atom(name, _element(name), residue)
        for name in ("N", "CB", "CA", "C", "O")
    }
    bonds = [("N", "CA")] if alpha_backbone else [("N", "CB"), ("CB", "CA")]
    for first, second in [*bonds, ("CA", "C"), ("C", "O")]:
        topology.add_bond(atoms[first], atoms[second])
    xyz = np.array([[[0, 1, 0], [0, 0, 0], [1.25, 0, 0], [2.5, 0, 0], [4.0, 0, 0], [4.0, 1.2, 0]]])

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz / 10.0, topology))

    assert stats["bad_bonds_backbone"].tolist() == [int(alpha_backbone)]
    assert stats["bad_bonds_sidechain"].tolist() == [0]


@pytest.mark.parametrize(
    "sulfur_distance,n_bad",
    [
        pytest.param(2.05, 0, id="sulfur-bond-accepted"),
        pytest.param(2.15, 1, id="sulfur-bond-too-long"),
        pytest.param(1.50, 1, id="sulfur-bond-too-short"),
    ],
)
def test_sidechain_windows_count_affected_residues(sulfur_distance: float, n_bad: int) -> None:
    topology = mdtraj.Topology()
    residue = topology.add_residue("MET", topology.add_chain())
    for name in ("CA", "CB", "CG", "SD", "CE"):
        topology.add_atom(name, _element(name), residue)
    xyz = np.zeros((1, 5, 3))
    xyz[0, :, 0] = np.array([0.0, 1.5, 3.0, 3.0 + sulfur_distance, 3.0 + 2 * sulfur_distance]) / 10

    # Both sulfur bonds fail together, but the residue is counted only once.
    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    assert stats["bad_bonds_sidechain"].tolist() == [n_bad]


@pytest.mark.parametrize(
    "selenium_distance,n_bad",
    [
        pytest.param(1.95, 0, id="selenium-bond-accepted"),
        pytest.param(2.35, 1, id="selenium-bond-too-long"),
    ],
)
def test_selenomethionine_sidechain_bond_window(selenium_distance: float, n_bad: int) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    # A standard residue makes the chain protein rather than a ligand.
    topology.add_atom("CA", mdtraj.element.carbon, topology.add_residue("GLY", chain))
    residue = topology.add_residue("MSE", chain)
    atoms = [
        topology.add_atom(name, element, residue)
        for name, element in (
            ("CB", mdtraj.element.carbon),
            ("CG", mdtraj.element.carbon),
            ("SE", mdtraj.element.selenium),
            ("CE", mdtraj.element.carbon),
        )
    ]
    for first, second in zip(atoms, atoms[1:]):
        topology.add_bond(first, second)
    xyz = np.zeros((1, 5, 3))
    xyz[0, 0, 1] = 1.0
    xyz[0, 1:, 0] = np.array([0.0, 1.5, 1.5 + selenium_distance, 1.5 + 2 * selenium_distance]) / 10

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    assert stats["bad_bonds_sidechain"].tolist() == [n_bad]


def test_too_close_and_isolation_checks_include_other_residues() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for _ in range(2):
        topology.add_atom("CA", mdtraj.element.carbon, topology.add_residue("ALA", chain))
    xyz = np.zeros((2, 2, 3))
    xyz[0, 1, 0] = 0.09
    xyz[1, 1, 0] = 0.15
    trajectory = mdtraj.Trajectory(xyz, topology)

    stats = compute_physicality_stats(trajectory)

    assert stats["within_residue_clashes"].tolist() == [1, 0]
    assert stats["isolated_atoms"].tolist() == [0, 0]
    custom = compute_physicality_stats(trajectory, PhysicalityCutoffs(too_close_distance=0.8))
    assert custom["within_residue_clashes"].tolist() == [0, 0]


@pytest.mark.parametrize("bonded", [True, False], ids=["bonded", "unbonded"])
def test_bonded_atom_within_bond_window_is_not_isolated(bonded: bool) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    topology.add_atom("CA", mdtraj.element.carbon, topology.add_residue("GLY", chain))
    residue = topology.add_residue("MSE", chain)
    selenium = topology.add_atom("SE", mdtraj.element.selenium, residue)
    carbon = topology.add_atom("CE", mdtraj.element.carbon, residue)
    if bonded:
        topology.add_bond(selenium, carbon)
    # The Se-C bond is longer than the isolated-atom distance but inside its bond window.
    xyz = np.array([[[0.0, 1.5, 0.0], [0.0, 0.0, 0.0], [2.1, 0.0, 0.0]]]) / 10.0

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    assert stats["isolated_atoms"].tolist() == [0 if bonded else 1]


def test_valence_counts_neighbors_at_bond_distances_across_residues() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for name in ("O", "CA", "CA", "CA"):
        topology.add_atom(name, _element(name), topology.add_residue("GLY", chain))
    xyz = np.array([[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [-0.15, 0.0, 0.0], [0.0, 0.15, 0.0]]])
    trajectory = mdtraj.Trajectory(xyz, topology)

    assert compute_physicality_stats(trajectory)["valence_exceeded"].tolist() == [1]
    stats = compute_physicality_stats(trajectory, PhysicalityCutoffs(inferred_bond_distance=1.4))
    assert stats["valence_exceeded"].tolist() == [0]


@pytest.mark.parametrize(
    "residue_name,atom_name,cross_chain",
    [
        pytest.param("ALA", "CB", False, id="same-chain-pair-counted-once"),
        pytest.param("GLY", "CA", True, id="cross-chain-glycine-included"),
        pytest.param("GLY", "N", True, id="cross-chain-backbone-included"),
        pytest.param("GLY", "N", False, id="same-chain-backbone-excluded"),
    ],
)
def test_clashes_count_unique_atom_pairs(
    residue_name: str, atom_name: str, cross_chain: bool
) -> None:
    topology = mdtraj.Topology()
    first_chain = topology.add_chain()
    second_chain = topology.add_chain() if cross_chain else first_chain
    for chain in (first_chain, second_chain):
        topology.add_atom(atom_name, _element(atom_name), topology.add_residue(residue_name, chain))
    xyz = np.array([[[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]]])

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    is_sidechain = atom_name == "CB"
    assert stats["same_chain_residue_clashes"].tolist() == [int(is_sidechain and not cross_chain)]
    assert stats["cross_chain_residue_clashes"].tolist() == [int(cross_chain)]


def test_sidechain_clashes_count_each_atom_pair_not_residue_pairs() -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    for _ in range(2):
        residue = topology.add_residue("ARG", chain)
        for name in ("CB", "CG"):
            topology.add_atom(name, mdtraj.element.carbon, residue)
    xyz = np.array([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.15, 0.0], [0.05, 0.15, 0.0]]])

    stats = compute_physicality_stats(mdtraj.Trajectory(xyz, topology))

    assert stats["same_chain_residue_clashes"].tolist() == [4]
    assert stats["cross_chain_residue_clashes"].tolist() == [0]


@pytest.mark.parametrize(
    "partner_residue,same_chain,stat",
    [
        pytest.param("SEP", True, "same_chain_residue_clashes", id="ptm-same-chain"),
        pytest.param("LIG", False, "cross_chain_residue_clashes", id="covalent-ligand"),
    ],
)
@pytest.mark.parametrize("bonded", [True, False], ids=["bonded", "unbonded"])
def test_topology_bonds_are_excluded_from_clashes(
    partner_residue: str, same_chain: bool, stat: str, bonded: bool
) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain()
    sulfur = topology.add_atom("SG", mdtraj.element.sulfur, topology.add_residue("CYS", chain))
    partner_chain = chain if same_chain else topology.add_chain()
    partner = topology.add_atom(
        "P", mdtraj.element.phosphorus, topology.add_residue(partner_residue, partner_chain)
    )
    if bonded:
        topology.add_bond(sulfur, partner)
    # 1.8 angstrom apart: inside the 2.0 angstrom clash cutoff.
    xyz = np.array([[[0.0, 0.0, 0.0], [0.18, 0.0, 0.0]]])

    # Geometry only: the topology check rejects non-standard residue names.
    stats = _compute_geometry_stats(mdtraj.Trajectory(xyz, topology), PhysicalityCutoffs())

    assert stats[stat].tolist() == [0 if bonded else 1]


def _protein_ligand_trajectory(n_frames: int) -> mdtraj.Trajectory:
    sulfurs = [Atom(name="SG", res_name="CYS", res_id=2, asym_id=chain) for chain in "AB"]
    system = System(
        entities=[
            Protein(sequence="ACDW", asym_ids=["A", "B"]),
            Ligand(smiles="CCO", asym_ids=["C"]),
        ],
        bonds=[Bond(atom1=sulfurs[0], atom2=sulfurs[1])],
    )
    structure = AllAtomStructure.from_system(system, mode="equilibrium")
    coords = structure.coords.numpy() / 10.0
    rng = np.random.default_rng(0)
    xyz = np.repeat(coords[None], n_frames, axis=0)
    # Overlap the chains and the ligand, then jitter atoms so every check is exercised.
    for chain in list(system.topology.chains)[1:]:
        indices = [atom.index for atom in chain.atoms]
        xyz[:, indices] += rng.uniform(-0.3, 0.3, size=3) - coords[indices].mean(0) + coords.mean(0)
    xyz += rng.normal(scale=0.03, size=xyz.shape)
    xyz[::3, 0] += 5.0  # Isolate one atom in some frames.
    return mdtraj.Trajectory(xyz, system.topology)


def test_ligand_bonds_are_excluded_from_topology_checks() -> None:
    trajectory = _protein_ligand_trajectory(1)
    protein = trajectory.atom_slice(get_protein_atom_indices(trajectory.topology))

    assert _check_topology(trajectory) == _check_topology(protein)


def test_chunked_geometry_stats_match_per_frame_neighbor_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = _protein_ligand_trajectory(20)
    trajectory.xyz[5, 3] = np.nan
    cutoffs = PhysicalityCutoffs()

    monkeypatch.setattr(physicality, "_FRAME_CHUNK_SIZE", 1)
    expected = _compute_geometry_stats(trajectory, cutoffs)
    monkeypatch.setattr(physicality, "_FRAME_CHUNK_SIZE", 7)
    stats = _compute_geometry_stats(trajectory, cutoffs)

    for key, values in expected.items():
        np.testing.assert_array_equal(stats[key], values, err_msg=key)
        assert values.any(), key


@pytest.mark.parametrize(
    ("sequence", "chain_ids", "partner", "key"),
    [
        ("CAC", ["A"], Atom(name="SG", res_name="CYS", res_id=3, asym_id="A"), "same_chain"),
        ("C", ["A", "B"], Atom(name="SG", res_name="CYS", res_id=1, asym_id="B"), "cross_chain"),
    ],
)
@pytest.mark.parametrize(
    ("distance_angstrom", "undeclared_clashes"),
    [(1.75, 1), (1.90, 0)],
    ids=["too-short", "disulfide-like"],
)
def test_disulfide_like_pair_is_not_a_residue_clash_with_or_without_declaration(
    sequence: str,
    chain_ids: list[str],
    partner: Atom,
    key: str,
    distance_angstrom: float,
    undeclared_clashes: int,
) -> None:
    sg = Atom(name="SG", res_name="CYS", res_id=1, asym_id="A")
    entities = [Protein(sequence=sequence, asym_ids=chain_ids)]
    declared = System(entities=entities, bonds=[Bond(atom1=sg, atom2=partner)])
    ((first, second),) = expected_extra_bonds_from_system(declared, declared.topology)
    # Atoms 1 nm apart, except the two sulfurs at the tested distance.
    xyz = np.zeros((1, declared.topology.n_atoms, 3))
    xyz[0, :, 0] = np.arange(declared.topology.n_atoms)
    xyz[0, second] = xyz[0, first] + [0.0, distance_angstrom / 10, 0.0]

    stats = {
        name: _compute_geometry_stats(mdtraj.Trajectory(xyz, s.topology), PhysicalityCutoffs())
        for name, s in [("declared", declared), ("undeclared", System(entities=entities))]
    }

    assert stats["undeclared"][f"{key}_residue_clashes"].tolist() == [undeclared_clashes]
    assert stats["declared"][f"{key}_residue_clashes"].tolist() == [0]


@pytest.mark.parametrize(
    ("third_position", "declared", "expected_clashes"),
    [
        pytest.param(3.8, False, 2, id="two-unlabelled-partners"),
        pytest.param(-1.9, True, 1, id="declared-and-unlabelled-partners"),
        pytest.param(-2.2, False, 1, id="competing-partner-beyond-clash-radius"),
    ],
)
def test_unlabelled_disulfide_needs_unique_cysteine_sulfur_partners(
    third_position: float, declared: bool, expected_clashes: int
) -> None:
    topology = mdtraj.Topology()
    chain = topology.add_chain("A")
    sulfurs = [
        topology.add_atom("SG", mdtraj.element.sulfur, topology.add_residue("CYS", chain))
        for _ in range(3)
    ]
    if declared:
        topology.add_bond(sulfurs[0], sulfurs[1])
    xyz = np.zeros((1, topology.n_atoms, 3))
    xyz[0, :, 0] = np.array([0.0, 1.9, third_position]) / 10.0

    stats = _compute_geometry_stats(mdtraj.Trajectory(xyz, topology), PhysicalityCutoffs())

    assert stats["same_chain_residue_clashes"].tolist() == [expected_clashes]


def test_physicality_filter_ignores_smiles_ligands() -> None:
    system = System(
        entities=[Protein(sequence="GG", asym_ids=["A"]), Ligand(smiles="CCO", asym_ids=["B"])]
    )
    structure = AllAtomStructure.from_system(system)
    assert structure.template_coords is not None
    trajectory = mdtraj.Trajectory(structure.template_coords.numpy()[None] / 10.0, system.topology)
    assert list(trajectory.topology.residues)[-1].name == "UNK"

    kept, rejected = apply_physicality_filter(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    filtered = remove_unphysical_frames(trajectory, PhysicalityFilterMode.HIGH_QUALITY)

    assert kept.n_frames == filtered.n_frames == 1
    assert rejected.n_frames == 0
    np.testing.assert_array_equal(kept.xyz, trajectory.xyz)
    np.testing.assert_array_equal(filtered.xyz, trajectory.xyz)


def test_physicality_filters_reject_non_finite_frames() -> None:
    system = System(entities=[Protein(sequence="GG", asym_ids=["A"])])
    structure = AllAtomStructure.from_system(system)
    assert structure.template_coords is not None
    xyz = np.repeat(structure.template_coords.numpy()[None] / 10.0, 4, axis=0)
    xyz[1, 0, 0] = np.nan
    xyz[2, 1, 2] = np.inf
    trajectory = mdtraj.Trajectory(xyz, system.topology)

    is_physical, stats, reasons = physicality.check_allatom_physicality(
        trajectory, mode=PhysicalityFilterMode.HIGH_QUALITY
    )
    kept, rejected = apply_physicality_filter(trajectory, PhysicalityFilterMode.HIGH_QUALITY)
    filtered = remove_unphysical_frames(trajectory, PhysicalityFilterMode.HIGH_QUALITY)

    assert is_physical.tolist() == [True, False, False, True]
    assert stats["non_finite_atoms"].tolist() == [0, 1, 1, 0]
    assert reasons == [None, "non_finite_atoms", "non_finite_atoms", None]
    assert kept.n_frames == filtered.n_frames == 2
    assert rejected.n_frames == 2
    np.testing.assert_array_equal(filtered.xyz, trajectory.xyz[[0, 3]])


@pytest.mark.parametrize("residue_numbers", [(0, 1), (1, 2), (42, 99)])
@pytest.mark.parametrize("chain_ids", [["A"], ["A", "B"]])
def test_topology_check_ignores_residue_numbering(residue_numbers, chain_ids) -> None:
    topology = System(entities=[Protein(sequence="GG", asym_ids=chain_ids)]).topology
    for chain in topology.chains:
        for residue, number in zip(chain.residues, residue_numbers, strict=True):
            residue.resSeq = number
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    assert _check_topology(trajectory) == {
        "duplicate_atoms": 0,
        "atoms_mismatch": 0,
        "bonds_mismatch": 0,
    }


def test_topology_check_after_cif_roundtrip(tmp_path: Path) -> None:
    system = System(entities=[Protein(sequence="GG", asym_ids=["A", "B"])])
    structure = AllAtomStructure.from_system(system, mode="equilibrium")
    path = tmp_path / "topology.cif"
    save_topology_cif(structure, system, structure.coords, path)
    trajectory = mdtraj.load(str(path))

    assert _check_topology(trajectory) == {
        "duplicate_atoms": 0,
        "atoms_mismatch": 0,
        "bonds_mismatch": 0,
    }


@pytest.mark.parametrize("missing_bond", [False, True])
def test_topology_check_detects_incorrect_bonds(missing_bond: bool) -> None:
    topology = System(entities=[Protein(sequence="GG", asym_ids=["A", "B"])]).topology
    for chain in topology.chains:
        for residue in chain.residues:
            residue.resSeq = 42
    if missing_bond:
        topology._bonds.pop(0)
    else:
        topology.add_bond(topology.atom(0), topology.atom(8))
    trajectory = mdtraj.Trajectory(np.zeros((1, topology.n_atoms, 3)), topology)

    assert _check_topology(trajectory)["bonds_mismatch"] == 1
