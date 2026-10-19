"""Bonded geometry graph and reference-bound regressions."""

import math
from itertools import combinations, product

import pytest
import torch
import yaml

from bioemu2.data_interface import DNA, RNA, Atom, Bond, Ligand, ModifiedResidue, Protein, System
from bioemu2.denoising.steering.geometry_topology import (
    ANGLE_FALLBACK,
    BOND_FALLBACK,
    DISULFIDE_ANGLE,
    StereoTables,
    _canonical_atom_order,
    _stereo_lookup,
    build_bonded_geometry_graph,
    compute_geometry_bounds,
    get_angle_reference,
    get_bond_reference,
    load_stereo_tables,
    pack_angle_constraints,
    pack_bond_constraints,
)
from bioemu2.paths import ASSETS_DIR
from bioemu2.structure import AllAtomStructure


def _structure(system):
    return AllAtomStructure.from_system(system, mode="structure")


@pytest.mark.parametrize("max_bonds", [1, 2, 3])
def test_nonbonded_exclusion_depth(max_bonds):
    """Exclude atom pairs only when their shortest covalent path is within the requested depth."""
    graph = build_bonded_geometry_graph(
        _structure(System(entities=[Protein(sequence="G", asym_ids="A")]))
    )
    atoms = {name: i for i, (_, name) in graph.atom_labels.items()}
    excluded = graph.nonbonded_exclusions(max_bonds=max_bonds)
    for name, distance in (("CA", 1), ("C", 2), ("O", 3)):
        pair = tuple(sorted((atoms["N"], atoms[name])))
        assert (pair in excluded) == (distance <= max_bonds)


def test_loaded_references_match_source_in_both_directions():
    """Retrieve every packaged bond and angle reference unchanged in either atom direction."""
    source = yaml.safe_load((ASSETS_DIR / "stereo_chemistry.yaml").read_text())
    tables = load_stereo_tables()
    for table, entries in (
        (tables.bonds, source["monlib"]["bond_data"]),
        (tables.angles, source["monlib"]["angle_data"]),
        (tables.link_bonds, source["monlib_link"]["bond_data"]),
        (tables.link_angles, source["monlib_link"]["angle_data"]),
    ):
        for residue, references in entries.items():
            for key, (mean, sigma) in references.items():
                names = tuple(key.split("_"))
                assert _stereo_lookup(table, residue, names) == (mean, sigma)
                assert _stereo_lookup(table, residue, names[::-1]) == (mean, sigma)


def test_angle_lookup_preserves_center_atom():
    """Reversing an N-CA-C angle keeps its reference, but moving its center to C does not."""
    tables = load_stereo_tables()
    reference = _stereo_lookup(tables.angles, "ALA", ("N", "CA", "C"))
    assert reference is not None
    assert _stereo_lookup(tables.angles, "ALA", ("C", "CA", "N")) == reference
    assert _stereo_lookup(tables.angles, "ALA", ("N", "C", "CA")) is None


@pytest.mark.parametrize(
    "polymer,alphabet", [(Protein, "ACDEFGHIKLMNPQRSTVWYGP"), (DNA, "ACGT"), (RNA, "ACGU")]
)
def test_canonical_graph_and_references(polymer, alphabet):
    """Check that normal proteins, DNA and RNA get the expected angles and reference bounds."""
    sequence = (
        alphabet if polymer is Protein else "".join(map("".join, product(alphabet, repeat=2)))
    )
    structure = _structure(System(entities=[polymer(sequence=sequence, asym_ids="A")]))
    graph = build_bonded_geometry_graph(structure)
    unresolved = build_bonded_geometry_graph(
        structure.replace(atom_mask=torch.zeros_like(structure.atom_mask))
    )
    assert unresolved.bonds == graph.bonds and unresolved.angles == graph.angles
    neighbors: dict[int, set[int]] = {}
    for i, j in graph.bonds:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)
    assert set(graph.angles) == {
        (i, k, center)
        for center, adjacent in neighbors.items()
        for i, k in combinations(sorted(adjacent), 2)
    }
    assert len(graph.angles) == len(set(graph.angles))
    tables = load_stereo_tables()
    source = yaml.safe_load((ASSETS_DIR / "stereo_chemistry.yaml").read_text())
    for angle, measurements in ((False, graph.bonds), (True, graph.angles)):
        assert measurements
        for indices in measurements:
            ordered = (indices[0], indices[2], indices[1]) if angle else indices
            identities = {graph.atom_residue[i] for i in indices}
            identity = graph.atom_labels[max(indices)][0]
            if len(identities) > 1:
                identity = (
                    {"GLY": "GLY", "PRO": "PRO_TRANS"}.get(identity, "PEPTIDE")
                    if polymer is Protein
                    else "NA"
                )
            table = source["monlib_link" if len(identities) > 1 else "monlib"][
                "angle_data" if angle else "bond_data"
            ][identity]
            names = tuple(graph.atom_labels[i][1] for i in ordered)
            key = "_".join(names)
            mean, sigma = table[key] if key in table else table["_".join(names[::-1])]
            if angle:
                actual = get_angle_reference(graph, *ordered, tables, {})
            else:
                actual = get_bond_reference(graph, *ordered, tables, {})
            assert actual == pytest.approx((mean, mean, sigma))
            bounds = (max(0, mean - 10 * sigma), mean + 10 * sigma)
            if angle:
                bounds = tuple(map(math.radians, (bounds[0], min(180, bounds[1]))))
            assert compute_geometry_bounds(actual, 10, angle=angle) == pytest.approx(
                bounds, abs=1e-6
            )


@pytest.mark.parametrize("unknown_cross_link", [False, True])
def test_disulfide_uses_reference_and_unmatched_cross_link_uses_fallback(unknown_cross_link):
    """Use the disulfide reference for an S-S bond and broad defaults for an unmatched cross-link."""
    system = System(
        entities=[
            Protein(sequence="CX", asym_ids="A"),
            Protein(sequence="C", asym_ids="B"),
            Ligand(smiles="CCO", asym_ids="L"),
        ]
    )
    structure = _structure(system)
    baseline = build_bonded_geometry_graph(structure)
    sulfurs = [i for i, label in baseline.atom_labels.items() if label == ("CYS", "SG")]
    source = (
        next(i for i in baseline.ordinary_x if baseline.atom_labels[i][1] == "CB")
        if unknown_cross_link
        else sulfurs[1]
    )
    target = next(
        i
        for i, (_, name) in baseline.atom_labels.items()
        if name == ("O" if unknown_cross_link else "O1")
    )
    disulfide, cross = tuple(sorted(sulfurs)), tuple(sorted((source, target)))
    links = structure.replace(
        bond_index=torch.cat(
            (
                structure.bond_index,
                torch.tensor(
                    [
                        [sulfurs[0], source],
                        [sulfurs[1], target],
                    ]
                ),
            ),
            dim=1,
        ),
        bond_type=torch.cat((structure.bond_type, torch.ones(2, dtype=torch.long))),
    )
    graph = build_bonded_geometry_graph(links)
    assert set(graph.bonds) == set(baseline.bonds) | {disulfide, cross}
    tables = load_stereo_tables()
    assert get_bond_reference(graph, *disulfide, tables, {}) == pytest.approx((2.033, 2.033, 0.016))
    assert get_bond_reference(graph, *cross, tables, {}) == BOND_FALLBACK
    assert graph.is_disulfide(*disulfide) and not graph.is_disulfide(*cross)
    # Both CB-SG-SG' angles use the disulfide reference.
    for center, partner in (sulfurs, sulfurs[::-1]):
        cb = next(
            i
            for i, label in graph.atom_labels.items()
            if label == ("CYS", "CB") and graph.atom_residue[i] == graph.atom_residue[center]
        )
        reference = get_angle_reference(graph, cb, center, partner, tables, {})
        assert reference == (DISULFIDE_ANGLE[0], *DISULFIDE_ANGLE)
    angle = next(
        (i, center, k) for i, k, center in graph.angles if target in (i, k) and center == source
    )
    assert get_angle_reference(graph, *angle, tables, {}) == ANGLE_FALLBACK


@pytest.mark.parametrize(
    "polymer,center_spec,supplied_spec,standard_spec",
    [
        (Protein, ("N", 2), ("CA", 3), ("C", 1)),
        (DNA, ("O3'", 1), ("C3'", 3), ("P", 2)),
    ],
    ids=["protein", "nucleotide"],
)
def test_cross_link_to_another_residue_does_not_use_backbone_angle_reference(
    polymer, center_spec, supplied_spec, standard_spec
):
    """Do not mistake an angle reaching into another residue for a normal backbone angle."""
    entity = polymer(sequence="AAA", asym_ids="A")
    structure = _structure(System(entities=[entity]))
    indices = {
        (name, residue + 1): i
        for i, (name, residue) in enumerate(
            zip(structure.atom_names, structure.residue_idx.tolist())
        )
    }
    center, supplied, standard = (
        indices[spec] for spec in (center_spec, supplied_spec, standard_spec)
    )

    def atom(index):
        return Atom(
            name=structure.atom_names[index],
            res_id=int(structure.residue_idx[index]) + 1,
            res_name=structure.residue_labels[index],
            asym_id="A",
        )

    system = System(entities=[entity], bonds=[Bond(atom1=atom(center), atom2=atom(supplied))])
    graph = build_bonded_geometry_graph(_structure(system))
    assert (*sorted((standard, supplied)), center) in graph.angles
    assert (
        get_angle_reference(graph, standard, center, supplied, load_stereo_tables(), {})
        == ANGLE_FALLBACK
    )


@pytest.mark.parametrize("angle", [False, True])
@pytest.mark.parametrize("link", [False, True])
def test_unknown_protein_pools_mean_extrema_and_maximum_sigma(angle, link):
    """For unknown X, combine the extreme reference means and largest standard deviation."""
    graph = build_bonded_geometry_graph(
        _structure(System(entities=[Protein(sequence="AXXG", asym_ids="A")]))
    )
    names = (
        (("C", "N", "CA") if link else ("N", "CA", "C"))
        if angle
        else (("C", "N") if link else ("N", "CA"))
    )
    table = {
        "GLY": {_canonical_atom_order(names): (100.0, 1.0)},
        "PRO_TRANS" if link else "ALA": {_canonical_atom_order(names): (110.0, 3.0)},
    }
    tables = StereoTables(bonds=table, angles=table, link_bonds=table, link_angles=table)
    pooled = {}
    matched = []
    for indices in graph.angles if angle else graph.bonds:
        ordered = (indices[0], indices[2], indices[1]) if angle else indices
        if tuple(graph.atom_labels[i][1] for i in ordered) not in (names, names[::-1]):
            continue
        if graph.ordinary_x.intersection(indices):
            actual = (
                get_angle_reference(graph, *ordered, tables, pooled)
                if angle
                else get_bond_reference(graph, *ordered, tables, pooled)
            )
            assert actual == (100.0, 110.0, 3.0)
            reversed_reference = (
                get_angle_reference(graph, *ordered[::-1], tables, pooled)
                if angle
                else get_bond_reference(graph, *ordered[::-1], tables, pooled)
            )
            assert reversed_reference == actual
            matched.append(indices)
    assert len(matched) == (3 if link else 2)
    assert len(pooled) == 1


@pytest.mark.parametrize(
    "tolerance,bond_bounds,angle_bounds",
    [(0, (1.1, 2.2), (60, 180)), (10, (0.8, 2.5), (30, 180)), (100, (0, 5.2), (0, 180))],
)
def test_ligand_graph_fallback_bounds_and_packed_constraints(tolerance, bond_bounds, angle_bounds):
    """Keep a small ligand's bonds and angle, widen their default bounds and pack the results."""
    structure = _structure(System(entities=[Ligand(smiles="CCO", asym_ids="L")]))
    graph = build_bonded_geometry_graph(structure)
    assert graph.bonds == [(0, 1), (1, 2)] and graph.angles == [(0, 2, 1)]
    tables = load_stereo_tables()
    bonds = [
        (i, j, compute_geometry_bounds(get_bond_reference(graph, i, j, tables, {}), tolerance))
        for i, j in graph.bonds
    ]
    angles = [
        (
            i,
            k,
            center,
            compute_geometry_bounds(
                get_angle_reference(graph, i, center, k, tables, {}), tolerance, angle=True
            ),
        )
        for i, k, center in graph.angles
    ]
    index, lower, upper = pack_bond_constraints(bonds, structure.coords.device)
    assert index.tolist() == [[[0, 1], [1, 2]]]
    torch.testing.assert_close(lower, torch.full((1, 2), float(bond_bounds[0])))
    torch.testing.assert_close(upper, torch.full((1, 2), float(bond_bounds[1])))
    index, lower, upper = pack_angle_constraints(angles, structure.coords.device)
    assert index.tolist() == [[[0], [2], [1]]]
    assert lower.item() == pytest.approx(math.radians(angle_bounds[0]), abs=1e-6)
    assert upper.item() == pytest.approx(math.radians(angle_bounds[1]), abs=1e-6)


def test_single_atom_has_no_bonded_constraints():
    """Return correctly shaped empty bond and angle arrays for a ligand with only one atom."""
    structure = _structure(System(entities=[Ligand(smiles="C", asym_ids="L")]))
    graph = build_bonded_geometry_graph(structure)
    assert not graph.bonds and not graph.angles
    for pack, arity in ((pack_bond_constraints, 2), (pack_angle_constraints, 3)):
        index, lower, upper = pack([], structure.coords.device)
        assert index.shape == (1, arity, 0)
        assert lower.shape == upper.shape == (1, 0)
        assert index.dtype == torch.long and lower.dtype == upper.dtype == torch.float32


def test_d_amino_acids_use_l_enantiomer_references():
    """Give a D-peptide, including D-proline links and a D-cysteine disulfide, L references."""

    def references(sequence, mods, cys_res_name):
        bonds = [
            Bond(
                atom1=Atom(name="SG", res_id=4, res_name=cys_res_name, asym_id="A"),
                atom2=Atom(name="SG", res_id=7, res_name=cys_res_name, asym_id="A"),
            )
        ]
        entity = Protein(sequence=sequence, asym_ids="A", mods=mods)
        graph = build_bonded_geometry_graph(_structure(System(entities=[entity], bonds=bonds)))
        tables = load_stereo_tables()

        def key(indices):
            return _canonical_atom_order(
                tuple((graph.atom_residue[i][1], graph.atom_labels[i][1]) for i in indices)
            )

        bond_refs = {
            key((i, j)): get_bond_reference(graph, i, j, tables, {}) for i, j in graph.bonds
        }
        angle_refs = {
            key((i, center, k)): get_angle_reference(graph, i, center, k, tables, {})
            for i, k, center in graph.angles
        }
        return bond_refs, angle_refs

    d_codes = {2: "DLY", 3: "DPR", 4: "DCY", 5: "DLE", 7: "DCY"}
    d_bonds, d_angles = references(
        "GXXXXGX",
        [ModifiedResidue(res_id=res_id, ccd_code=code) for res_id, code in d_codes.items()],
        "DCY",
    )
    l_bonds, l_angles = references("GKPCLGC", [], "CYS")
    assert d_bonds == l_bonds
    assert d_angles == l_angles
    assert BOND_FALLBACK not in d_bonds.values()
    assert ANGLE_FALLBACK not in d_angles.values()


def test_modified_residue_is_distinct_from_ordinary_unknown():
    """Keep modified serine's identity and internal bonds instead of treating it as unknown X."""
    structure = _structure(
        System(
            entities=[
                Protein(
                    sequence="AXXAA", asym_ids="A", mods=[ModifiedResidue(res_id=3, ccd_code="SEP")]
                )
            ]
        )
    )
    graph = build_bonded_geometry_graph(structure)
    modified = {i for i, (residue, _) in graph.atom_labels.items() if residue == "SEP"}
    assert modified and graph.ordinary_x and modified.isdisjoint(graph.ordinary_x)
    assert {graph.atom_residue[i][1] for i in graph.ordinary_x} == {1}
    assert {graph.atom_residue[i][1] for i in modified} == {2}
    assert any(i in modified and j in modified for i, j in graph.bonds)
