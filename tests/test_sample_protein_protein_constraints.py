import ast
from pathlib import Path

import mdtraj
import numpy as np
import pytest
import torch

from bioemu2 import MSA
from bioemu2.structure.preparation import prepare_structure


@pytest.fixture(scope="module", params=["script", "notebook"])
def tutorial(request: pytest.FixtureRequest):
    path = Path(__file__).parents[1] / "examples" / "sample_protein_protein_constraints.py"
    nodes = ast.parse(path.read_text()).body
    setup_end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.FunctionDef))
    # CI tests numerical helpers without the optional visualization dependencies.
    setup = [
        node
        for node in nodes[:setup_end]
        if not (
            isinstance(node, ast.Import)
            and any(alias.name == "matplotlib.pyplot" for alias in node.names)
        )
    ]
    setup.extend(node for node in nodes if isinstance(node, ast.FunctionDef))
    namespace = {"__file__": str(path)} if request.param == "script" else {}
    with pytest.MonkeyPatch.context() as patcher:
        patcher.chdir(path.parents[1])
        exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_fixed_contacts_match_reference_and_cpu_features(tutorial):
    system = tutorial["system"]
    reference = tutorial["reference"]
    contacts = tutorial["contacts"]
    residues = {chain.chain_id: list(chain.residues) for chain in reference.topology.chains}
    assert [entity.sequence for entity in system.entities] == reference.topology.to_fasta()
    assert len(contacts) == 3
    assert len({contact.token1.res_id for contact in contacts}) == 3
    assert len({contact.token2.res_id for contact in contacts}) == 3
    for contact in contacts:
        residue_a = residues[contact.token1.asym_id][contact.token1.res_id - 1]
        residue_b = residues[contact.token2.asym_id][contact.token2.res_id - 1]
        ca_a = next(atom.index for atom in residue_a.atoms if atom.name == "CA")
        ca_b = next(atom.index for atom in residue_b.atoms if atom.name == "CA")
        ca_distance = np.linalg.norm(reference.xyz[0, ca_a] - reference.xyz[0, ca_b]) * 10
        assert 1 <= contact.max_distance - ca_distance <= 1.5
        heavy_a = [atom.index for atom in residue_a.atoms if atom.element.symbol != "H"]
        heavy_b = [atom.index for atom in residue_b.atoms if atom.element.symbol != "H"]
        closest = (
            np.linalg.norm(
                reference.xyz[0, heavy_a, None] - reference.xyz[0, heavy_b], axis=-1
            ).min()
            * 10
        )
        assert closest <= 5

    msa = MSA(
        {
            entity.asym_ids[0]: f">query_{entity.asym_ids[0]}\n{entity.sequence}\n"
            for entity in system.entities
        }
    )
    for condition_contacts in ([], contacts):
        structure = prepare_structure(
            system, msa=msa, mode="structure", contacts=condition_contacts
        )
        assert structure.num_tokens == sum(map(len, reference.topology.to_fasta()))
        if condition_contacts:
            thresholds = structure.contact_threshold
            assert torch.count_nonzero(thresholds) == 6
            for contact in condition_contacts:
                i = contact.token1.res_id - 1
                j = len(system.entities[0].sequence) + contact.token2.res_id - 1
                assert thresholds[i, j] == contact.max_distance
                assert thresholds[j, i] == contact.max_distance
        else:
            assert not hasattr(structure, "contact_threshold")


def test_complex_rmsd_ignores_numbering_and_rigid_motion(tutorial):
    reference = tutorial["reference"]
    topology = reference.topology.copy()
    for chain in topology.chains:
        for index, residue in enumerate(chain.residues):
            residue.resSeq = index
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transformed = mdtraj.Trajectory(reference.xyz @ rotation + np.array([2.1, -3.2, 0.8]), topology)
    original = transformed.xyz.copy()
    np.testing.assert_allclose(tutorial["complex_rmsd"](transformed, reference), 0, atol=0.01)
    np.testing.assert_array_equal(transformed.xyz, original)

    displaced = reference[:]
    binder = [atom.index for atom in displaced.topology.atoms if atom.residue.chain.chain_id == "B"]
    displaced.xyz[:, binder, 0] += 0.5
    assert tutorial["complex_rmsd"](displaced, reference)[0] > 0.1
