"""Steric pair selection, energies, gradients, and numerical regressions."""

import gzip
from itertools import combinations
from pathlib import Path

import pytest
import torch

from bioemu2.data_interface import DNA, RNA, Atom, Bond, Ligand, ModifiedResidue, Protein, System
from bioemu2.denoising.steering import potentials
from bioemu2.denoising.steering.potentials import (
    InterChainVDWOverlapPotential,
    IntraChainVDWOverlapPotential,
    PiecewiseStepFunction,
    PolymerLocalClashPotential,
    SymmetricChainCOMPotential,
    _steric_overlap_penalty,
    compute_atom_pairs_within_cutoff,
)
from bioemu2.structure import AllAtomStructure
from bioemu2.structure.inference_inputs import topology_from_structure

_PARAMS = {
    InterChainVDWOverlapPotential: {"buffer": 0.225},
    IntraChainVDWOverlapPotential: {"buffer": 0.225},
    PolymerLocalClashPotential: {"minimum_distance": 2.0},
}


def _evaluated(potential, coords):
    coords = coords.detach().clone().requires_grad_()
    energy = potential(coords, t=0.5)
    return energy.detach(), torch.autograd.grad(energy.sum(), coords)[0]


def _dense_pairs(structure, cls):
    topology = topology_from_structure(structure)
    atoms = list(topology.atoms)
    bonds = {tuple(sorted((a.index, b.index))) for a, b in topology.bonds}
    bonds.update(tuple(sorted(pair)) for pair in structure.bond_index.T.tolist())
    adjacency = torch.eye(len(atoms))
    i, j = torch.tensor(sorted(bonds)).T
    adjacency[i, j] = adjacency[j, i] = 1
    # Self-loops include paths shorter than three edges in the third matrix power.
    near = adjacency.matrix_power(3) > 0
    chains = [a.residue.chain.index for a in atoms]
    eligible = [a.residue.name not in ("SEP", "NA") for a in atoms]
    pairs = []
    for i, j in combinations(range(len(atoms)), 2):
        if cls is InterChainVDWOverlapPotential:
            keep = chains[i] != chains[j] and not near[i, j]
            keep &= all(atoms[k].residue.chain.n_atoms > 1 for k in (i, j))
        else:
            keep = eligible[i] and eligible[j]
            if cls is IntraChainVDWOverlapPotential:
                keep &= chains[i] == chains[j] and not near[i, j]
            else:
                # Local pairs may cross chains through a declared bond.
                keep &= near[i, j] and (i, j) not in bonds and atoms[i].residue != atoms[j].residue
        if keep:
            pairs.append((i, j))
    return torch.tensor(pairs, dtype=torch.long).T


@pytest.mark.parametrize("cls", list(_PARAMS))
@pytest.mark.parametrize("entity_type,sequence", [(Protein, "CXSC"), (DNA, "ACG"), (RNA, "ACG")])
@pytest.mark.parametrize("dtype,block_size", [(torch.float32, 7), (torch.float64, 512)])
def test_dense_energy_and_gradient(cls, entity_type, sequence, dtype, block_size):
    """Test that energies and gradients match an all-pairs calculation, including padded inputs."""
    mods = [ModifiedResidue(res_id=3, ccd_code="SEP")] if entity_type is Protein else []
    entity = entity_type(sequence=sequence, asym_ids=["A", "B", "C"], mods=mods)
    bonds = []
    if entity_type is Protein:
        ends = [
            Atom(asym_id=c, res_id=r, res_name="CYS", name="SG")
            for c, r in [("A", 1), ("A", 4), ("B", 4)]
        ]
        bonds = [Bond(atom1=ends[0], atom2=ends[1]), Bond(atom1=ends[1], atom2=ends[2])]
    system = System(entities=[entity, Ligand(smiles="[Na+]", asym_ids="D")], bonds=bonds)
    structure = AllAtomStructure.from_system(system, mode="structure")
    potential = cls(parameters=_PARAMS[cls])
    potential.block_size = block_size
    potential.prepare(system)
    i, j = _dense_pairs(structure, cls)
    lower = 2.0
    if cls is not PolymerLocalClashPotential:
        radii = potential.feats["atom_vdw_radius"][0]
        lower = (radii[i] + radii[j]) * 0.775
    coords = torch.randn(
        2, structure.num_nodes + 5, 3, dtype=dtype, generator=torch.Generator().manual_seed(73)
    )
    coords[0] *= 50
    coords.requires_grad_()
    distances = (coords[:, i] - coords[:, j]).square().sum(-1).add(1e-12).sqrt()
    expected = (lower - distances).clamp_min(0).sum(-1)
    energy, gradient = _evaluated(potential, coords)
    assert expected.sum() > 0
    torch.testing.assert_close(energy, expected, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(
        gradient, torch.autograd.grad(expected.sum(), coords)[0], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("cls", [IntraChainVDWOverlapPotential, PolymerLocalClashPotential])
def test_crystal_and_perturbation(cls, tmp_path):
    """Test that crystal coordinates have zero penalty and gradient descent reduces added clashes."""
    cif = tmp_path / "101m.cif"
    cif.write_bytes(gzip.decompress((Path(__file__).parent / "fixtures/101m.cif.gz").read_bytes()))
    system = System.from_cif(cif)
    coords = AllAtomStructure.from_system(system, ref_struct_or_fpath=cif).coords[None]
    potential = cls(parameters=_PARAMS[cls])
    potential.prepare(system)
    energy, gradient = _evaluated(potential, coords)
    assert not energy.any() and not gradient.any()
    noisy = coords + 0.3 * torch.randn(coords.shape, generator=torch.Generator().manual_seed(0))
    energy, gradient = _evaluated(potential, noisy)
    assert energy.item() > 0 and torch.isfinite(gradient).all()
    assert _evaluated(potential, noisy - 0.01 * gradient)[0].item() < energy.item()


@pytest.mark.parametrize("cls", list(_PARAMS))
@pytest.mark.parametrize(
    "entity",
    [
        Protein(sequence="G", asym_ids="A"),
        DNA(sequence="N", asym_ids="A"),
        RNA(sequence="N", asym_ids="A"),
        RNA(sequence="UN", asym_ids="A", mods=[ModifiedResidue(res_id=1, ccd_code="PSU")]),
        Ligand(smiles="[Na+]", asym_ids=["A", "B"]),
    ],
)
def test_reprepare_empty_restraints(cls, entity):
    """Test that energies and gradients become zero after preparing a system with no relevant pairs."""
    potential = cls(parameters=_PARAMS[cls])
    potential.prepare(System(entities=[Protein(sequence="AA", asym_ids=["A", "B"])]))
    potential.prepare(System(entities=[entity]))
    potential.cutoff = 10.0 if isinstance(entity, Protein) else 0.1
    energy, gradient = _evaluated(
        potential, torch.zeros(2, potential.feats["atom_pad_mask"].shape[1], 3)
    )
    assert not energy.any() and not gradient.any()


@pytest.mark.parametrize("cls", [IntraChainVDWOverlapPotential, PolymerLocalClashPotential])
def test_supplied_linker_preserves_covalent_paths(cls):
    """Test that a ligand bridge excludes the linked atoms from VDW but allows local-clash penalties."""
    entities = [Protein(sequence="KK", asym_ids="A"), Ligand(smiles="C", asym_ids="L")]
    structure = AllAtomStructure.from_system(System(entities=entities), mode="structure")
    linker = Atom(
        asym_id="L", res_id=1, res_name=structure.residue_labels[-1], name=structure.atom_names[-1]
    )
    bonds = [
        Bond(atom1=Atom(asym_id="A", res_id=r, res_name="LYS", name="NZ"), atom2=linker)
        for r in (1, 2)
    ]
    potential = cls(parameters=_PARAMS[cls])
    potential.prepare(System(entities=entities, bonds=bonds))
    ends = [i for i, name in enumerate(structure.atom_names) if name == "NZ"]
    coords = torch.zeros(2, structure.num_nodes, 3)
    coords[:, :, 0] = 100 * (torch.arange(structure.num_nodes) + 1)
    coords[:, ends[0], 0], coords[:, ends[1], 0] = 0, torch.tensor([1.5, 2.4])
    energy, gradient = _evaluated(potential, coords)
    expected = torch.tensor([0.5, 0.0]) if cls is PolymerLocalClashPotential else torch.zeros(2)
    torch.testing.assert_close(energy, expected)
    expected_gradient = torch.zeros_like(coords)
    expected_gradient[0, ends, 0] = torch.tensor([1.0, -1.0]) if expected[0] else 0
    torch.testing.assert_close(gradient, expected_gradient)


def test_interchain_bond_excludes_only_nearby_pairs():
    """Test that a declared inter-chain disulfide skips SG-SG but still penalises other pairs
    between the linked chains, such as overlapping CA atoms."""
    sg = [Atom(asym_id=c, res_id=2, res_name="CYS", name="SG") for c in "AB"]
    system = System(
        entities=[Protein(sequence="AC", asym_ids=["A", "B"])],
        bonds=[Bond(atom1=sg[0], atom2=sg[1])],
    )
    potential = InterChainVDWOverlapPotential(parameters=_PARAMS[InterChainVDWOverlapPotential])
    potential.prepare(system)
    atom = {
        (a.residue.chain.chain_id, a.residue.name, a.name): a.index for a in system.topology.atoms
    }
    coords = torch.zeros(1, system.topology.n_atoms, 3)
    coords[0, :, 0] = 100 * torch.arange(system.topology.n_atoms)  # every pair far apart
    for name, distance in (("SG", 2.05), ("CA", 0.5)):  # bonded SG-SG; clashing CA-CA
        first, second = atom["A", "CYS", name], atom["B", "CYS", name]
        coords[0, second] = coords[0, first] + torch.tensor([0.0, distance, 0.0])
    energy, _ = _evaluated(potential, coords)
    torch.testing.assert_close(energy, torch.tensor([2 * 1.7 * 0.775 - 0.5]))


@pytest.mark.parametrize(
    "entities,bonds,skipped",
    [
        ([Protein(sequence="AC", asym_ids=["A", "B"])], [], 0),
        # SG-SG', CB-SG', SG-CB', CA-SG', SG-CA', CB-CB'.
        (
            [Protein(sequence="AC", asym_ids=["A", "B"])],
            [
                Bond(
                    atom1=Atom(asym_id="A", res_id=2, res_name="CYS", name="SG"),
                    atom2=Atom(asym_id="B", res_id=2, res_name="CYS", name="SG"),
                )
            ],
            6,
        ),
        # Same-chain pairs are never scored, so an intra-chain disulfide skips nothing.
        (
            [Protein(sequence="CAC", asym_ids=["A", "B"])],
            [
                Bond(
                    atom1=Atom(asym_id="A", res_id=1, res_name="CYS", name="SG"),
                    atom2=Atom(asym_id="A", res_id=3, res_name="CYS", name="SG"),
                )
            ],
            0,
        ),
        # Ethanol C2-C1-O1 on Lys NZ: NZ-C2, CE-C2, NZ-C1, CD-C2, CE-C1, NZ-O1.
        (
            [Protein(sequence="K", asym_ids="A"), Ligand(smiles="CCO", asym_ids="L")],
            [
                Bond(
                    atom1=Atom(asym_id="A", res_id=1, res_name="LYS", name="NZ"),
                    atom2=Atom(asym_id="L", res_id=1, res_name="UNK", name="C2"),
                )
            ],
            6,
        ),
    ],
    ids=["no-bond", "interchain-disulfide", "intrachain-disulfide", "covalent-ligand"],
)
def test_interchain_scores_all_pairs_except_near_declared_bonds(
    monkeypatch, entities, bonds, skipped
):
    """Test that every atom pair between the two chains is scored except the pairs at most
    three bonds apart across a declared cross-chain bond."""
    system = System(entities=entities, bonds=bonds)
    potential = InterChainVDWOverlapPotential(parameters=_PARAMS[InterChainVDWOverlapPotential])
    potential.prepare(system)
    scored = []

    def record_pairs(coords, index, clash_distance):
        scored.append(index)
        return coords.new_zeros(len(coords))

    monkeypatch.setattr(potentials, "_steric_overlap_penalty", record_pairs)
    potential(torch.zeros(1, system.topology.n_atoms, 3), t=0.5)  # every pair within the cutoff
    first, second = (chain.n_atoms for chain in system.topology.chains)
    assert scored[0].shape[1] == first * second - skipped


@pytest.mark.parametrize("cls", list(_PARAMS))
@pytest.mark.parametrize("separation", [0.0, 50.0])
def test_scheduled_threshold_before_pruning(cls, separation):
    """Test that scheduled settings take effect whether atoms are overlapping or far apart."""
    parameter = "minimum_distance" if cls is PolymerLocalClashPotential else "buffer"
    values = [0.0, 2.0] if cls is PolymerLocalClashPotential else [0.8, 0.0]
    potential = cls(parameters={parameter: PiecewiseStepFunction([0.5], values)})
    potential.prepare(System(entities=[Protein(sequence="AA", asym_ids=["A", "B"])]))
    coords = torch.zeros(2, potential.feats["atom_pad_mask"].shape[1], 3)
    coords[:, :, 0] = torch.arange(coords.shape[1]) * separation
    potential.cutoff = 1.0
    assert torch.isfinite(potential(coords, t=0.0)).all()
    if cls is PolymerLocalClashPotential:
        assert potential(coords, t=0.0).sum() == 0
        assert bool(potential(coords, t=1.0).sum() > 0) == (separation == 0)
    else:
        with pytest.raises(ValueError, match="must exceed every clash distance"):
            potential(coords, t=1.0)


@pytest.mark.parametrize("cls", [InterChainVDWOverlapPotential, IntraChainVDWOverlapPotential])
def test_cutoff_rounding_without_cuda_sync(cls, gpu_device):
    """Test that a cutoff below the float32 limit raises an error without waiting for the GPU."""
    potential = cls(parameters={"buffer": 0.3})
    potential.prepare(System(entities=[Protein(sequence="G", asym_ids=["A", "B"])]))
    potential.cutoff = 2.3800001
    potential.feats = {k: v.to(gpu_device) for k, v in potential.feats.items()}
    coords = torch.zeros(1, potential.feats["atom_pad_mask"].shape[1], 3, device=gpu_device)
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        with pytest.raises(ValueError, match="must exceed every clash distance"):
            potential(coords, t=0.5)
    finally:
        torch.cuda.set_sync_debug_mode(previous)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_distance_precision(device, dtype):
    """Test that mixed-precision gradients and nearby-pair selection agree with float64 calculations."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    coords = torch.zeros(3, 2, 3, device=device, dtype=dtype)
    coords[:, 1, 0] = torch.tensor([0.0, 1e-7, 1e-6], device=device, dtype=dtype)
    coords.requires_grad_()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        energy = _steric_overlap_penalty(coords, torch.tensor([[0], [1]], device=device), 2.0)
    reference = coords.detach().double().requires_grad_()
    expected = 2.0 - ((reference[:, 1] - reference[:, 0]).square().sum(-1) + 1e-12).sqrt()
    torch.testing.assert_close(energy, expected.to(energy.dtype))
    torch.testing.assert_close(
        torch.autograd.grad(energy.sum(), coords)[0],
        torch.autograd.grad(expected.sum(), reference)[0].to(dtype),
    )
    coords = torch.zeros(1, 32, 3, device=device, dtype=dtype)
    coords[0, :, 0] = torch.arange(32, device=device) * 4
    coords += 1024 if dtype == torch.float16 else 65536
    reference = coords.double()
    distances = (reference[:, :, None] - reference[:, None, :]).norm(dim=-1)[0]
    expected = (distances < 5).triu(diagonal=1).nonzero().T
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        actual = compute_atom_pairs_within_cutoff(
            coords,
            torch.ones(32, device=device, dtype=torch.bool),
            torch.zeros(32, device=device, dtype=torch.long),
            torch.ones(1, 1, device=device, dtype=torch.bool),
            5.0,
            32,
        )
    torch.testing.assert_close(actual, expected)


def test_symmetric_chain_centroid_energy_and_gradient():
    """Test that chain-center energies and gradients match a direct calculation that ignores padding."""
    system = System(entities=[Protein(sequence="AG", asym_ids=["A", "B"])])
    structure = AllAtomStructure.from_system(system, mode="structure")
    potential = SymmetricChainCOMPotential(parameters={"buffer": 4.0})
    potential.prepare(system)
    coords = torch.randn(
        2, structure.num_nodes + 5, 3, generator=torch.Generator().manual_seed(37)
    ).requires_grad_()
    real = coords[:, : structure.num_nodes]
    centers = [real[:, structure.allatom_asym_id == c].mean(1) for c in (0, 1)]
    expected = (4.0 - (centers[0] - centers[1]).norm(dim=-1)).clamp_min(0)
    energy, gradient = _evaluated(potential, coords)
    torch.testing.assert_close(energy, expected)
    torch.testing.assert_close(gradient, torch.autograd.grad(expected.sum(), coords)[0])
