import torch
from rdkit.Chem import MolFromSmiles

from bioemu2.structure.templates import _template_from_rdmol, load_polymer_templates


def test_single_heavy_atom_template_has_empty_bond_matrix() -> None:
    mol = MolFromSmiles("[NH2]")
    assert mol is not None

    template = _template_from_rdmol(mol)

    assert template.bond_index.tolist() == [[], []]
    assert template.bond_type.tolist() == []


def test_packaged_polymer_templates_are_centered() -> None:
    for name, template in load_polymer_templates().items():
        assert template.template_coords is not None
        torch.testing.assert_close(
            template.template_coords.mean(dim=0),
            torch.zeros(3, dtype=template.template_coords.dtype),
            atol=1e-6,
            rtol=0,
            msg=f"Packaged template {name} is not centered",
        )
