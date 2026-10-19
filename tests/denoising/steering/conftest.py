from __future__ import annotations

import pytest

from bioemu2.structure.inference_inputs import inference_input_structure_from_protein_sequence
from bioemu2.structure.templates import load_polymer_templates


@pytest.fixture(scope="module")
def test_structure():
    """A small protein AllAtomStructure for sampler parity tests."""
    return inference_input_structure_from_protein_sequence("GYDPETGTWG", load_polymer_templates())
