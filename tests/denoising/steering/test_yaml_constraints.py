"""Manually-configured `ContactPotential` on the Crambin fixture.

The three known disulfide bridges (see fixtures/crambin.yaml) are enforced
by patching contact feats onto a `build_potential_features` dict without
reading contact configuration from the System. Asserts a positive violation
energy when the flagged SG atoms are moved apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bioemu2.data_interface import System
from bioemu2.denoising.steering.feats import build_potential_features
from bioemu2.denoising.steering.potentials import ContactPotential
from bioemu2.structure.batch import batch_collate
from bioemu2.structure.inference_inputs import inference_input_structure_from_protein_sequence
from bioemu2.structure.templates import load_polymer_templates

# Disulfide SG atom indices in bioemu's canonical atom ordering for the
# 46-residue Crambin sequence — mirrors the comment in
# fixtures/crambin.yaml.  A regression here means the ordering shifted.
_DISULFIDE_PAIRS = [(19, 281), (25, 228), (115, 187)]
_THRESHOLD = 2.5

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "crambin.yaml"


@pytest.fixture(scope="module")
def crambin_batch():
    task = System.from_yaml(_FIXTURE)
    seq = task.entities[0].sequence
    s = inference_input_structure_from_protein_sequence(seq, load_polymer_templates())
    return batch_collate([s])


def test_contact_potential_positive_on_violation(crambin_batch):
    # Manually inject the SG-SG pairs and threshold that a caller would put
    # on their steering YAML; move the atoms apart and check that
    # ContactPotential fires a positive violation.
    feats = build_potential_features(crambin_batch)
    device = crambin_batch.coords.device
    feats["contact_pair_index"] = torch.tensor(
        [[i for i, _ in _DISULFIDE_PAIRS], [j for _, j in _DISULFIDE_PAIRS]],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)  # [1, 2, 3]
    feats["contact_thresholds"] = torch.full(
        (1, len(_DISULFIDE_PAIRS)), _THRESHOLD, dtype=torch.float32, device=device
    )

    coords = crambin_batch.coords.clone()
    for i, j in _DISULFIDE_PAIRS:
        coords[0, i, 0] = -20.0
        coords[0, j, 0] = +20.0

    potential = ContactPotential()
    potential.feats = feats
    energy = potential(coords, t=0.0)
    assert energy.shape == (1,)
    assert energy.item() > 0.0, f"expected positive violation energy, got {energy.item()}"
