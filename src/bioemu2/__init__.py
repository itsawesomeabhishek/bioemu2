"""Public BioEmu2 Python interface."""

from importlib.util import find_spec

# Define availability before importing the public API, which imports model modules.
HAS_TRITON: bool = find_spec("triton") is not None

from bioemu2.patches import mdtraj_patcher  # noqa: F401, I001

from bioemu2.conditioning.method import SamplingMode
from bioemu2.data_interface import (
    DNA,
    RNA,
    Atom,
    Bond,
    Branched,
    Contact,
    Ligand,
    ModifiedResidue,
    Protein,
    System,
    convert_boltz,
    load_system_and_contacts,
)
from bioemu2.denoising.samplers import EDMSampler, FKCSampler, GuidedFKSampler, Sampler
from bioemu2.biomoa.models.score_model import TrunkEmbeddings
from bioemu2.denoising.steering import Potential
from bioemu2.inference import EmuSession
from bioemu2.model import load_model
from bioemu2.msa.alignment import MSA, get_msa, load_a3m_msa
from bioemu2.paths import ASSETS_DIR
from bioemu2.physicality import PhysicalityFilterMode, PhysicalityReport
from bioemu2.propagation import (
    join_trajectory_steps,
    propagate,
    propagate_with_embeddings,
    simulate_trajectory,
    simulate_trajectory_with_embeddings,
)
from bioemu2.sampling import (
    check_physicality,
    predict_structure,
    sample_equilibrium,
    sample_with_embeddings,
)
from bioemu2.trajectory_io import map_cif_atom_order, save_frames

__all__ = [
    "ASSETS_DIR",
    "Atom",
    "Bond",
    "Branched",
    "Contact",
    "DNA",
    "EDMSampler",
    "EmuSession",
    "FKCSampler",
    "GuidedFKSampler",
    "Ligand",
    "MSA",
    "ModifiedResidue",
    "PhysicalityFilterMode",
    "PhysicalityReport",
    "Potential",
    "Protein",
    "RNA",
    "Sampler",
    "SamplingMode",
    "System",
    "TrunkEmbeddings",
    "convert_boltz",
    "check_physicality",
    "load_system_and_contacts",
    "get_msa",
    "join_trajectory_steps",
    "load_a3m_msa",
    "load_model",
    "predict_structure",
    "propagate",
    "propagate_with_embeddings",
    "sample_equilibrium",
    "sample_with_embeddings",
    "simulate_trajectory",
    "simulate_trajectory_with_embeddings",
    "save_frames",
    "map_cif_atom_order",
]
