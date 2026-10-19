# SPDX-License-Identifier: LGPL-3.0-or-later
# This file incorporates modified portions of MDTraj's Topology.from_openmm
# and PDBxTrajectoryFile implementations.
#
# Copyright 2012-2014 Stanford University and the Authors
# Copyright 2012-2022 Stanford University and the Authors
# Authors: Peter Eastman, Robert McGibbon
# Contributors: Kyle A. Beauchamp, Matthew Harrigan, Carlos Xavier Hernandez,
# Jason Swails
# Copyright (c) Microsoft Corporation.
# Modified by Microsoft Corporation for BioEmu2, September 29, 2026.
#
# The MDTraj portions were originally distributed under the GNU Lesser
# General Public License, version 2.1 or (at the recipient's option) any later
# version. This modified file is distributed under version 3.0 or any later
# version. See licenses/LICENSE.LGPL-3.0 and licenses/LICENSE.GPL-3.0.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# Portions of the copied MDTraj implementations originate from the OpenMM
# molecular simulation toolkit:
# - Topology.from_openmm portions copyright (c) 2012 Stanford University and
#   Peter Eastman.
# - PDBxTrajectoryFile portions copyright 2008-2012 Stanford University and
#   Peter Eastman.
#
# Those portions are distributed under the following terms:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
# OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Patches to MDTraj topology conversion, PDBx/mmCIF I/O, and contacts."""

from types import MethodType

import mdtraj
import numpy as np
import openmm.app as app
from mdtraj.core.selection import _chain
from mdtraj.core.topology import Amide, Aromatic, Double, Single, Topology, Triple, elem
from mdtraj.formats import pdbx
from mdtraj.formats.registry import FormatRegistry
from mdtraj.utils import open_maybe_zipped
from mdtraj.utils.unitcell import (
    box_vectors_to_lengths_and_angles,
    lengths_and_angles_to_box_vectors,
)
from openmm.unit import nanometers

from bioemu2.patches.openmm_patches import PDBxFile

""" several patches to MDtraj:
- Patches mdtraj's selection mechanism to include `chainname`, and the reader to get the
  chain ID from the openMM object that is used by mdtraj under the hood.
- Patches mdtraj's core residue name dictionary to include names of protonated residues

MDtraj is patched for all emu project code. To use it outside of the project, use
```python
from bioemu2.patches.mdtraj_patcher import mdtraj
traj = mdtraj.load('/tmp/3ife.cif')
traj.top.select('chainname B')
>>> array([3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292,
       3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303,
       3304])
```
"""

# patch selection routines to cover chain_id
mdtraj.core.selection.SelectionKeyword.keyword_aliases.update(
    {"chainname": _chain("residue", "chain", "chain_id")}
)
mdtraj.core.selection.parse_selection = (
    mdtraj.core.selection.parse_selection.__class__()
)  # weird overwrite of class by instance

# Mapping from protonated amino acid residue names (e.g. "ASH") to standard forms ("ASP").
PROTONATED_AA_SUBSTITUTIONS: dict[str, str] = {
    "ASH": "ASP",
    "CYX": "CYS",
    "CYM": "CYS",
    "GLH": "GLU",
    "HID": "HIS",
    "HIE": "HIS",
    "HIP": "HIS",
    "HIN": "HIS",
    "LYN": "LYS",
}

# some protonated amino acid residue names are missing in the mdtraj lookup dictionary
# symptom: residues do not show up in sequence and backbone selection fails
_MISSING_PROTONATED_AA_CODES: dict[str, str] = {
    key: mdtraj.core.residue_names._AMINO_ACID_CODES[val]
    for key, val in PROTONATED_AA_SUBSTITUTIONS.items()
}

mdtraj.core.residue_names._AMINO_ACID_CODES.update(_MISSING_PROTONATED_AA_CODES)
_AMINO_ACID_CODES = mdtraj.core.residue_names._AMINO_ACID_CODES

# update downstream definition of protein residues
_PROTEIN_RESIDUES = frozenset(_AMINO_ACID_CODES.keys())
mdtraj.core.residue_names._PROTEIN_RESIDUES = _PROTEIN_RESIDUES

# topology imported in top-level __init__; climb down the ladder
mdtraj.core.topology._PROTEIN_RESIDUES = _PROTEIN_RESIDUES
mdtraj.core.topology._AMINO_ACID_CODES = _AMINO_ACID_CODES


# patch reading function from open MM to pass chain_id attribute to mdtraj
def from_openmm(cls, value: app.Topology):
    """Create a mdtraj topology from an OpenMM topology

    PATCH: also copy over original chain IDs

    Parameters
    ----------
    value : openmm.app.Topology
        An OpenMM topology that you wish to convert to a
        mdtraj topology.
    """

    bond_mapping = {
        app.Single: Single,
        app.Double: Double,
        app.Triple: Triple,
        app.Amide: Amide,
        app.Aromatic: Aromatic,
        None: None,
    }

    out = cls()
    atom_mapping = {}

    for chain in value.chains():
        c = out.add_chain(chain_id=chain.id)  # ONLY CHANGE: new kwarg
        for residue in chain.residues():
            try:
                r = out.add_residue(residue.name, c, resSeq=int(residue.id))
            except ValueError:
                r = out.add_residue(residue.name, c)
            for atom in residue.atoms():
                if atom.element is None:
                    element = elem.virtual
                else:
                    element = elem.get_by_symbol(atom.element.symbol)
                a = out.add_atom(atom.name, element, r)
                atom_mapping[atom] = a

    for bond in value.bonds():
        a1, a2 = bond
        out.add_bond(
            atom_mapping[a1],
            atom_mapping[a2],
            type=bond_mapping[bond.type],
            order=bond.order,
        )

    return out


# Monkey-patches mdtraj PDBxTrajectoryFile to use modified OpenMM's PDBxFile that
# always reads the PDB-assigned chain identifier and ignores the author one
# in core.utils.openmm_patcher
@FormatRegistry.register_fileobject(".pdbx")
@FormatRegistry.register_fileobject(".cif")
class PDBxTrajectoryFile:
    """Interface for reading and writing PDBx/mmCIF files

    Parameters
    ----------
    filename : path-like
        The filename to open. A path to a file on disk.
    mode : {'r', 'w'}
        The mode in which to open the file, either 'r' for read or 'w' for write.
    force_overwrite : bool
        If opened in write mode, and a file by the name of `filename` already
        exists on disk, should we overwrite it?
    top : mdtraj.core.Topology, default=None
        if you give a topology as input the topology won't be parsed from the file

    Attributes
    ----------
    positions : np.ndarray, shape=(n_frames, n_atoms, 3)
    topology : mdtraj.Topology
    closed : bool

    See Also
    --------
    mdtraj.load_pdbx : High-level wrapper that returns a ``md.Trajectory``
    """

    distance_unit = "nanometers"

    def __init__(self, filename, mode="r", force_overwrite=True, top=None):
        self._open = False
        self._mode = mode

        if mode == "r":
            self._open = True
            pdbx = PDBxFile(filename)
            if top is None:
                self._topology = Topology.from_openmm(pdbx.topology)
            else:
                self._topology = top
            positions = [
                pdbx.getPositions(asNumpy=True, frame=i).value_in_unit(nanometers)
                for i in range(pdbx.getNumFrames())
            ]
            self._positions = np.array(positions)
            vectors = pdbx.topology.getPeriodicBoxVectors()
            if vectors is not None:
                vectors = [np.array(v.value_in_unit(nanometers)) for v in vectors]
                l1, l2, l3, alpha, beta, gamma = box_vectors_to_lengths_and_angles(*vectors)
                self._unitcell_lengths = (l1, l2, l3)
                self._unitcell_angles = (alpha, beta, gamma)
            else:
                self._unitcell_lengths = None
                self._unitcell_angles = None
        elif mode == "w":
            self._open = True
            self._next_model = 0
            self._file = open_maybe_zipped(filename, "w", force_overwrite)
        else:
            raise ValueError(f"invalid mode: {mode}")

    def write(
        self,
        positions,
        topology,
        unitcell_lengths=None,
        unitcell_angles=None,
        keepIds=False,
    ):
        """Write one frame of a molecular dynamics trajectory to disk in PDBx/mmCIF format.

        Parameters
        ----------
        positions : array_like
            The list of atomic positions to write.
        topology : mdtraj.Topology
            The Topology defining the model to write.
        unitcell_lengths : {tuple, None}
            Lengths of the three unit cell vectors, or None for a non-periodic system
        unitcell_angles : {tuple, None}
            Angles between the three unit cell vectors, or None for a non-periodic system
        keepIds : bool, default=False
            Whether to preserve the chain and residue IDs from the topology.
        """
        if not self._mode == "w":
            raise ValueError("file not opened for writing")

        if self._next_model == 0:
            self._openmm_topology = topology.to_openmm()
            if unitcell_lengths is None:
                self._openmm_topology.setPeriodicBoxVectors(None)
            else:
                vectors = lengths_and_angles_to_box_vectors(
                    *unitcell_lengths[0], *unitcell_angles[0]
                )
                self._openmm_topology.setPeriodicBoxVectors(vectors * nanometers)
            PDBxFile.writeHeader(self._openmm_topology, self._file, keepIds=keepIds)
            self._next_model = 1
        if len(positions.shape) == 3:
            positions = positions[0]
        PDBxFile.writeModel(
            self._openmm_topology,
            positions * nanometers,
            self._file,
            self._next_model,
            keepIds=keepIds,
        )
        self._next_model += 1

    @property
    def positions(self):
        """The cartesian coordinates of all of the atoms in each frame. Available when a file is opened in mode='r'"""
        return self._positions

    @property
    def topology(self):
        """The topology from this PDBx/mmCIF file. Available when a file is opened in mode='r'"""
        return self._topology

    @property
    def unitcell_lengths(self):
        """The unitcell lengths (3-tuple) in this PDBx/mmCIF file. May be None"""
        return self._unitcell_lengths

    @property
    def unitcell_angles(self):
        """The unitcell angles (3-tuple) in this PDBx/mmCIF file. May be None"""
        return self._unitcell_angles

    @property
    def closed(self):
        """Whether the file is closed"""
        return not self._open

    def close(self):
        """Close the PDBx/mmCIF file"""
        if self._mode == "w" and self._open:
            self._file.close()
        self._open = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def __len__(self):
        "Number of frames in the file"
        if str(self._mode) != "r":
            raise NotImplementedError('len() only available in mode="r" currently')
        return len(self._positions)


mdtraj.core.topology.Topology.from_openmm = MethodType(from_openmm, mdtraj.core.topology.Topology)
pdbx.PDBxTrajectoryFile = PDBxTrajectoryFile


# Patch compute_contacts: numpy ≥ 2.0 makes `array != "all"` raise ValueError.
# Fix by converting ndarray contacts to a list before forwarding to the original.
_original_compute_contacts = mdtraj.geometry.contact.compute_contacts


def _patched_compute_contacts(
    traj,
    **kwargs,
):  # noqa: E501
    """This patch converts `contacts` to a list if it's an np.ndarray before forwarding to the
    original compute_contacts. This is to avoid a ValueError with numpy 2.0+ when `contacts` is an
    array and compared to a string "all".

    Caveat: now when all contact pairs are filtered out, it will raise obscure error at
    compute_distances step "ValueError: atom_pairs must be ndim 2. You supplied 1" due to another
    code issue in mdtraj. We choose to leave that out for now since there's no simple wrapper fix.
    """
    contacts = kwargs.get("contacts", None)
    if isinstance(contacts, np.ndarray):
        kwargs["contacts"] = contacts.tolist()
    return _original_compute_contacts(
        traj,
        **kwargs,
    )


_patched_compute_contacts.__doc__ = _original_compute_contacts.__doc__
mdtraj.compute_contacts = _patched_compute_contacts
mdtraj.geometry.contact.compute_contacts = _patched_compute_contacts

mdtraj.IS_PATCHED = True
