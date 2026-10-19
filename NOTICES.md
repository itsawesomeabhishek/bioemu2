# NOTICES

Except where otherwise noted, code authored for BioEmu2 by Microsoft is
licensed under the MIT License. This distribution also contains third-party
code under the licenses and notices described below and in the relevant source
files. See `LICENSE` and `licenses/LICENSE.BOLTZ-MIT`, `licenses/LICENSE.GEOSTD`,
`licenses/LICENSE.NVIDIA-FOLD-CP-MIT`, `licenses/LICENSE.PYTORCH3D-BSD`,
`licenses/LICENSE.TORCH-SCATTER-MIT`, `licenses/LICENSE.LGPL-3.0`, and
`licenses/LICENSE.GPL-3.0`.

## Stereochemistry reference data

`src/bioemu2/assets/stereo_chemistry.yaml` contains bond-length and
bond-angle reference data exported from
[OpenStructure 2.11.1](https://git.scicore.unibas.ch/schwede/openstructure/-/tree/2.11.1)
(commit `aeebc45c0fc2e3255cb45178013dad5a00999a39`). The `monlib` section
comes from OpenStructure's `modules/mol/alg/pymod/stereo_data.json`, derived
from CCP4 MON_LIB, which is licensed under the GNU Lesser General Public
License, version 3.0. The license is reproduced in `licenses/LICENSE.LGPL-3.0`,
together with the GNU GPL version 3 in `licenses/LICENSE.GPL-3.0`.

The `monlib_link` section comes from OpenStructure's
`modules/mol/alg/pymod/stereochemistry.py` (`GetDefaultStereoLinkData`).
Its nucleotide link parameters are adapted from geostd's
`rna_dna/chain_link_rna2p.cif`. A
[historical revision](https://github.com/phenix-project/geostd/blob/aef2b63e65ffab32157da4d6422723982a3e7d6d/rna_dna/chain_link_rna2p.cif)
contains the corresponding values; OpenStructure does not specify the exact
geostd revision it used. The geostd license is reproduced in `licenses/LICENSE.GEOSTD`.
Its peptide and disulfide link parameters come from CCP4's
`standard_geometry.cif`. OpenStructure identifies both link-data sources in
the source file above. No legacy Engh-Huber section is included in this asset.

The following geostd notice is reproduced from its `COPYRIGHT.txt`:

```text
geostd Copyright (c) 2009, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to
receipt of any required approvals from the U.S. Dept. of Energy).  All
rights reserved.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov referring to "geostd (LBNL Ref CR-2725)."

NOTICE.  This software was developed under funding from the
U.S. Department of Energy.  As such, the U.S. Government has been
granted for itself and others acting on its behalf a paid-up,
nonexclusive, irrevocable, worldwide license in the Software to
reproduce, prepare derivative works, and perform publicly and display
publicly.  Beginning five (5) years after the date permission to
assert copyright is obtained from the U.S. Department of Energy, and
subject to any subsequent five (5) year renewals, the U.S. Government
is granted for itself and others acting on its behalf a paid-up,
nonexclusive, irrevocable, worldwide license in the Software to
reproduce, prepare derivative works, distribute copies to the public,
perform publicly and display publicly, and to permit others to do so.
```

## Boltz-derived code

BioMoA layers and operations, denoising and steering code, MSA feature
preparation, data-interface code, and all-atom feature preparation under
`src/bioemu2` contain portions copied or adapted from
[Boltz v1.0.0](https://github.com/jwohlwend/boltz/tree/v1.0.0). Those portions
are licensed under the MIT License reproduced in `licenses/LICENSE.BOLTZ-MIT`. This
notice continues to apply where modules and symbols use BioEmu2-local names.

Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro.

Contact-conditioning code and feature conventions in
`src/bioemu2/biomoa/modules/conditioning.py` and
`src/bioemu2/conditioning/contact.py` are adapted from
[Boltz at commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`](https://github.com/jwohlwend/boltz/tree/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc).
Those portions are also licensed under the MIT License reproduced in
`licenses/LICENSE.BOLTZ-MIT`.

## PyTorch3D rotation helpers

The quaternion and rotation helpers in
`src/bioemu2/denoising/augmentation.py` contain adapted portions of
[PyTorch3D](https://github.com/facebookresearch/pytorch3d) under the BSD
License reproduced in `licenses/LICENSE.PYTORCH3D-BSD`.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

## torch_scatter-derived code

`src/bioemu2/structure/scatter.py` contains portions copied or adapted from
[`torch_scatter`](https://github.com/rusty1s/pytorch_scatter) at commit
`8ec9364b0bdcd99149952a25749ad211c2d0567b`, via the Microsoft-authored
`mdmlff.src.mlff.utils.scatter` wrapper. The upstream MIT License is reproduced
in `licenses/LICENSE.TORCH-SCATTER-MIT`.

Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>.

## NVIDIA Fold-CP-derived code

Context-parallel distributed primitives and operations under
`src/bioemu2/context_parallel` contain portions copied, adapted, or ported from
[NVIDIA Fold-CP](https://github.com/NVIDIA-BioNeMo/boltz-cp) at commit
`f76f37a77c854b56b6250b426af8c2d63b501f7f`. Source files identify their
upstream paths and describe material modifications. The upstream MIT License is
reproduced in `licenses/LICENSE.NVIDIA-FOLD-CP-MIT`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Fold-CP also identifies portions derived from
[Boltz](https://github.com/jwohlwend/boltz) at commit
`cb04aeccdd480fd4db707f0bbafde538397fa2ac`; the Boltz notice and license are
provided above and in `licenses/LICENSE.BOLTZ-MIT`.

## MDTraj patches

`src/bioemu2/patches/mdtraj_patcher.py` contains modified portions of
[MDTraj](https://github.com/mdtraj/mdtraj), specifically the
`Topology.from_openmm` and `PDBxTrajectoryFile` implementations. The surrounding
patching and integration code was authored for BioEmu2.

The file is licensed under the GNU Lesser General Public License, version 3.0
or any later version. The copied MDTraj portions were originally offered under
the GNU Lesser General Public License, version 2.1 or any later version; this
distribution exercises the option to use version 3.0 or any later version.

The upstream MDTraj files also contain portions originating from OpenMM under
the MIT License. Their complete copyright, permission, and warranty notices are
retained in `src/bioemu2/patches/mdtraj_patcher.py`.

## OpenMM patches

`src/bioemu2/patches/openmm_patches.py` contains modified portions of
[OpenMM](https://github.com/openmm/openmm) and is licensed under OpenMM's MIT
license. The complete OpenMM copyright and permission notice is reproduced in
that file.

Portions copyright (c) 2015-2023 Stanford University and the Authors.
