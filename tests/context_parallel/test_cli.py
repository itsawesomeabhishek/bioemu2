import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import mdtraj
import numpy as np
import pytest
import torch

from bioemu2 import MSA, Protein, System, check_physicality, cli


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires four CUDA GPUs")
@pytest.mark.parametrize("physicality_mode", ["no_filter", "high_quality"])
@pytest.mark.parametrize(
    ("command", "sampler", "input_kind"),
    [
        ("sample-equilibrium", "edm", "yaml"),
        ("predict-structure", "edm", "pdb"),
        ("sample-equilibrium", "edm_fk", "yaml"),
    ],
)
def test_cp_cli_pipeline(
    tmp_path: Path,
    checkpoint: Path,
    command: str,
    sampler: str,
    input_kind: str,
    physicality_mode: str,
):
    system = System(entities=[Protein(sequence="GYDPETGTWG", asym_ids=["A"])])
    system_path = tmp_path / "system.yaml"
    system.to_yaml(system_path)
    out = tmp_path / "output"
    potentials = tmp_path / "potentials.yaml"
    potentials.write_text(
        "potentials:\n"
        "  - _target_: bioemu2.denoising.steering.potentials.BondLengthPotential\n"
        "    sigma_tolerance: 10.0\n"
        "    parameters: {resampling_weight: 0.01, guidance_weight: 0.01, guidance_interval: 1}\n"
    )
    inputs = ["--system", str(system_path)] if input_kind == "yaml" else ["--pdb-id", "TEST"]
    args = [
        command,
        *inputs,
        "--out",
        str(out),
        "--checkpoint-path",
        str(checkpoint),
        "--num-samples",
        "3",
        "--batch-size",
        "2",
        "--num-steps",
        "2",
        "--physicality-filter",
        physicality_mode,
        "--seed",
        "42",
        "--sampler",
        sampler,
    ]
    if sampler != "edm":
        args += ["--potentials", str(potentials)]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            "--module",
            "tests.context_parallel.test_cli",
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=360,
        env=os.environ | {"OMP_NUM_THREADS": "1", "EMU_CP_GRID": "2,2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {path.name for path in out.iterdir()} == {
        "samples.xtc",
        "physicality.json",
        "topology.cif",
        "system.yaml",
        "command.txt",
    }
    trajectory = mdtraj.load(str(out / "samples.xtc"), top=str(out / "topology.cif"))
    assert trajectory.n_frames == 3
    report = check_physicality(trajectory, mode=physicality_mode, system=system)
    assert json.loads((out / "physicality.json").read_text()) == [
        {
            "frame": frame,
            "status": (
                "not_checked"
                if physicality_mode == "no_filter"
                else "passed"
                if report.passed[frame]
                else "rejected"
            ),
            "reason": report.reasons[frame],
        }
        for frame in range(3)
    ]
    assert trajectory.n_atoms == system.topology.n_atoms
    assert np.isfinite(trajectory.xyz).all()
    assert not np.array_equal(trajectory.xyz[0], trajectory.xyz[1])
    expected_mode = "structure" if command == "predict-structure" else "equilibrium"
    assert f"mode: {expected_mode}" in (out / "system.yaml").read_text()
    assert f"cp.py {command}" in (out / "command.txt").read_text()


if __name__ == "__main__":
    # Replace network acquisition only; Click, preparation, model and collectives are real.
    with (
        patch.object(cli, "get_msa", return_value=MSA({"A": ">query\nGYDPETGTWG\n"})),
        patch.object(
            cli.System,
            "from_pdb_id",
            return_value=System(entities=[Protein(sequence="GYDPETGTWG", asym_ids=["A"])]),
        ),
    ):
        runpy.run_module("bioemu2.cp", run_name="__main__", alter_sys=True)
