import subprocess
import sys


def test_import_bioemu2_patches_mdtraj():
    code = """
import mdtraj

assert not hasattr(mdtraj, "IS_PATCHED")

import bioemu2

assert mdtraj.IS_PATCHED
"""
    subprocess.run([sys.executable, "-c", code], check=True)
