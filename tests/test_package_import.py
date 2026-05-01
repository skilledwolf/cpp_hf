from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import cpp_hf


def test_root_package_exports_new_api():
    assert hasattr(cpp_hf, "HartreeFockKernel")
    assert hasattr(cpp_hf, "SolverConfig")
    assert hasattr(cpp_hf, "SolveResult")
    assert hasattr(cpp_hf, "solve")
    assert hasattr(cpp_hf, "solve_direct_minimization")
    assert hasattr(cpp_hf, "SCFConfig")
    assert hasattr(cpp_hf, "SCFResult")
    assert hasattr(cpp_hf, "solve_scf")
    assert hasattr(cpp_hf, "build_fock")
    assert hasattr(cpp_hf, "hf_energy")
    assert hasattr(cpp_hf, "free_energy")
    assert hasattr(cpp_hf, "ContinuationResult")
    assert hasattr(cpp_hf, "solve_continuation")
    assert hasattr(cpp_hf, "resample_kgrid")


def test_removed_v1_api_is_absent():
    """The pre-rewrite cpp_hf exposed an SCF-only kernel under different names."""
    assert not hasattr(cpp_hf, "hartreefock_iteration_cpp"), (
        "Old SCF-only API removed; use solve_scf"
    )
    assert not hasattr(cpp_hf, "hartreefock_iteration_cpp_ex"), (
        "Old extended SCF API removed; use solve_scf"
    )
    assert not hasattr(cpp_hf, "PRECOND_EIGH"), (
        "Old preconditioner constants removed; SolverConfig replaces them"
    )


def test_import_does_not_allocate_arrays_at_module_import():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    pythonpath_parts = [str(root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    code = """
import numpy as np

def _fail(*args, **kwargs):
    raise AssertionError("array creation was called during import")

np.zeros = _fail
np.asarray = _fail
np.array = _fail
import cpp_hf
assert hasattr(cpp_hf, "__all__")
"""

    subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
