#!/usr/bin/env python
"""Diagnostic: does the iter-count blow-up with nk come from Hartree or
exchange (Fock)?

Runs the same (n, D) cell with three kernel variants:
  H   — include_hartree=True,  include_exchange=False
  X   — include_hartree=False, include_exchange=True
  HF  — both

at nk ∈ {24, 36, 48} on the SVP-projected branch.  Reports iter count
and convergence for each.

The cell (n=0.30, D=0.30) is the "phase boundary" point that needed 25
iters at nk=36 but 52 iters at nk=48 in the prior nk-scan.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parent
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpp_hf
from cpp_hf import SCFConfig, solve_scf
import _quartermetal_common as qm


CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.80, 0.50, "QM-heart"),
]
NK_VALUES = (24, 36, 48)
TOL_E = 1e-5
MAX_ITER = 25


def run_one(kernel, seed_P, n_e, project_fn, max_iter=MAX_ITER):
    cfg = SCFConfig(
        max_iter=max_iter, density_tol=TOL_E, comm_tol=TOL_E * 10,
        mixing=0.3, acceleration="diis", diis_size=8, diis_start=2,
        diis_damping=1.0, block_sizes=(8, 8, 8, 8),
        project_fn=project_fn,
    )
    t0 = time.perf_counter()
    res = solve_scf(kernel, seed_P, n_e, config=cfg)
    return (int(res.iterations), bool(res.converged),
            float(res.energy), time.perf_counter() - t0)


def run_at_nk(n_cm12, D_Vnm, nk, label):
    setup = qm.build_setup(D_Vnm=D_Vnm, nk=nk, init_scale=50.0,
                            small_orbital=False)
    refP = qm.noninteracting_cn_reference(setup)
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, qm.TEMPERATURE)
    h_arr = np.asarray(h_run.hs)
    seed_svp = qm.initial_density_from_seed(h_run, setup.seeds["SVP"],
                                              qm.TEMPERATURE)

    print(f"\n  nk={nk:3d}, (n={n_cm12:.2f}, D={D_Vnm:.2f}) [{label}]:")

    # Note: cpp_hf requires at least one of Hartree or Exchange enabled.
    for kernel_label, hartree, exchange in [
        ("H ", True,  False),
        ("X ", False, True),
        ("HF", True,  True),
    ]:
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights, hamiltonian=h_arr,
            coulomb_q=setup.Vq, T=qm.TEMPERATURE,
            include_hartree=hartree, include_exchange=exchange,
            reference_density=refP,
            hartree_matrix=setup.hartree_matrix,
        )
        n_it, conv, E, dt = run_one(
            kernel, seed_svp, n_e, setup.project_fns["SVP"],
        )
        marker = " " if conv else "*"
        print(f"    {kernel_label}  it={n_it:3d}{marker}  E={E:+9.3f}  t={dt:6.1f}s")


def main():
    for n, D, label in CELLS:
        print(f"\n========= cell: (n={n}, D={D}) [{label}] =========")
        for nk in NK_VALUES:
            run_at_nk(n, D, nk, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
