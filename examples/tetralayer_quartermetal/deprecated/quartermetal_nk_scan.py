#!/usr/bin/env python
"""nk-scan diagnostic: at a few representative (n, D) cells, run HF_PM and
HF_SVP at nk=36, 48, 96 and compare:

  - iteration count
  - converged energy
  - SV imbalance (HF_SVP)
  - per-cell wall time

Two solver back-ends are tried side-by-side:
  - ``solve_scf`` with DIIS (current pipeline default)
  - ``solve_direct_minimization`` (Riemannian CG; more robust to basin
    oscillations near band-edge crossings)

Goal: confirm the user's expectation that converged HF runs need ≤ 50
iters, and find which solver achieves that across nk values.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

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
from cpp_hf import (
    SCFConfig, solve_scf,
    SolverConfig, solve_direct_minimization,
)
import _quartermetal_common as qm

# Three "representative points" picked from the (n, D) phase landscape:
#   (n=0.30, D=0.30)  — low n, low D: phase boundary / PM region
#   (n=0.80, D=0.50)  — middle: heart of QM phase per the nk=12 scan
#   (n=1.30, D=0.80)  — high n, high D: deep QM
CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.80, 0.50, "QM-heart"),
    (1.30, 0.80, "deep-QM"),
]
NK_VALUES = (36, 48, 96)

# Strict convergence target: 1e-5 (reasonable for ΔE differences ~ 1 meV)
TOL_E = 1e-5
MAX_ITER = 25


def run_hf(kernel, seed_P, n_e, *, project_fn, solver: str, max_iter: int):
    if solver == "diis":
        cfg = SCFConfig(
            max_iter=max_iter, density_tol=TOL_E, comm_tol=TOL_E * 10,
            mixing=0.3, acceleration="diis", diis_size=8, diis_start=2,
            diis_damping=1.0, block_sizes=(8, 8, 8, 8),
            project_fn=project_fn,
        )
        t0 = time.perf_counter()
        res = solve_scf(kernel, seed_P, n_e, config=cfg)
        return (int(res.iterations), bool(res.converged),
                float(res.energy), np.asarray(res.density_matrix),
                time.perf_counter() - t0)
    elif solver == "dm":
        cfg = SolverConfig(
            max_iter=max_iter, tol_E=TOL_E, max_step=0.6,
            hartree_precondition=True, hartree_pc_scale=1.0,
            occupation_precondition=True,
            block_sizes=(8, 8, 8, 8),
            project_fn=project_fn,
        )
        t0 = time.perf_counter()
        res = solve_direct_minimization(kernel, seed_P, n_e, config=cfg)
        return (int(res.n_iter), bool(res.converged),
                float(res.energy), np.asarray(res.density),
                time.perf_counter() - t0)
    raise ValueError(solver)


def sv_imbalance(P, refP, w2d, n_orb_per_sv=8):
    diag_P = np.einsum("ij,ijaa->a", w2d, np.asarray(P)).real
    diag_R = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
    n_blocks = diag_P.shape[0] // n_orb_per_sv
    delta = np.array([
        (diag_P[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum()
         - diag_R[i*n_orb_per_sv:(i+1)*n_orb_per_sv].sum())
        for i in range(n_blocks)
    ])
    idx = int(np.argmax(np.abs(delta)))
    others = np.delete(delta, idx)
    return float(abs(delta[idx]) / max(np.mean(np.abs(others)), 1e-12))


def run_one_cell_at_nk(n_cm12: float, D_Vnm: float, nk: int):
    """Build the full setup at this (D, nk), run HF_PM and HF_SVP with
    both DIIS and DM solvers from cold seeds, return diagnostics."""
    print(f"\n=== nk={nk:3d}, (n={n_cm12:.2f}, D={D_Vnm:.2f}) ===")
    t0 = time.perf_counter()
    setup = qm.build_setup(D_Vnm=D_Vnm, nk=nk, init_scale=50.0,
                            small_orbital=False, bz_kind="rhombic")
    refP = qm.pm_c3_cn_reference_density(
        setup, T=qm.TEMPERATURE, max_iter=MAX_ITER, tol_E=TOL_E,
    )
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, qm.TEMPERATURE)
    print(f"  setup built in {time.perf_counter() - t0:.1f}s; ne={n_e:.6f}")

    kernel_hf = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=qm.TEMPERATURE,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )

    # Cold seeds: PM = h+0; SVP = h + sv-bias
    seed_pm = qm.initial_density_from_seed(h_run, setup.seeds["PM"],
                                             qm.TEMPERATURE)
    seed_svp = qm.initial_density_from_seed(h_run, setup.seeds["SVP"],
                                              qm.TEMPERATURE)

    rows = []
    branches = (
        ("PM_C3", "PM_C3", seed_pm),
        ("SVP_C3", "SVP_C3", seed_svp),
        ("SVP", "SVP", seed_svp),
    )
    for branch, projector_key, seed in branches:
        for solver in ("diis", "dm"):
            n_it, conv, E, P, dt = run_hf(
                kernel_hf, seed, n_e,
                project_fn=setup.project_fns[projector_key],
                solver=solver, max_iter=MAX_ITER,
            )
            imb = (sv_imbalance(P, refP, setup.weights)
                   if branch.startswith("SVP") else 1.0)
            print(f"  {branch:6s} {solver:4s}  it={n_it:3d} conv={int(conv)} "
                  f"E={E:+9.3f}  imb={imb:7.1f}  t={dt:5.1f}s")
            rows.append(dict(branch=branch, solver=solver, nk=nk,
                              iters=n_it, converged=conv, E=E,
                              imbalance=imb, t=dt))
    return rows


def main() -> int:
    all_rows = []
    for n, D, label in CELLS:
        print(f"\n############ {label}: (n={n}, D={D}) ############")
        for nk in NK_VALUES:
            rows = run_one_cell_at_nk(n, D, nk)
            for r in rows:
                r["label"] = label
                all_rows.append(r)

    # ---- summary ----
    print("\n\n=== SUMMARY: iter counts per (label × branch × solver × nk) ===")
    print(f"{'label':16s} {'branch':6s} {'solver':5s}", end="")
    for nk in NK_VALUES:
        print(f"  nk={nk:3d}", end="")
    print()
    for label in (l for _,_,l in CELLS):
        for branch in ("PM_C3", "SVP_C3", "SVP"):
            for solver in ("diis", "dm"):
                print(f"{label:16s} {branch:6s} {solver:5s}", end="")
                for nk in NK_VALUES:
                    matches = [r for r in all_rows
                                if r["label"] == label and r["branch"] == branch
                                and r["solver"] == solver and r["nk"] == nk]
                    if matches:
                        r = matches[0]
                        marker = " " if r["converged"] else "*"
                        print(f"  {r['iters']:3d}{marker} ", end="")
                    else:
                        print(f"   ?    ", end="")
                print()
    print("\n* = did not converge to tol")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
