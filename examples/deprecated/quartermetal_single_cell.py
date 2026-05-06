#!/usr/bin/env python
"""Single-(n, D)-cell driver for tetralayer ABC quartermetal study.

Runs all three configs at one (n, D) and prints diagnostics:

  * H_PM: Hartree-only, PM-projected.  Uses ``solve_hartree_newton``.
  * HF_SVP: full HF, SVP projection (no PM).  Uses ``solve_direct_minimization``
    with SVP-biased seed.
  * HF_PM: full HF, PM-projected (control — energy upper bound for SVP).

The HF_SVP energy at doped (n > 0.3, D < 0.5) cells should be ≤ HF_PM
energy.  If it isn't, the SVP solver hasn't actually broken the symmetry.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
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
    HartreeNewtonConfig, solve_hartree_newton,
    SCFConfig, solve_scf,
    SolverConfig, solve_direct_minimization,
)

import _quartermetal_common as qm


def run_H_PM(setup: qm.QMSetup, n_cm12: float, *, T: float,
              verbose: bool = True) -> dict:
    """Hartree-only PM-projected SCF on the doped tetralayer."""
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)

    # PM-projected refP at this filling (zero-bias CN reference).
    refP = qm.bootstrap_cn_reference(setup, T=T)

    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights,
        hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq,
        T=float(T),
        include_hartree=True,
        include_exchange=False,
        reference_density=refP,
        hartree_matrix=setup.hartree_matrix,
    )
    cfg = HartreeNewtonConfig(
        max_iter=25, tol_E=1e-4, tol_sigma=1e-4,
        backtrack_max=6, backtrack_shrink=0.5,
        project_fn=setup.project_fns["PM"],
    )
    t0 = time.perf_counter()
    res = solve_hartree_newton(kernel, refP, n_e, config=cfg)
    elapsed = time.perf_counter() - t0

    layer = np.asarray(setup.model.layer)
    delta = qm.delta_tb_from_fock(np.asarray(res.fock_matrix), layer)
    if verbose:
        print(f"  [H_PM ] n={n_cm12:.3f}  E={float(res.energy):+.4f}  "
              f"μ={float(res.chemical_potential):+.3f}  Δ_tb={delta:+8.3f}  "
              f"it={res.iterations:2d} conv={int(res.converged)}  t={elapsed:.1f}s")
    return dict(label="H_PM", n_cm12=n_cm12, n_e=n_e,
                energy=float(res.energy), mu=float(res.chemical_potential),
                delta_tb=delta, iters=int(res.iterations),
                converged=bool(res.converged), elapsed=elapsed,
                density=np.asarray(res.density_matrix),
                fock=np.asarray(res.fock_matrix))


def run_HF(setup: qm.QMSetup, n_cm12: float, branch: str, *,
            T: float, refP: np.ndarray | None = None,
            seed_P: np.ndarray | None = None,
            verbose: bool = True) -> dict:
    """Full HF (Hartree + exchange) with the given projector branch."""
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)

    if refP is None:
        refP = qm.bootstrap_cn_reference(setup, T=T)
    if seed_P is None:
        seed_P = qm.initial_density_from_seed(h_run, setup.seeds[branch], T)

    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights,
        hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq,
        T=float(T),
        include_hartree=True,
        include_exchange=True,
        reference_density=refP,
        hartree_matrix=setup.hartree_matrix,
    )
    # DIIS-SCF for HF: handles the divergent Hartree well; DM with the
    # Hartree preconditioner can struggle on the same problems.
    cfg = SCFConfig(
        max_iter=25, density_tol=1e-4, comm_tol=1e-3, mixing=0.3,
        acceleration="diis", diis_size=8, diis_start=2, diis_damping=1.0,
        trust_radius=0.0,
        project_fn=setup.project_fns[branch],
    )
    t0 = time.perf_counter()
    res = solve_scf(kernel, seed_P, n_e, config=cfg)
    elapsed = time.perf_counter() - t0

    layer = np.asarray(setup.model.layer)
    delta = qm.delta_tb_from_fock(np.asarray(res.fock_matrix), layer)
    if verbose:
        label = f"HF_{branch}"
        print(f"  [{label:6s}] n={n_cm12:.3f}  E={float(res.energy):+.4f}  "
              f"μ={float(res.chemical_potential):+.3f}  Δ_tb={delta:+8.3f}  "
              f"it={int(res.iterations):3d} conv={int(res.converged)}  t={elapsed:.1f}s")
    return dict(label=f"HF_{branch}", n_cm12=n_cm12, n_e=n_e,
                energy=float(res.energy), mu=float(res.chemical_potential),
                delta_tb=delta, iters=int(res.iterations),
                converged=bool(res.converged), elapsed=elapsed,
                density=np.asarray(res.density_matrix),
                fock=np.asarray(res.fock_matrix))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=float, default=0.5,
                   help="Carrier density in 10^12 cm^-2 (default 0.5).")
    p.add_argument("--D", type=float, default=0.30,
                   help="Displacement field in V/nm (default 0.30).")
    p.add_argument("--nk", type=int, default=24,
                   help="k-mesh density (default 24 for fast iteration).")
    p.add_argument("--init-scale", type=float, default=30.0,
                   help="Strength of the SVP-biased seed in meV.")
    p.add_argument("--hartree-kind",
                   choices=("ungated", "dualgate"),
                   default="ungated",
                   help="Hartree q=0 kernel: ungated layer Coulomb (default) "
                        "or dualgate (literal spec wording).")
    p.add_argument("--skip-svp", action="store_true",
                   help="Skip the HF_SVP run (debugging).")
    args = p.parse_args(argv)

    print(f"=== tetralayer ABC, (n={args.n} ×10^12 cm^-2, D={args.D} V/nm), "
          f"nk={args.nk}, T={qm.TEMPERATURE} meV, hartree={args.hartree_kind} ===")
    print()

    t_setup = time.perf_counter()
    setup = qm.build_setup(D_Vnm=args.D, nk=args.nk,
                            init_scale=args.init_scale,
                            hartree_kind=args.hartree_kind)
    print(f"setup built in {time.perf_counter() - t_setup:.1f}s; "
          f"nb={setup.h_template.hs.shape[-1]}, "
          f"‖HH‖₂={np.linalg.norm(setup.hartree_matrix, 2):.2e}, "
          f"ne_cn={setup.ne_cn:.4f}")
    print()

    # Spec-defined refP: non-interacting CN at U=0, half-filling.
    print("Building non-interacting CN reference (U=0)...")
    t0 = time.perf_counter()
    refP = qm.noninteracting_cn_reference(setup)
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    print()

    rows = []

    # Config 1: Hartree-only, PM-projected.  DIIS-SCF outperforms
    # solve_hartree_newton when HH has nontrivial dipole-mode spread —
    # Newton oscillates because Π·HH ≈ 1 there and the linearized step
    # overshoots; DIIS extrapolates over those oscillations cleanly.
    print("Running H_PM (Hartree-only DIIS-SCF)...")
    n_e, h_run = qm.n_electrons_for_density(setup, args.n, qm.TEMPERATURE)
    kernel_h = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=float(qm.TEMPERATURE),
        include_hartree=True, include_exchange=False,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    cfg_h = SCFConfig(
        max_iter=25, density_tol=1e-4, comm_tol=1e-3, mixing=0.3,
        acceleration="diis", diis_size=8, diis_start=2, diis_damping=1.0,
    )
    t0 = time.perf_counter()
    res_h = solve_scf(kernel_h, refP, n_e, config=cfg_h)
    elapsed = time.perf_counter() - t0
    layer = np.asarray(setup.model.layer)
    delta_h = qm.delta_tb_from_fock(np.asarray(res_h.fock_matrix), layer)
    rows.append(dict(label="H_PM", energy=float(res_h.energy),
                      mu=float(res_h.chemical_potential),
                      delta_tb=delta_h, iters=int(res_h.iterations),
                      converged=bool(res_h.converged), elapsed=elapsed))
    print(f"  E={rows[-1]['energy']:+.4f}  μ={rows[-1]['mu']:+.3f}  "
          f"Δ_tb={delta_h:+8.3f}  it={rows[-1]['iters']:3d}  "
          f"conv={int(rows[-1]['converged'])}  t={elapsed:.1f}s")
    print()

    # Config 2: HF, PM-projected (control)
    print("Running HF_PM (HF DM, PM seed)...")
    res_pm = run_HF(setup, args.n, "PM", T=qm.TEMPERATURE, refP=refP)
    rows.append(res_pm)
    print()

    if not args.skip_svp:
        # Config 3: HF, SVP-projected (broken-symmetry)
        print("Running HF_SVP (HF DM, SVP-biased seed + SVP projector)...")
        res_svp = run_HF(setup, args.n, "SVP", T=qm.TEMPERATURE, refP=refP)
        rows.append(res_svp)
        print()
        e_pm  = res_pm["energy"]
        e_svp = res_svp["energy"]
        diff = e_svp - e_pm
        verdict = ("✓ SVP energy LOWER than PM (broken-symmetry phase found)"
                   if diff < -1e-4 else
                   "≈ SVP energy ≈ PM (no symmetry breaking at this cell)"
                   if abs(diff) < 1e-4 else
                   "✗ SVP energy HIGHER than PM (SVP solver did NOT break symmetry)")
        print(f"  E_SVP - E_PM = {diff:+.4f}  →  {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
