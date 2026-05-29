"""Scan a few (n, D) cells at ε_eff=3 with cn_sc reference, looking for
where SVP energy drops below PM (indicating QM phase).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpp_hf
from cpp_hf import SolverConfig, solve_direct_minimization
import _common as qm

T = 0.3
NK = 64           # smaller for scan speed
KMAX = 0.30
HARTREE_KIND = "dualgate_eps10"
REF_KIND = "PM_C3_CN"
MAX_ITER = 25

CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.50, 0.50, ""),
    (0.80, 0.30, ""),
    (0.80, 0.50, "QM-heart"),
    (0.80, 0.80, ""),
    (1.00, 0.50, ""),
    (1.00, 0.80, ""),
    (1.30, 0.50, ""),
    (1.30, 0.80, "deep-QM"),
    (1.30, 1.00, ""),
]


def imbalance(P, refP, w2d, n_orb=8):
    diag_P = np.einsum("ij,ijaa->a", w2d, np.asarray(P)).real
    diag_R = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
    n_blocks = diag_P.shape[0] // n_orb
    delta = np.array([
        (diag_P[i*n_orb:(i+1)*n_orb].sum() - diag_R[i*n_orb:(i+1)*n_orb].sum())
        for i in range(n_blocks)
    ])
    idx = int(np.argmax(np.abs(delta)))
    others = np.delete(delta, idx)
    return float(abs(delta[idx]) / max(np.mean(np.abs(others)), 1e-12))


def main():
    print(f"Scan with PM_C3 CN ref, T={T}, nk={NK}, kmax={KMAX}")
    print(f"{'n':>5s} {'D':>5s}  label              "
          f"{'E_PM':>9s} {'E_SVP':>9s} {'E_PM+C3':>9s} {'E_SVP+C3':>9s}  "
          f"{'svp-pmc3':>9s} {'imb_SVP':>8s} {'imb_SVPC3':>10s}")

    for n_cm12, D_Vnm, label in CELLS:
        setup = qm.build_setup(D_Vnm=D_Vnm, nk=NK, kmax=KMAX, T=T,
                                bz_kind="rhombic", init_scale=50.0,
                                hartree_kind=HARTREE_KIND)
        refP = qm.pm_c3_cn_reference_density(
            setup, T=T, max_iter=MAX_ITER, tol_E=1e-4,
        )
        n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
            coulomb_q=setup.Vq, T=T, include_hartree=True, include_exchange=True,
            reference_density=refP, hartree_matrix=setup.hartree_matrix,
        )
        results = {}
        for cfg_label, projector_key, seed_branch in [
            ("PM",      "PM",     "PM"),
            ("PM+C3",   "PM_C3",  "PM"),
            ("SVP",     "SVP",    "SVP"),
            ("SVP+C3",  "SVP_C3", "SVP"),
        ]:
            seed = qm.initial_density_from_seed(
                h_run, setup.seeds[seed_branch], T,
            )
            cfg = SolverConfig(
                max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
                block_sizes=(8, 8, 8, 8),
                project_fn=setup.project_fns[projector_key],
            )
            res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
            results[cfg_label] = (
                float(res.energy),
                imbalance(res.density, refP, setup.weights),
                int(res.n_iter), bool(res.converged),
            )

        E_pm   = results["PM"][0]
        E_svp  = results["SVP"][0]
        E_pmc3 = results["PM+C3"][0]
        E_svpc3 = results["SVP+C3"][0]
        imb_svp = results["SVP"][1]
        imb_svpc3 = results["SVP+C3"][1]
        print(f"{n_cm12:>5.2f} {D_Vnm:>5.2f}  {label:18s}"
              f"{E_pm:>9.2f} {E_svp:>9.2f} {E_pmc3:>9.2f} {E_svpc3:>9.2f}  "
              f"{E_svp - E_pmc3:>+9.2f} {imb_svp:>8.0f} {imb_svpc3:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
