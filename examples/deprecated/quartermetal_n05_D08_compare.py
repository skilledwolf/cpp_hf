"""Single-cell focused: (n=0.5, D=0.8) with both cn_sc and sameU references,
to see if reference choice changes the QM phase verdict here.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
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
from cpp_hf import SolverConfig, solve_direct_minimization
import _quartermetal_common as qm

N_CM12 = 0.50
D_VNM = 0.80
T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_eps3"
MAX_ITER = 25


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
    print(f"Cell: n={N_CM12}, D={D_VNM}, T={T}, nk={NK}, kmax={KMAX}, "
          f"HH={HARTREE_KIND}")
    setup = qm.build_setup(D_Vnm=D_VNM, nk=NK, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0,
                            hartree_kind=HARTREE_KIND)
    n_e, h_run = qm.n_electrons_for_density(setup, N_CM12, T)

    for ref_kind in ("u0", "sameU", "cn_sc"):
        print(f"\n--- ref_kind={ref_kind} ---")
        refP = qm.noninteracting_cn_reference(
            setup, T=T, ref_kind=ref_kind,
            cn_projector="PM_C3" if ref_kind == "cn_sc" else "PM",
        )
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
            coulomb_q=setup.Vq, T=T, include_hartree=True, include_exchange=True,
            reference_density=refP, hartree_matrix=setup.hartree_matrix,
        )
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
            t0 = time.perf_counter()
            res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
            imb = imbalance(res.density, refP, setup.weights)
            print(f"  {cfg_label:8s}  it={int(res.n_iter):3d} conv={int(res.converged)} "
                  f"E={float(res.energy):+9.3f}  μ={float(res.mu):+7.2f}  "
                  f"imb={imb:7.0f}  t={time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
