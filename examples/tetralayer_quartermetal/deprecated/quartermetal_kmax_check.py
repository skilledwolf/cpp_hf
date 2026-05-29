#!/usr/bin/env python
"""Compare kmax=0.30 (spec) vs kmax=0.22 (reference's value) at T=0.3 meV,
nk=96 on the representative cells, to check if the smaller patch is
sufficient.

For each (cell, kmax), runs PM_C3 and SVP+C3 with DM and plots bands.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpp_hf
from cpp_hf import SolverConfig, solve_direct_minimization
import _common as qm


CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.80, 0.50, "QM-heart"),
    (1.30, 0.80, "deep-QM"),
]
KMAX_VALUES = (0.30, 0.22)
T = 0.3
NK = 96
MAX_ITER = 25


def run_one(setup, projector_key, seed_branch_label, T):
    refP = qm.pm_c3_cn_reference_density(
        setup, T=T, max_iter=MAX_ITER, tol_E=1e-4,
    )
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12_global, T)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed = qm.initial_density_from_seed(h_run, setup.seeds[seed_branch_label], T)
    cfg = SolverConfig(
        max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns[projector_key],
    )
    t0 = time.perf_counter()
    res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
    return res, time.perf_counter() - t0, refP


def horizontal_slice_bands(fock, kmax):
    nk1, nk2, nb, _ = fock.shape
    k_y_idx = nk2 // 2
    eps_row = np.empty((nk1, nb), dtype=np.float64)
    for ik1 in range(nk1):
        fk = fock[ik1, k_y_idx]
        eps_row[ik1] = np.linalg.eigvalsh(0.5 * (fk + fk.conj().T))
    return np.linspace(-kmax, kmax, nk1), eps_row


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


# Global so run_one can pull n
n_cm12_global = None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global n_cm12_global
    results = {}
    for n_cm12, D_Vnm, label in CELLS:
        n_cm12_global = n_cm12
        print(f"\n=== cell: (n={n_cm12}, D={D_Vnm}) [{label}] ===")
        for kmax in KMAX_VALUES:
            setup = qm.build_setup(D_Vnm=D_Vnm, nk=NK, kmax=kmax, T=T,
                                    bz_kind="rhombic", init_scale=50.0)
            for cfg_label, projector_key, seed_branch in [
                ("PM+C3",  "PM_C3",  "PM"),
                ("SVP+C3", "SVP_C3", "SVP"),
            ]:
                print(f"  kmax={kmax}, {cfg_label}: ", end="", flush=True)
                res, dt, refP = run_one(setup, projector_key, seed_branch, T)
                imb = sv_imbalance(res.density, refP, setup.weights)
                print(f"it={int(res.n_iter):3d} conv={int(res.converged)} "
                      f"E={float(res.energy):+9.3f} μ={float(res.mu):+7.2f} "
                      f"imb={imb:7.1f}  t={dt:.1f}s")
                kx, eps = horizontal_slice_bands(np.asarray(res.fock), kmax)
                results[(label, kmax, cfg_label)] = dict(
                    kx=kx, eps=eps, mu=float(res.mu), E=float(res.energy),
                    iters=int(res.n_iter), converged=bool(res.converged),
                    imbalance=imb,
                )

    # Plot: rows = (cell, kmax), cols = config
    fig, axes = plt.subplots(len(CELLS) * len(KMAX_VALUES), 2,
                              figsize=(8, 3.0 * len(CELLS) * len(KMAX_VALUES)),
                              sharey=False, constrained_layout=True)
    row = 0
    for n_cm12, D_Vnm, label in CELLS:
        for kmax in KMAX_VALUES:
            for col, cfg_label in enumerate(("PM+C3", "SVP+C3")):
                ax = axes[row, col]
                r = results[(label, kmax, cfg_label)]
                for b in range(r["eps"].shape[1]):
                    ax.plot(r["kx"], r["eps"][:, b], "-", lw=0.6,
                             color="black", alpha=0.5)
                ax.axhline(r["mu"], color="red", lw=1.0, linestyle="--")
                mark = "✓" if r["converged"] else "✗"
                ax.set_title(f"{label}, kmax={kmax}, {cfg_label}  it={r['iters']}{mark}\n"
                              f"E={r['E']:+.2f} μ={r['mu']:+.1f} imb={r['imbalance']:.0f}")
                ax.set_xlabel(r"$k_x$ (1/$a_G$)")
                ax.set_ylim(r["mu"] - 80, r["mu"] + 80)
                ax.grid(True, alpha=0.3)
            axes[row, 0].set_ylabel(r"$\varepsilon$ (meV)")
            row += 1

    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_kmax_check.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
