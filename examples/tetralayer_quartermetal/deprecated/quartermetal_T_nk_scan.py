#!/usr/bin/env python
"""Test convergence sensitivity to temperature T and k-mesh density nk
on the cells where PM bands looked suspicious.

For each (T, nk) combination, run PM_C3 (DM solver, rhombic BZ) and plot
the converged bands.  If T=1 meV is too coarse to resolve narrow flat-band
features at the Fermi level, lower T should change the converged state
and the band shape near μ.
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


# QM-heart and deep-QM where PM looked suspicious
CELLS = [
    (0.80, 0.50, "QM-heart"),
    (1.30, 0.80, "deep-QM"),
]
# (T_meV, nk) pairs.  Cover both axes.
GRID = [
    (1.0, 48),  # spec defaults
    (0.3, 48),  # lower T at spec nk
    (1.0, 96),  # higher nk at spec T
    (0.3, 96),  # both
]
KMAX = 0.30
MAX_ITER = 25


def run_cell(n_cm12, D_Vnm, T, nk):
    setup = qm.build_setup(D_Vnm=D_Vnm, nk=nk, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0)
    refP = qm.pm_c3_cn_reference_density(
        setup, T=T, max_iter=MAX_ITER, tol_E=1e-4,
    )
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed_pm = qm.initial_density_from_seed(h_run, setup.seeds["PM"], T)
    cfg = SolverConfig(
        max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns["PM_C3"],
    )
    t0 = time.perf_counter()
    res = solve_direct_minimization(kernel, seed_pm, n_e, config=cfg)
    return res, setup, time.perf_counter() - t0


def horizontal_slice_bands(fock):
    nk1, nk2, nb, _ = fock.shape
    k_y_idx = nk2 // 2
    eps_row = np.empty((nk1, nb), dtype=np.float64)
    for ik1 in range(nk1):
        fk = fock[ik1, k_y_idx]
        eps_row[ik1] = np.linalg.eigvalsh(0.5 * (fk + fk.conj().T))
    return np.linspace(-KMAX, KMAX, nk1), eps_row


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for n_cm12, D_Vnm, label in CELLS:
        print(f"\n=== cell: (n={n_cm12}, D={D_Vnm}) [{label}] ===")
        for T_meV, nk in GRID:
            print(f"  T={T_meV} meV, nk={nk}: ", end="", flush=True)
            res, setup, dt = run_cell(n_cm12, D_Vnm, T_meV, nk)
            print(f"it={int(res.n_iter):3d} conv={int(res.converged)} "
                  f"E={float(res.energy):+9.3f} μ={float(res.mu):+7.2f}  "
                  f"t={dt:.1f}s")
            kx, eps = horizontal_slice_bands(np.asarray(res.fock))
            results[(label, T_meV, nk)] = dict(
                kx=kx, eps=eps, mu=float(res.mu),
                E=float(res.energy), iters=int(res.n_iter),
                converged=bool(res.converged),
            )

    fig, axes = plt.subplots(len(CELLS), len(GRID),
                              figsize=(4*len(GRID), 3.2*len(CELLS)),
                              sharey="row", constrained_layout=True)
    if len(CELLS) == 1:
        axes = axes[None, :]
    for i, (n_cm12, D_Vnm, label) in enumerate(CELLS):
        mu_avg = np.mean([results[(label, T, nk)]["mu"] for T, nk in GRID])
        ymin, ymax = mu_avg - 80.0, mu_avg + 80.0
        for j, (T_meV, nk) in enumerate(GRID):
            ax = axes[i, j]
            r = results[(label, T_meV, nk)]
            for b in range(r["eps"].shape[1]):
                ax.plot(r["kx"], r["eps"][:, b], "-", lw=0.6,
                         color="black", alpha=0.5)
            ax.axhline(r["mu"], color="red", lw=1.0, linestyle="--")
            mark = "✓" if r["converged"] else "✗"
            ax.set_title(f"{label}  T={T_meV}meV nk={nk}  it={r['iters']}{mark}\n"
                          f"E={r['E']:+.2f}")
            ax.set_xlabel(r"$k_x$ (1/$a_G$)")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
        axes[i, 0].set_ylabel(r"$\varepsilon$ (meV)")

    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_T_nk_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
