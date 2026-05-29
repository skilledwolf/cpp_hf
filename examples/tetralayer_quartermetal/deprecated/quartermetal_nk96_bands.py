#!/usr/bin/env python
"""nk=96 iter-count check + band-structure plots for the representative
cells used in the prior nk-scan.

Runs HF (combined Hartree+exchange) at nk ∈ {24, 36, 48, 96} on:
  - (n=0.30, D=0.30, "phase-boundary") — SVP-projected
  - (n=0.80, D=0.50, "QM-heart") — SVP-projected
and reports iter count + per-iter time per nk.

After each converged solve, computes band structures along a horizontal
slice through the K-point at k_y = 0, plots all bands ε_n(k_x) using a
shared y-axis range across the (cell × nk) panel grid.

Skips exchange-only at nk=96 (it consistently took 130-200 iters and
isn't part of the user-relevant comparison).
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
from cpp_hf import SCFConfig, solve_scf
import _common as qm


CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.80, 0.50, "QM-heart"),
]
NK_VALUES = (24, 36, 48, 96)
TOL_E = 1e-5
MAX_ITER = 25


def run_hf_and_get_fock(setup, kernel, seed_P, n_e, branch):
    cfg = SCFConfig(
        max_iter=MAX_ITER, density_tol=TOL_E, comm_tol=TOL_E * 10,
        mixing=0.3, acceleration="diis", diis_size=8, diis_start=2,
        diis_damping=1.0, block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns[branch],
    )
    t0 = time.perf_counter()
    res = solve_scf(kernel, seed_P, n_e, config=cfg)
    return res, time.perf_counter() - t0


def horizontal_slice_bands(fock: np.ndarray, kmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues along the row k_y = 0 (closest mesh row to k_y=0).

    Returns (k_x_values, eps_array) with shape (nk1, nb) for the eps.
    """
    nk1, nk2, nb, _ = fock.shape
    # Mesh is symmetric around 0 in both k_x and k_y per discretize convention.
    # Pick the row closest to k_y = 0.
    k_y_idx = nk2 // 2
    eps_row = np.empty((nk1, nb), dtype=np.float64)
    for ik1 in range(nk1):
        fk = fock[ik1, k_y_idx]
        eps_row[ik1] = np.linalg.eigvalsh(0.5 * (fk + fk.conj().T))
    # k_x values: linspace from -kmax to +kmax (symmetric)
    k_x = np.linspace(-kmax, kmax, nk1)
    return k_x, eps_row


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Iter-count + band-structure scan @ nk in", NK_VALUES)
    results = {}  # (cell_label, nk) -> dict

    for n_cm12, D_Vnm, label in CELLS:
        print(f"\n========= cell: (n={n_cm12}, D={D_Vnm}) [{label}] =========")
        for nk in NK_VALUES:
            print(f"\n  nk={nk}:")
            t_total = time.perf_counter()
            setup = qm.build_setup(D_Vnm=D_Vnm, nk=nk, init_scale=50.0,
                                    small_orbital=False)
            refP = qm.noninteracting_cn_reference(setup)
            n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, qm.TEMPERATURE)
            h_arr = np.asarray(h_run.hs)
            seed_svp = qm.initial_density_from_seed(
                h_run, setup.seeds["SVP"], qm.TEMPERATURE)

            kernel_hf = cpp_hf.HartreeFockKernel(
                weights=setup.weights, hamiltonian=h_arr,
                coulomb_q=setup.Vq, T=qm.TEMPERATURE,
                include_hartree=True, include_exchange=True,
                reference_density=refP, hartree_matrix=setup.hartree_matrix,
            )
            res_svp, dt = run_hf_and_get_fock(setup, kernel_hf, seed_svp,
                                                n_e, "SVP")
            print(f"    HF+SVP it={res_svp.iterations:3d} "
                  f"conv={int(res_svp.converged)} "
                  f"E={float(res_svp.energy):+.3f}  t={dt:.1f}s")

            fock = np.asarray(res_svp.fock_matrix)
            k_x, eps_row = horizontal_slice_bands(fock, kmax=qm.KMAX)
            results[(label, nk)] = dict(
                k_x=k_x, eps=eps_row,
                E=float(res_svp.energy),
                mu=float(res_svp.chemical_potential),
                iters=int(res_svp.iterations),
                converged=bool(res_svp.converged),
            )

    # ---- Plot ----
    fig, axes = plt.subplots(len(CELLS), len(NK_VALUES),
                              figsize=(4*len(NK_VALUES), 3.2*len(CELLS)),
                              sharey="row", constrained_layout=True)
    if len(CELLS) == 1:
        axes = axes[None, :]
    for i, (n, D, label) in enumerate(CELLS):
        # y-range across nk for this cell
        all_eps = np.concatenate([results[(label, nk)]["eps"].ravel()
                                    for nk in NK_VALUES])
        mu_avg = np.mean([results[(label, nk)]["mu"] for nk in NK_VALUES])
        # Window ±100 meV around mu
        ymin, ymax = mu_avg - 100.0, mu_avg + 100.0

        for j, nk in enumerate(NK_VALUES):
            ax = axes[i, j]
            r = results[(label, nk)]
            kx = r["k_x"]
            for b in range(r["eps"].shape[1]):
                ax.plot(kx, r["eps"][:, b], "-", lw=0.6, color="black",
                         alpha=0.5)
            ax.axhline(r["mu"], color="red", lw=1.0, linestyle="--",
                        label=f"μ={r['mu']:+.1f} meV")
            mark = "✓" if r["converged"] else "✗"
            ax.set_title(f"{label}, nk={nk}  it={r['iters']}{mark}\n"
                          f"E={r['E']:+.2f} meV")
            ax.set_xlabel(r"$k_x$ (1/$a_G$)")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=7)
        axes[i, 0].set_ylabel(r"$\varepsilon$ (meV)")

    fig.suptitle("HF+SVP band structure along $k_y=0$ slice — convergence with nk")
    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_nk96_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
