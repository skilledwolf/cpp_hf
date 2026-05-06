#!/usr/bin/env python
"""All 4 HF configs (PM, PM+C3, SVP, SVP+C3) at representative (n, D) cells,
with band-structure plots from converged Fock at each config × cell.

Uses DM solver, rhombic BZ, nk=48, kmax=0.30 (production settings).
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
from cpp_hf import SolverConfig, solve_direct_minimization
import _quartermetal_common as qm


CELLS = [
    (0.30, 0.30, "phase-boundary"),
    (0.80, 0.50, "QM-heart"),
    (1.30, 0.80, "deep-QM"),
]
NK = 48
KMAX = 0.30
TOL_E = 1e-4
MAX_ITER = 25


def horizontal_slice_bands(fock: np.ndarray, kmax: float):
    nk1, nk2, nb, _ = fock.shape
    k_y_idx = nk2 // 2
    eps_row = np.empty((nk1, nb), dtype=np.float64)
    for ik1 in range(nk1):
        fk = fock[ik1, k_y_idx]
        eps_row[ik1] = np.linalg.eigvalsh(0.5 * (fk + fk.conj().T))
    k_x = np.linspace(-kmax, kmax, nk1)
    return k_x, eps_row


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


def run_config(setup, refP, h_run, n_e, branch_seed: str, projector_key: str):
    seed = qm.initial_density_from_seed(h_run, setup.seeds[branch_seed],
                                          qm.TEMPERATURE)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=qm.TEMPERATURE,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    cfg = SolverConfig(
        max_iter=MAX_ITER, tol_E=TOL_E, max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns[projector_key],
    )
    t0 = time.perf_counter()
    res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
    return res, time.perf_counter() - t0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for n_cm12, D_Vnm, label in CELLS:
        print(f"\n=== cell: (n={n_cm12}, D={D_Vnm}) [{label}] ===")
        setup = qm.build_setup(D_Vnm=D_Vnm, nk=NK, kmax=KMAX,
                                bz_kind="rhombic", init_scale=50.0)
        refP = qm.noninteracting_cn_reference(setup)
        n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, qm.TEMPERATURE)

        for cfg_label, branch_seed, projector_key in [
            ("PM",     "PM",  "PM"),
            ("PM+C3",  "PM",  "PM_C3"),
            ("SVP",    "SVP", "SVP"),
            ("SVP+C3", "SVP", "SVP_C3"),
        ]:
            res, dt = run_config(setup, refP, h_run, n_e,
                                  branch_seed, projector_key)
            imb = sv_imbalance(res.density, refP, setup.weights)
            print(f"  {cfg_label:8s}  it={int(res.n_iter):3d} "
                  f"conv={int(res.converged)}  E={float(res.energy):+9.3f}  "
                  f"μ={float(res.mu):+7.2f}  imb={imb:7.1f}  t={dt:5.1f}s")
            fock = np.asarray(res.fock)
            k_x, eps_row = horizontal_slice_bands(fock, KMAX)
            results[(label, cfg_label)] = dict(
                k_x=k_x, eps=eps_row,
                E=float(res.energy), mu=float(res.mu),
                iters=int(res.n_iter), converged=bool(res.converged),
                imbalance=imb,
            )

    # --- Plot: rows = (n, D) cells, cols = configs ---
    cfgs = ("PM", "PM+C3", "SVP", "SVP+C3")
    fig, axes = plt.subplots(len(CELLS), len(cfgs),
                              figsize=(4*len(cfgs), 3.2*len(CELLS)),
                              sharey="row", constrained_layout=True)
    if len(CELLS) == 1:
        axes = axes[None, :]
    for i, (n_cm12, D_Vnm, label) in enumerate(CELLS):
        mu_avg = np.mean([results[(label, c)]["mu"] for c in cfgs])
        ymin, ymax = mu_avg - 100.0, mu_avg + 100.0
        for j, cfg_label in enumerate(cfgs):
            ax = axes[i, j]
            r = results[(label, cfg_label)]
            for b in range(r["eps"].shape[1]):
                ax.plot(r["k_x"], r["eps"][:, b], "-", lw=0.6,
                         color="black", alpha=0.5)
            ax.axhline(r["mu"], color="red", lw=1.0, linestyle="--")
            mark = "✓" if r["converged"] else "✗"
            ax.set_title(f"{label}  {cfg_label}  it={r['iters']}{mark}\n"
                          f"E={r['E']:+.2f}  imb={r['imbalance']:.0f}")
            ax.set_xlabel(r"$k_x$ (1/$a_G$)")
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.3)
        axes[i, 0].set_ylabel(r"$\varepsilon$ (meV)")

    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_4configs_bands.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
