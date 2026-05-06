#!/usr/bin/env python
"""Trace the convergence of a single (n, D) cell that took many iterations.

Picks the QM-heart cell with SVP+C3 at the production settings (T=0.3,
nk=96, kmax=0.30) — took ~42 iters in the prior scan.  Reports:
  - free-energy and gradient-norm history per iter (from cpp_hf hist_E,
    hist_grad)
  - SV-population imbalance per iter (requires running iter-by-iter)
  - Δ_tb per iter
to see if the SCF oscillates between basins, decays smoothly to a
fixed point, or stalls in a flat region.
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


# pick a representative ~40-iter cell:  QM-heart, SVP+C3, T=0.3, nk=96, kmax=0.3
N_CM12 = 0.80
D_VNM = 0.50
T = 0.3
NK = 96
KMAX = 0.30


def _imbalance(P, refP, w2d, n_orb=8):
    diag_P = np.einsum("ij,ijaa->a", w2d, np.asarray(P)).real
    diag_R = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
    n_blocks = diag_P.shape[0] // n_orb
    delta = np.array([
        (diag_P[i*n_orb:(i+1)*n_orb].sum()
         - diag_R[i*n_orb:(i+1)*n_orb].sum())
        for i in range(n_blocks)
    ])
    idx = int(np.argmax(np.abs(delta)))
    others = np.delete(delta, idx)
    return float(abs(delta[idx]) / max(np.mean(np.abs(others)), 1e-12)), delta


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = qm.build_setup(D_Vnm=D_VNM, nk=NK, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0)
    refP = qm.noninteracting_cn_reference(setup, T=T)
    n_e, h_run = qm.n_electrons_for_density(setup, N_CM12, T)
    layer = np.asarray(setup.model.layer)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed = qm.initial_density_from_seed(h_run, setup.seeds["SVP"], T)

    # Run by capping max_iter at increasing values, capturing P at each step.
    # cpp_hf's hist_E and hist_grad are returned; we'll restart from each P.
    print("Running SVP+C3 with iter-by-iter trace (max_iter=N for N=1..50)...")
    iters_to_trace = list(range(1, 51))
    history = []
    cur_seed = seed
    last_iter = 0
    for max_it in iters_to_trace:
        cfg = SolverConfig(
            max_iter=max_it - last_iter, tol_E=1e-12, max_step=0.6,
            block_sizes=(8, 8, 8, 8),
            project_fn=setup.project_fns["SVP_C3"],
        )
        # Tighter tol_E so the SCF doesn't early-exit
        res = solve_direct_minimization(kernel, cur_seed, n_e, config=cfg)
        cur_seed = np.asarray(res.density)
        # Compute imbalance
        imb, delta_n = _imbalance(cur_seed, refP, setup.weights)
        # Δ_tb
        diag_F = np.einsum("ijaa->a", np.asarray(res.fock)).real / (NK*NK)
        layers_sorted = np.sort(np.unique(layer))
        bot = float(diag_F[layer == layers_sorted[0]].mean())
        top = float(diag_F[layer == layers_sorted[-1]].mean())
        history.append(dict(
            iter=int(res.n_iter) + last_iter,
            E=float(res.energy),
            mu=float(res.mu),
            imb=imb,
            delta_n=delta_n,
            delta_tb=top - bot,
            grad_norm_last=float(np.asarray(res.hist_grad)[-1])
                            if hasattr(res, 'hist_grad') and len(np.asarray(res.hist_grad)) > 0
                            else float('nan'),
        ))
        last_iter = int(res.n_iter) + last_iter
        if res.converged:
            print(f"  Converged at iter {last_iter}")
            break
        if last_iter >= 50:
            break

    iters = np.array([h["iter"] for h in history])
    E = np.array([h["E"] for h in history])
    mu = np.array([h["mu"] for h in history])
    imb = np.array([h["imb"] for h in history])
    delta_tb = np.array([h["delta_tb"] for h in history])
    delta_n_history = np.array([h["delta_n"] for h in history])
    grad_norm = np.array([h["grad_norm_last"] for h in history])

    print("\nSummary:")
    print(f"  iter range: {iters.min()}..{iters.max()}")
    print(f"  E range: [{E.min():.4f}, {E.max():.4f}]")
    print(f"  E final: {E[-1]:.4f}, ΔE final = {E[-1] - E[-2]:.4f}")
    print(f"  imb range: [{imb.min():.1f}, {imb.max():.1f}]")
    print(f"  imb final: {imb[-1]:.1f}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].plot(iters, E, "o-", color="tab:blue")
    axes[0, 0].set_xlabel("iter")
    axes[0, 0].set_ylabel(r"$E$ (meV)")
    axes[0, 0].set_title(f"Free energy ({len(iters)} iters)")
    axes[0, 0].grid(True, alpha=0.3)

    # E - E_final on log scale to see decay rate
    e_final = E[-1]
    de = E - e_final + 1e-12
    axes[0, 1].semilogy(iters, np.abs(de), "o-", color="tab:blue")
    axes[0, 1].set_xlabel("iter")
    axes[0, 1].set_ylabel(r"$|E_n - E_{\rm final}|$ (meV)")
    axes[0, 1].set_title("E decay rate")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iters, imb, "o-", color="tab:orange")
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].set_ylabel(r"SV imbalance")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Spin-valley population imbalance")
    axes[1, 0].grid(True, alpha=0.3)

    # Per-sector Δn evolution
    for sector in range(delta_n_history.shape[1]):
        axes[1, 1].plot(iters, delta_n_history[:, sector], "o-",
                          label=f"sector {sector}")
    axes[1, 1].set_xlabel("iter")
    axes[1, 1].set_ylabel(r"$\Delta n_{\rm sector}$")
    axes[1, 1].set_title("Per-sector population deviation from CN")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(iters, mu, "o-", color="tab:green")
    axes[2, 0].set_xlabel("iter")
    axes[2, 0].set_ylabel(r"$\mu$ (meV)")
    axes[2, 0].set_title("Chemical potential")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(iters, delta_tb, "o-", color="tab:red")
    axes[2, 1].set_xlabel("iter")
    axes[2, 1].set_ylabel(r"$\Delta_{\rm tb}$ (meV)")
    axes[2, 1].set_title("Top - bottom layer potential")
    axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle(f"SVP+C3 convergence trace at (n={N_CM12}, D={D_VNM}), "
                  f"T={T} meV, nk={NK}, kmax={KMAX}")
    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_convergence_trace.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
