#!/usr/bin/env python
"""Cleaner convergence trace: use hist_E and hist_grad arrays from a single
SCF run.  These are recorded by cpp_hf at every iter and are the actual
trajectory of the optimizer — no spurious restart artifacts.

Compares SVP and SVP+C3 at one (n, D) cell.
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


N_CM12 = 0.80
D_VNM = 0.50
T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_eps20"
REF_KIND = "u0"
MAX_ITER = 25


CACHE = (REPO_ROOT / "examples" / "outputs" /
          f"quartermetal_convergence_trace2_nk{NK}_kmax{KMAX}_{HARTREE_KIND}_{REF_KIND}.npz")
PNG_OUT = (REPO_ROOT / "examples" / "outputs" /
           f"quartermetal_convergence_trace2_nk{NK}_kmax{KMAX}_{HARTREE_KIND}_{REF_KIND}.png")


def _load_cached_runs():
    if not CACHE.exists():
        return None
    data = np.load(CACHE, allow_pickle=True)
    return list(data["runs"])


def _save_runs(runs):
    np.savez(CACHE, runs=np.asarray(runs, dtype=object))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _load_cached_runs()
    if runs is not None:
        print(f"Loaded {len(runs)} cached runs from {CACHE}")
        return _plot(runs)

    setup = qm.build_setup(D_Vnm=D_VNM, nk=NK, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0,
                            hartree_kind=HARTREE_KIND)
    refP = qm.noninteracting_cn_reference(
        setup, T=T, ref_kind=REF_KIND, cn_projector="PM_C3",
    )
    n_e, h_run = qm.n_electrons_for_density(setup, N_CM12, T)
    kernel_HF = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    # Fock-only kernel (no Hartree) to test the user's hypothesis: is the
    # slow convergence caused by Hartree-induced charge sloshing?
    kernel_X = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=False, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed_pm = qm.initial_density_from_seed(h_run, setup.seeds["PM"], T)
    seed_svp = qm.initial_density_from_seed(h_run, setup.seeds["SVP"], T)

    layer = np.asarray(setup.model.layer)

    def make_capturing_proj(base_proj, capture_list):
        def proj(P):
            out = base_proj(P)
            # cpp_hf calls project_fn on density (before/after Fock).  Save a
            # copy each call so we can reconstruct Δ_tb per iter post-hoc.
            capture_list.append(np.array(out, copy=True))
            return out
        return proj

    runs = []
    for label, kernel_use, projector_key, seed in [
        ("PM",            kernel_HF, "PM",      seed_pm),
        ("PM+C3",         kernel_HF, "PM_C3",   seed_pm),
        ("SVP",           kernel_HF, "SVP",     seed_svp),
        ("SVP+C3",        kernel_HF, "SVP_C3",  seed_svp),
        ("X-only PM+C3",  kernel_X,  "PM_C3",   seed_pm),
        ("X-only SVP",    kernel_X,  "SVP",     seed_svp),
        ("X-only SVP+C3", kernel_X,  "SVP_C3",  seed_svp),
    ]:
        capture = []
        wrapped = make_capturing_proj(setup.project_fns[projector_key], capture)
        cfg = SolverConfig(
            max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
            block_sizes=(8, 8, 8, 8),
            project_fn=wrapped,
        )
        print(f"Running {label}...", flush=True)
        t0 = time.perf_counter()
        res = solve_direct_minimization(kernel_use, seed, n_e, config=cfg)
        E_hist = np.asarray(res.history["E"])
        G_hist = np.asarray(res.history["grad_norm"])
        # cpp_hf calls project_fn ~2× per iter (before Fock build, after eigh).
        # Sample one per iter for Δ_tb history.
        n_calls_per_iter = max(1, len(capture) // max(int(res.n_iter), 1))
        delta_tb = []
        h_diag = np.einsum("ijaa->a", np.asarray(h_run.hs)).real / (NK*NK)
        for i in range(0, len(capture), n_calls_per_iter):
            P_i = capture[i]
            diag_P = np.einsum("ij,ijaa->a", setup.weights, P_i).real
            diag_R = np.einsum("ij,ijaa->a", setup.weights, refP).real
            sigma = diag_P - diag_R
            V_H = setup.hartree_matrix @ sigma
            F_diag = h_diag + V_H
            layers_sorted = np.sort(np.unique(layer))
            bot = float(F_diag[layer == layers_sorted[0]].mean())
            top = float(F_diag[layer == layers_sorted[-1]].mean())
            delta_tb.append(top - bot)
        delta_tb = np.array(delta_tb)
        runs.append(dict(
            label=label, n_iter=int(res.n_iter),
            converged=bool(res.converged),
            E=float(res.energy), E_hist=E_hist, G_hist=G_hist,
            delta_tb=delta_tb,
            time=time.perf_counter() - t0,
        ))
        print(f"  it={int(res.n_iter)} conv={int(res.converged)} "
              f"E={float(res.energy):+.3f}  t={runs[-1]['time']:.1f}s")
    _save_runs(runs)
    return _plot(runs)


def _plot(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    # Plot 1: E(iter) for all 4 configs
    for r in runs:
        axes[0, 0].plot(np.arange(len(r["E_hist"])), r["E_hist"], "o-",
                          label=f"{r['label']} ({r['n_iter']} it)")
    axes[0, 0].set_xlabel("iter")
    axes[0, 0].set_ylabel(r"$E$ (meV)")
    axes[0, 0].set_title("Free-energy trajectory")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: |E_n - E_final| (semilog) — convergence rate
    for r in runs:
        E = r["E_hist"]
        if len(E) < 2: continue
        de = np.abs(E - E[-1]) + 1e-15
        axes[0, 1].semilogy(np.arange(len(de)), de, "o-",
                              label=r['label'])
    axes[0, 1].set_xlabel("iter")
    axes[0, 1].set_ylabel(r"$|E_n - E_{\rm final}|$ (meV)")
    axes[0, 1].set_title("Convergence rate (lower better)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: gradient norm
    for r in runs:
        G = r["G_hist"]
        axes[1, 0].semilogy(np.arange(len(G)), G + 1e-15, "o-",
                              label=r['label'])
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].set_ylabel(r"$\|\nabla E\|$")
    axes[1, 0].set_title("Gradient norm")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: ΔE per iter (zoomed)
    for r in runs:
        E = r["E_hist"]
        if len(E) < 2: continue
        dE = np.diff(E)
        axes[1, 1].plot(np.arange(1, len(E)), dE, "o-",
                          label=r['label'])
    axes[1, 1].set_xlabel("iter")
    axes[1, 1].set_ylabel(r"$E_{n+1} - E_n$ (meV)")
    axes[1, 1].set_title("Per-iter energy change (<=0 for descent)")
    axes[1, 1].axhline(0, color="black", lw=0.5)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Δ_tb history per iter (skip iter 0 — cold seed outlier)
    for r in runs:
        d = r["delta_tb"]
        if len(d) <= 1:
            continue
        axes[0, 2].plot(np.arange(1, len(d)), d[1:], "o-", label=r["label"])
    axes[0, 2].set_xlabel("iter")
    axes[0, 2].set_ylabel(r"$\Delta_{\rm tb}$ (meV)")
    axes[0, 2].set_title(r"Top - bottom layer potential (iter $\geq 1$)")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 6: zoomed Δ_tb (iter >= 3) — should show oscillation if any
    for r in runs:
        d = r["delta_tb"]
        if len(d) <= 3:
            continue
        axes[1, 2].plot(np.arange(3, len(d)), d[3:], "o-", label=r["label"])
    axes[1, 2].set_xlabel("iter")
    axes[1, 2].set_ylabel(r"$\Delta_{\rm tb}$ (meV)")
    axes[1, 2].set_title(r"$\Delta_{\rm tb}$ zoomed (iter $\geq 3$)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f"Convergence at (n={N_CM12}, D={D_VNM}), "
                  f"T={T} meV, nk={NK}, kmax={KMAX}")
    out = PNG_OUT
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
