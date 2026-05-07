"""Scan D at fixed n for convergence trace2-style outputs."""
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

N_CM12 = 0.80
D_VALUES = (0.03, 0.10, 0.50, 1.00)
T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_eps2"
REF_KIND = "PM_C3_CN"
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


def run_one(D_Vnm):
    setup = qm.build_setup(D_Vnm=D_Vnm, nk=NK, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0,
                            hartree_kind=HARTREE_KIND)
    refP = qm.pm_c3_cn_reference_density(
        setup, T=T, max_iter=200, tol_E=1e-4,
    )
    n_e, h_run = qm.n_electrons_for_density(setup, N_CM12, T)
    layer = np.asarray(setup.model.layer)
    kernel_HF = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T, include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    kernel_X = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T, include_hartree=False, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed_pm = qm.initial_density_from_seed(h_run, setup.seeds["PM"], T)
    seed_svp = qm.initial_density_from_seed(h_run, setup.seeds["SVP"], T)

    runs = []
    for label, kernel_use, projector_key, seed in [
        ("PM",          kernel_HF, "PM",      seed_pm),
        ("PM+C3",       kernel_HF, "PM_C3",   seed_pm),
        ("SVP",         kernel_HF, "SVP",     seed_svp),
        ("SVP+C3",      kernel_HF, "SVP_C3",  seed_svp),
        ("X-only PM+C3", kernel_X, "PM_C3",   seed_pm),
        ("X-only SVP",   kernel_X, "SVP",     seed_svp),
        ("X-only SVP+C3", kernel_X, "SVP_C3", seed_svp),
    ]:
        cfg = SolverConfig(max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
                            block_sizes=(8, 8, 8, 8),
                            project_fn=setup.project_fns[projector_key])
        t0 = time.perf_counter()
        res = solve_direct_minimization(kernel_use, seed, n_e, config=cfg)
        E_hist = np.asarray(res.history["E"])
        G_hist = np.asarray(res.history["grad_norm"])
        imb = imbalance(res.density, refP, setup.weights)
        runs.append(dict(
            label=label, n_iter=int(res.n_iter), converged=bool(res.converged),
            E=float(res.energy), mu=float(res.mu), imbalance=imb,
            E_hist=E_hist, G_hist=G_hist,
        ))
        print(f"  {label:14s}  it={int(res.n_iter):3d} conv={int(res.converged)} "
              f"E={float(res.energy):+9.3f}  imb={imb:7.0f}  t={time.perf_counter()-t0:.1f}s")
    return runs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_runs = {}
    for D in D_VALUES:
        print(f"\n=== D = {D} V/nm ===")
        all_runs[D] = run_one(D)

    fig, axes = plt.subplots(len(D_VALUES), 4,
                              figsize=(20, 4*len(D_VALUES)),
                              constrained_layout=True)
    for i, D in enumerate(D_VALUES):
        runs = all_runs[D]
        for r in runs:
            axes[i, 0].plot(np.arange(len(r["E_hist"])), r["E_hist"], "o-",
                              label=f"{r['label']} (imb={r['imbalance']:.0f})")
        axes[i, 0].set_xlabel("iter")
        axes[i, 0].set_ylabel(r"$E$ (meV)")
        axes[i, 0].set_title(f"D={D} V/nm: free energy")
        axes[i, 0].legend(fontsize=7)
        axes[i, 0].grid(True, alpha=0.3)

        for r in runs:
            E = r["E_hist"]
            if len(E) < 2: continue
            de = np.abs(E - E[-1]) + 1e-15
            axes[i, 1].semilogy(np.arange(len(de)), de, "o-", label=r["label"])
        axes[i, 1].set_xlabel("iter")
        axes[i, 1].set_ylabel(r"$|E_n - E_{\rm final}|$")
        axes[i, 1].set_title(f"D={D}: convergence")
        axes[i, 1].legend(fontsize=7)
        axes[i, 1].grid(True, alpha=0.3)

        for r in runs:
            axes[i, 2].semilogy(np.arange(len(r["G_hist"])),
                                  r["G_hist"] + 1e-15, "o-", label=r["label"])
        axes[i, 2].set_xlabel("iter")
        axes[i, 2].set_ylabel(r"$\|\nabla E\|$")
        axes[i, 2].set_title(f"D={D}: gradient")
        axes[i, 2].legend(fontsize=7)
        axes[i, 2].grid(True, alpha=0.3)

        # Energy gap vs PM_C3 reference branch
        pm_E = next((r["E"] for r in runs if r["label"] == "PM+C3"), None)
        if pm_E is not None:
            labels = [r["label"] for r in runs]
            dE = [r["E"] - pm_E for r in runs]
            axes[i, 3].bar(range(len(labels)), dE)
            axes[i, 3].set_xticks(range(len(labels)))
            axes[i, 3].set_xticklabels(labels, rotation=45, ha="right",
                                          fontsize=7)
            axes[i, 3].set_ylabel(r"$E - E_{\rm PM+C3}$ (meV)")
            axes[i, 3].set_title(f"D={D}: gap vs PM+C3")
            axes[i, 3].axhline(0, color="black", lw=0.5)
            axes[i, 3].grid(True, alpha=0.3)

    fig.suptitle(f"D scan at n={N_CM12}, T={T}, nk={NK}, kmax={KMAX}, "
                  f"HH={HARTREE_KIND}, ref={REF_KIND}")
    out = (REPO_ROOT / "examples" / "outputs" /
           f"quartermetal_D_scan_n{N_CM12}_{HARTREE_KIND}_{REF_KIND}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
