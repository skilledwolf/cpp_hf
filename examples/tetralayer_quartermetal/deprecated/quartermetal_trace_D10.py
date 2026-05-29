"""Trace at (n=0.8, D=1.0) with ε_eff=5.25 (geomean) — copies the convergence
trace2 logic but at a different cell."""
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

N_CM12 = 0.80
D_VNM = 1.00
T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_geomean"
MAX_ITER = 25

CACHE = (REPO_ROOT / "examples" / "outputs" /
          f"quartermetal_trace_n{N_CM12}_D{D_VNM}_{HARTREE_KIND}.npz")
PNG_OUT = (REPO_ROOT / "examples" / "outputs" /
           f"quartermetal_trace_n{N_CM12}_D{D_VNM}_{HARTREE_KIND}.png")


def _imbalance(P, refP, w2d, n_orb=8):
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    setup = qm.build_setup(D_Vnm=D_VNM, nk=NK, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0,
                            hartree_kind=HARTREE_KIND)
    refP = qm.noninteracting_cn_reference(setup, T=T)
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
        ("PM", kernel_HF, "PM", seed_pm),
        ("PM+C3", kernel_HF, "PM_C3", seed_pm),
        ("SVP", kernel_HF, "SVP", seed_svp),
        ("SVP+C3", kernel_HF, "SVP_C3", seed_svp),
        ("X-only PM+C3", kernel_X, "PM_C3", seed_pm),
        ("X-only SVP", kernel_X, "SVP", seed_svp),
        ("X-only SVP+C3", kernel_X, "SVP_C3", seed_svp),
    ]:
        cfg = SolverConfig(max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
                            block_sizes=(8, 8, 8, 8),
                            project_fn=setup.project_fns[projector_key])
        print(f"Running {label}...", flush=True)
        t0 = time.perf_counter()
        res = solve_direct_minimization(kernel_use, seed, n_e, config=cfg)
        E_hist = np.asarray(res.history["E"])
        G_hist = np.asarray(res.history["grad_norm"])
        imb = _imbalance(res.density, refP, setup.weights)
        runs.append(dict(
            label=label, n_iter=int(res.n_iter),
            converged=bool(res.converged),
            E=float(res.energy), mu=float(res.mu), imbalance=imb,
            E_hist=E_hist, G_hist=G_hist,
        ))
        print(f"  it={int(res.n_iter)} conv={int(res.converged)} "
              f"E={float(res.energy):+.3f} μ={float(res.mu):+.2f} "
              f"imb={imb:7.1f}  t={time.perf_counter()-t0:.1f}s")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for r in runs:
        axes[0, 0].plot(np.arange(len(r["E_hist"])), r["E_hist"], "o-",
                          label=f"{r['label']} (it={r['n_iter']}, imb={r['imbalance']:.0f})")
    axes[0, 0].set_xlabel("iter")
    axes[0, 0].set_ylabel(r"$E$ (meV)")
    axes[0, 0].set_title("Free-energy trajectory")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    for r in runs:
        E = r["E_hist"]
        if len(E) < 2: continue
        de = np.abs(E - E[-1]) + 1e-15
        axes[0, 1].semilogy(np.arange(len(de)), de, "o-", label=r["label"])
    axes[0, 1].set_xlabel("iter")
    axes[0, 1].set_ylabel(r"$|E_n - E_{\rm final}|$")
    axes[0, 1].set_title("Convergence rate")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    for r in runs:
        axes[1, 0].semilogy(np.arange(len(r["G_hist"])), r["G_hist"] + 1e-15,
                              "o-", label=r["label"])
    axes[1, 0].set_xlabel("iter")
    axes[1, 0].set_ylabel(r"$\|\nabla E\|$")
    axes[1, 0].set_title("Gradient norm")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Energy gap vs PM
    pm_E = next((r["E"] for r in runs if r["label"] == "PM"), None)
    if pm_E is not None:
        labels = [r["label"] for r in runs]
        dE_per_pm = [r["E"] - pm_E for r in runs]
        axes[1, 1].bar(range(len(labels)), dE_per_pm)
        axes[1, 1].set_xticks(range(len(labels)))
        axes[1, 1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes[1, 1].set_ylabel(r"$E - E_{\rm PM}$ (meV)")
        axes[1, 1].set_title("Energy gap vs PM")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(0, color="black", lw=0.5)

    fig.suptitle(f"Convergence at (n={N_CM12}, D={D_VNM}), T={T}, "
                  f"nk={NK}, kmax={KMAX}, ε_eff=5.25 (geomean)")
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)
    print(f"\nSaved {PNG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
