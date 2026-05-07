"""Confirm whether 15-iter D-scan SVP/SVP+C3 energies are stuck or near-converged.

Re-run a focused subset (n=0.8, D=0.50, ε=5, sameU and u0 ref) for 100 iters
and compare to the 15-iter values.  If SVP energy continues dropping below
PM with more iterations, our PM-wins verdict is suspect.
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

N_CM12 = 0.80
D_VNM = 0.50
T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_eps5"
MAX_ITER = 100


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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    all_runs = {}

    for col, ref_kind in enumerate(("u0", "sameU")):
        print(f"\n=== ref={ref_kind} ===")
        setup = qm.build_setup(D_Vnm=D_VNM, nk=NK, kmax=KMAX, T=T,
                                bz_kind="rhombic", init_scale=50.0,
                                hartree_kind=HARTREE_KIND)
        refP = qm.noninteracting_cn_reference(setup, T=T, ref_kind=ref_kind)
        n_e, h_run = qm.n_electrons_for_density(setup, N_CM12, T)
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
            coulomb_q=setup.Vq, T=T,
            include_hartree=True, include_exchange=True,
            reference_density=refP, hartree_matrix=setup.hartree_matrix,
        )
        seed_pm = qm.initial_density_from_seed(h_run, setup.seeds["PM"], T)
        seed_svp = qm.initial_density_from_seed(h_run, setup.seeds["SVP"], T)

        runs = []
        for label, projector_key, seed in [
            ("PM",      "PM",     seed_pm),
            ("PM+C3",   "PM_C3",  seed_pm),
            ("SVP",     "SVP",    seed_svp),
            ("SVP+C3",  "SVP_C3", seed_svp),
        ]:
            cfg = SolverConfig(
                max_iter=MAX_ITER, tol_E=1e-6, max_step=0.6,
                block_sizes=(8, 8, 8, 8),
                project_fn=setup.project_fns[projector_key],
            )
            print(f"Running {label} ({ref_kind})...", flush=True)
            t0 = time.perf_counter()
            res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
            E_hist = np.asarray(res.history["E"])
            G_hist = np.asarray(res.history["grad_norm"])
            imb = imbalance(res.density, refP, setup.weights)
            runs.append(dict(
                label=label, n_iter=int(res.n_iter),
                converged=bool(res.converged),
                E=float(res.energy), E_hist=E_hist, G_hist=G_hist,
                imbalance=imb, ref_kind=ref_kind,
            ))
            print(f"  it={int(res.n_iter):3d} conv={int(res.converged)} "
                  f"E={float(res.energy):+9.3f}  imb={imb:7.0f}  "
                  f"t={time.perf_counter()-t0:.1f}s")
        all_runs[ref_kind] = runs

        # Energy trajectory
        for r in runs:
            axes[0, col].plot(np.arange(len(r["E_hist"])), r["E_hist"], "o-",
                              label=f"{r['label']} (it={r['n_iter']}, imb={r['imbalance']:.0f})")
        axes[0, col].set_xlabel("iter")
        axes[0, col].set_ylabel(r"$E$ (meV)")
        axes[0, col].set_title(f"{ref_kind}: free energy")
        axes[0, col].legend(fontsize=8)
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].axvline(15, color="gray", ls="--", alpha=0.5,
                             label="15-iter cap")

        # Gradient norm
        for r in runs:
            axes[1, col].semilogy(np.arange(len(r["G_hist"])),
                                  r["G_hist"] + 1e-15, "o-",
                                  label=r["label"])
        axes[1, col].set_xlabel("iter")
        axes[1, col].set_ylabel(r"$\|\nabla E\|$")
        axes[1, col].set_title(f"{ref_kind}: gradient")
        axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].axvline(15, color="gray", ls="--", alpha=0.5)

    # Comparison column: gap-vs-PM at iter 15 vs final
    ax = axes[0, 2]
    labels = ["PM", "PM+C3", "SVP", "SVP+C3"]
    width = 0.35
    x = np.arange(len(labels))
    for i, ref_kind in enumerate(("u0", "sameU")):
        runs = all_runs[ref_kind]
        pm_E = next((r["E"] for r in runs if r["label"] == "PM"), None)
        gaps = []
        for lab in labels:
            r = next((rr for rr in runs if rr["label"] == lab), None)
            gaps.append(r["E"] - pm_E if r is not None else 0)
        ax.bar(x + i*width - width/2, gaps, width, label=ref_kind)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$E - E_{\rm PM}$ (meV)")
    ax.set_title(f"Final gap vs PM, max_iter={MAX_ITER}")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Show energy at iter 15 vs final to indicate how much the cap was biasing
    ax = axes[1, 2]
    for i, ref_kind in enumerate(("u0", "sameU")):
        runs = all_runs[ref_kind]
        for r in runs:
            E = r["E_hist"]
            if len(E) <= 15:
                continue
            E15 = E[15]
            Ef = E[-1]
            ax.plot([f"{r['label']}\n{ref_kind}"], [E15 - Ef], "o",
                    label=None)
    ax.set_ylabel(r"$E_{15} - E_{\rm final}$ (meV)")
    ax.set_title("How much did 15 vs 100 iters matter?")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Long-iter check: n={N_CM12}, D={D_VNM}, "
                 f"T={T}, nk={NK}, ε={HARTREE_KIND}, max_iter={MAX_ITER}")
    out = (REPO_ROOT / "examples" / "outputs" /
           f"quartermetal_long_iter_check_n{N_CM12}_D{D_VNM}_{HARTREE_KIND}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
