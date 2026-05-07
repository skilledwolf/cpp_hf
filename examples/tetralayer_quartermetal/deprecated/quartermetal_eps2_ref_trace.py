"""Trace the PM_C3 self-consistent CN reference at ε=2, n=0.8, with the
project_fn-capture trick used in convergence_trace2.py.  Shows E, gradient
norm, and Δ_tb(top-bottom layer potential) per iter — diagnoses whether the
reference is oscillating or just slow.
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

T = 0.3
NK = 96
KMAX = 0.30
HARTREE_KIND = "dualgate_eps2"
MAX_ITER = 25
D_VALUES = (0.03, 0.10, 0.50)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, len(D_VALUES), figsize=(5 * len(D_VALUES), 11),
                              constrained_layout=True)

    for col, D in enumerate(D_VALUES):
        print(f"\n=== D = {D} V/nm ===", flush=True)
        setup = qm.build_setup(D_Vnm=D, nk=NK, kmax=KMAX, T=T,
                                bz_kind="rhombic", init_scale=50.0,
                                hartree_kind=HARTREE_KIND)
        # Bootstrap PM_C3 self-consistent reference at CN filling, capturing
        # the density at every project_fn call.
        h_arr = np.asarray(setup.h_template.hs)
        refP_init = qm.noninteracting_cn_reference(setup, T=T, ref_kind="u0")
        kernel = cpp_hf.HartreeFockKernel(
            weights=setup.weights, hamiltonian=h_arr,
            coulomb_q=setup.Vq, T=T,
            include_hartree=True, include_exchange=True,
            reference_density=refP_init,
            hartree_matrix=setup.hartree_matrix,
        )

        capture = []
        base_proj = setup.project_fns["PM_C3"]
        def proj(P):
            out = base_proj(P)
            capture.append(np.array(out, copy=True))
            return out

        cfg = SolverConfig(
            max_iter=MAX_ITER, tol_E=1e-4, max_step=0.6,
            block_sizes=(8, 8, 8, 8), project_fn=proj,
        )
        t0 = time.perf_counter()
        res = solve_direct_minimization(
            kernel, refP_init, float(setup.ne_cn), config=cfg,
        )
        dt = time.perf_counter() - t0
        print(f"  it={int(res.n_iter)} conv={int(res.converged)} "
              f"E={float(res.energy):+.3f}  t={dt:.1f}s  "
              f"({len(capture)} project_fn calls)")

        E_hist = np.asarray(res.history["E"])
        G_hist = np.asarray(res.history["grad_norm"])

        # Δ_tb per iter using captured densities and external Hartree response.
        # Use bare h_diag plus V_H = HH @ (diag(P) - diag(refP_init)).
        n_calls_per_iter = max(1, len(capture) // max(int(res.n_iter), 1))
        layer = np.asarray(setup.model.layer)
        layers_sorted = np.sort(np.unique(layer))
        h_diag_k0 = np.einsum("ijaa->a", h_arr).real / (NK * NK)
        diag_R = np.einsum("ij,ijaa->a", setup.weights, refP_init).real
        delta_tb = []
        for i in range(0, len(capture), n_calls_per_iter):
            P_i = capture[i]
            diag_P = np.einsum("ij,ijaa->a", setup.weights, P_i).real
            sigma = diag_P - diag_R
            V_H = setup.hartree_matrix @ sigma
            F_diag = h_diag_k0 + V_H
            bot = float(F_diag[layer == layers_sorted[0]].mean())
            top = float(F_diag[layer == layers_sorted[-1]].mean())
            delta_tb.append(top - bot)
        delta_tb = np.array(delta_tb)

        # Row 0: E history
        axes[0, col].plot(np.arange(len(E_hist)), E_hist, "o-",
                          label=f"PM_C3 ref (it={res.n_iter}, conv={int(res.converged)})")
        axes[0, col].set_xlabel("iter")
        axes[0, col].set_ylabel(r"$E_{\rm CN, PM\_C3}$ (meV)")
        axes[0, col].set_title(f"D={D}: free-energy trajectory of CN ref")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=8)

        # Row 1: gradient norm
        axes[1, col].semilogy(np.arange(len(G_hist)), G_hist + 1e-15, "o-",
                               color="tab:red")
        axes[1, col].set_xlabel("iter")
        axes[1, col].set_ylabel(r"$\|\nabla E\|$")
        axes[1, col].set_title(f"D={D}: gradient norm")
        axes[1, col].grid(True, alpha=0.3)

        # Row 2: Δ_tb
        axes[2, col].plot(np.arange(len(delta_tb)), delta_tb, "o-",
                          color="tab:green")
        axes[2, col].set_xlabel("iter")
        axes[2, col].set_ylabel(r"$\Delta_{\rm tb}$ (meV)")
        axes[2, col].set_title(f"D={D}: top-bottom layer potential")
        axes[2, col].grid(True, alpha=0.3)
        axes[2, col].axhline(0, color="black", lw=0.5)

    fig.suptitle(f"PM_C3 CN reference convergence at ε=2, n=CN, "
                 f"T={T}, nk={NK}, max_iter={MAX_ITER}")
    out = (REPO_ROOT / "examples" / "outputs" /
           f"quartermetal_eps2_ref_trace_nk{NK}_kmax{KMAX}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
