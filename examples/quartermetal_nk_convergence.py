"""nk convergence check at ε=2 with PM_C3_CN reference.

Tests a handful of representative (n, D) cells at nk ∈ (32, 48, 64, 96)
and tabulates Energy, Δ_tb, DOS(E_F) per cell.  Goal: decide whether
nk=48 is sufficient for the 30×30 grid (4× faster than nk=96).
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
from cpp_hf.fock import build_fock
import _quartermetal_common as qm

T = 0.3
KMAX = 0.30
HARTREE_KIND = "dualgate_eps2"
MAX_ITER_HF = 25
MAX_ITER_REF = 200
TOL_E = 1e-4

NKS = (32, 48, 64, 96)
CELLS = [
    (0.30, 0.30),
    (0.80, 0.30),
    (0.80, 0.50),
    (1.20, 0.80),
]


def run_one(n_cm12, D_Vnm, nk):
    setup = qm.build_setup(D_Vnm=D_Vnm, nk=nk, kmax=KMAX, T=T,
                            bz_kind="rhombic", init_scale=50.0,
                            hartree_kind=HARTREE_KIND)
    refP = qm.pm_c3_cn_reference_density(
        setup, T=T, max_iter=MAX_ITER_REF, tol_E=TOL_E,
    )
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed = qm.initial_density_from_seed(h_run, setup.seeds["PM"], T)
    cfg = SolverConfig(
        max_iter=MAX_ITER_HF, tol_E=TOL_E, max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns["PM_C3"],
    )
    t0 = time.perf_counter()
    res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
    dt = time.perf_counter() - t0

    P_arr = np.ascontiguousarray(np.asarray(res.density), dtype=np.complex128)
    Sigma, H, _ = build_fock(
        P_arr, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g, contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    fock = np.asarray(kernel.h) + np.asarray(Sigma) + np.asarray(H)
    layer = np.asarray(setup.model.layer)
    delta_tb = qm.delta_tb_from_fock(fock, layer)

    # DOS at E_F via three methods to diagnose nk discretization sensitivity:
    #   (a) Lorentzian smearing  (eta = 5T)
    #   (b) Fermi-Dirac smearing (-df/dε, width T) — what existing grid uses
    #   (c) bztetra.twod tetrahedron interpolation — should be best.
    eigs = np.linalg.eigvalsh(fock.reshape(-1, fock.shape[-2], fock.shape[-1]))
    eigs = eigs.reshape(*fock.shape[:2], -1)
    mu_E = float(res.mu)
    w = np.asarray(setup.weights)

    eta = 5.0 * T
    L = (eta / np.pi) / ((eigs - mu_E) ** 2 + eta ** 2)
    dos_lorentz = float(np.einsum("ij,ijb->", w, L).real)

    dos_fermi = qm.dos_at_mu(eigs, w, mu_E, T)

    import bztetra.twod as twod
    s = 2.0 * KMAX
    h_tri = s * np.sqrt(3.0) / 2.0
    Bmat = np.array([[s / 2.0, s / 2.0], [-h_tri, h_tri]])
    W = twod.density_of_states_weights(Bmat, eigs, np.array([mu_E]))
    # Canonical bztetra DOS: sum(W) * area_prefactor where
    # area_prefactor = |det(B)| / (2pi)^2.  In contimod's setup,
    # setup.weights (which we use for everything else) sums to exactly
    # this prefactor — so multiplying sum(W) by w.sum() puts bztetra
    # DOS in the same units as qm.dos_at_mu.
    area_prefactor = float(w.sum())  # equals |det(Bmat)| / (2pi)^2
    dos_bztetra = float(W.sum()) * area_prefactor

    near_EF = np.abs(eigs - mu_E)
    n_within_eta = int(np.sum(near_EF < eta))
    min_gap = float(np.min(near_EF))
    return dict(
        nk=nk, n=n_cm12, D=D_Vnm,
        energy=float(res.energy), mu=float(res.mu),
        delta_tb=float(delta_tb),
        dos_lorentz=dos_lorentz, dos_fermi=dos_fermi, dos_bztetra=dos_bztetra,
        n_near_EF=n_within_eta, min_gap=min_gap,
        iters=int(res.n_iter), conv=bool(res.converged),
        wall_s=float(dt),
    )


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"{'cell':<14s} {'nk':>4s} {'iters':>5s} {'conv':>4s} "
          f"{'E':>10s} {'Δ_tb':>9s} "
          f"{'DOS_lor':>9s} {'DOS_fer':>9s} {'DOS_bzt':>9s} "
          f"{'n_nearEF':>9s} {'mingap':>8s} {'t(s)':>6s}")
    rows = []
    for n_cm12, D_Vnm in CELLS:
        for nk in NKS:
            r = run_one(n_cm12, D_Vnm, nk)
            rows.append(r)
            print(f"({n_cm12:.2f},{D_Vnm:.2f}) {nk:>4d} {r['iters']:>5d} "
                  f"{int(r['conv']):>4d} {r['energy']:>+10.4f} "
                  f"{r['delta_tb']:>+9.3f} "
                  f"{r['dos_lorentz']:>10.6f} {r['dos_fermi']:>10.6f} "
                  f"{r['dos_bztetra']:>10.6f} "
                  f"{r['n_near_EF']:>9d} {r['min_gap']:>8.4f} "
                  f"{r['wall_s']:>6.1f}", flush=True)

    fig, axes = plt.subplots(3, len(CELLS), figsize=(4*len(CELLS), 10),
                              constrained_layout=True)
    for col, (n_cm12, D_Vnm) in enumerate(CELLS):
        cell_rows = [r for r in rows if r["n"] == n_cm12 and r["D"] == D_Vnm]
        nks = [r["nk"] for r in cell_rows]
        E = np.array([r["energy"] for r in cell_rows])
        d_tb = np.array([r["delta_tb"] for r in cell_rows])
        dos_l = np.array([r["dos_lorentz"] for r in cell_rows])
        dos_f = np.array([r["dos_fermi"] for r in cell_rows])
        dos_b = np.array([r["dos_bztetra"] for r in cell_rows])
        E_ref = E[-1]; dtb_ref = d_tb[-1]

        axes[0, col].plot(nks, E - E_ref, "o-")
        axes[0, col].set_xlabel("nk")
        axes[0, col].set_ylabel(r"$E - E_{\rm nk=96}$ (meV)")
        axes[0, col].set_title(f"(n={n_cm12}, D={D_Vnm}): energy")
        axes[0, col].axhline(0, color="black", lw=0.5)
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(nks, d_tb - dtb_ref, "o-", color="tab:green")
        axes[1, col].set_xlabel("nk")
        axes[1, col].set_ylabel(r"$\Delta_{\rm tb} - \Delta_{\rm tb, nk=96}$ (meV)")
        axes[1, col].set_title(f"(n={n_cm12}, D={D_Vnm}): Δ_tb")
        axes[1, col].axhline(0, color="black", lw=0.5)
        axes[1, col].grid(True, alpha=0.3)

        axes[2, col].plot(nks, dos_l, "o-", color="tab:purple", label="Lorentz")
        axes[2, col].plot(nks, dos_f, "s-", color="tab:orange", label="Fermi")
        axes[2, col].plot(nks, dos_b, "^-", color="tab:red", label="bztetra")
        axes[2, col].set_xlabel("nk")
        axes[2, col].set_ylabel(r"DOS at $E_F$")
        axes[2, col].set_title(f"(n={n_cm12}, D={D_Vnm}): DOS")
        axes[2, col].legend(fontsize=8)
        axes[2, col].grid(True, alpha=0.3)

    fig.suptitle(f"nk convergence check, ε=2, T={T}, kmax={KMAX}, "
                 f"max_iter_hf={MAX_ITER_HF}")
    out = (REPO_ROOT / "examples" / "outputs" /
           "quartermetal_nk_convergence_eps2.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
