"""Fermi-surface maps at representative (n, D) cells, for both PM_C3 and
SVP_C3 phases, using the corrected physical Fock and re-found mu.

For each cell:
  1. Build setup at (n, D), compute the PM_C3 self-consistent CN reference
  2. Run PM_C3 SCF and SVP_C3 SCF at the same n (separate seeds)
  3. Build physical Fock = relative + Σ_x[refP] + diag(HH @ diag(refP))
  4. Find E_F such that occupation = n_e
  5. Plot the spectral weight at E_F per k:
       A(k, E_F) = Σ_n  η/π / ((E_n(k) - E_F)² + η²)
     summed over bands.  This lights up wherever a band crosses E_F.

Saves a multi-panel PNG comparing the two phases side by side at each
chosen (n, D) cell.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
from cpp_hf.utils import selfenergy_fft, find_chemical_potential
import _quartermetal_common as qm


T = 0.3
NK = 48
KMAX = 0.30
HARTREE_KIND = "dualgate_eps2"
MAX_ITER = 25
MAX_ITER_REF = 200
TOL_E = 1e-4
ETA = 0.5  # spectral broadening (meV) for FS visualization

CELLS = [
    (0.20, 0.30, "gapped low-n"),
    (0.40, 0.70, "boundary"),
    (0.50, 0.70, "bright DOS spot"),
    (0.80, 0.50, "metallic mid"),
    (1.00, 0.80, "metallic high-D"),
    (1.30, 0.30, "metallic high-n"),
]


def run_scf(setup, refP, kernel_HF, projector_key, seed_key, n_cm12):
    n_e, h_run = qm.n_electrons_for_density(setup, n_cm12, T)
    kernel = cpp_hf.HartreeFockKernel(
        weights=setup.weights, hamiltonian=np.asarray(h_run.hs),
        coulomb_q=setup.Vq, T=T,
        include_hartree=True, include_exchange=True,
        reference_density=refP, hartree_matrix=setup.hartree_matrix,
    )
    seed = qm.initial_density_from_seed(h_run, setup.seeds[seed_key], T)
    cfg = SolverConfig(
        max_iter=MAX_ITER, tol_E=TOL_E, max_step=0.6,
        block_sizes=(8, 8, 8, 8),
        project_fn=setup.project_fns[projector_key],
    )
    res = solve_direct_minimization(kernel, seed, n_e, config=cfg)
    return res, kernel, n_e


def physical_fock_eigs(res, kernel, refP, setup, V_H_ref, Sigma_ref):
    P_arr = np.ascontiguousarray(np.asarray(res.density), dtype=np.complex128)
    Sigma, H, _ = build_fock(
        P_arr, h=kernel.h, VR=kernel._VR_shifted, refP=kernel.refP,
        HH=kernel.HH, w2d=kernel.w2d,
        include_exchange=kernel.include_exchange,
        include_hartree=kernel.include_hartree,
        exchange_hermitian_channel_packing=kernel.exchange_hermitian_channel_packing,
        contact_g=kernel.contact_g,
        contact_Oi=kernel.contact_Oi,
        contact_Oj=kernel.contact_Oj,
    )
    fock_rel = np.asarray(kernel.h) + np.asarray(Sigma) + np.asarray(H)
    n_orb_total = fock_rel.shape[-1]
    fock = fock_rel + np.asarray(Sigma_ref)
    for a in range(n_orb_total):
        fock[:, :, a, a] = fock[:, :, a, a] + V_H_ref[a]
    eigs = np.linalg.eigvalsh(fock.reshape(-1, fock.shape[-2], fock.shape[-1]))
    eigs = eigs.reshape(*fock.shape[:2], -1)
    return eigs


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Cache per-D setup + reference (so we don't recompute for shared D)
    setup_cache = {}

    # Layout: 2 columns (PM_C3, SVP_C3), len(CELLS) rows
    fig, axes = plt.subplots(len(CELLS), 2,
                              figsize=(12, 5.5 * len(CELLS)),
                              constrained_layout=True)
    if len(CELLS) == 1:
        axes = axes[None, :]

    for row, (n_cm12, D_Vnm, label) in enumerate(CELLS):
        print(f"\n=== ({n_cm12}, {D_Vnm}) {label} ===", flush=True)
        if D_Vnm not in setup_cache:
            setup = qm.build_setup(D_Vnm=D_Vnm, nk=NK, kmax=KMAX, T=T,
                                    bz_kind="rhombic", init_scale=50.0,
                                    hartree_kind=HARTREE_KIND)
            refP = qm.pm_c3_cn_reference_density(
                setup, T=T, max_iter=MAX_ITER_REF, tol_E=TOL_E,
            )
            kernel_HF = cpp_hf.HartreeFockKernel(
                weights=setup.weights, hamiltonian=np.asarray(setup.h_template.hs),
                coulomb_q=setup.Vq, T=T,
                include_hartree=True, include_exchange=True,
                reference_density=refP, hartree_matrix=setup.hartree_matrix,
            )
            w2d = np.asarray(setup.weights)
            Sigma_ref = selfenergy_fft(
                np.asarray(kernel_HF._VR_shifted), np.asarray(refP),
                _apply_ifftshift=False,
                hermitian_channel_packing=kernel_HF.exchange_hermitian_channel_packing,
            )
            diag_R_ref = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
            V_H_ref = setup.hartree_matrix @ diag_R_ref
            setup_cache[D_Vnm] = (setup, refP, kernel_HF, V_H_ref, Sigma_ref)
        setup, refP, kernel_HF, V_H_ref, Sigma_ref = setup_cache[D_Vnm]
        ks = np.asarray(setup.h_template.kmesh.ks)
        kx = ks[..., 0]
        ky = ks[..., 1]

        for col, (proj_key, seed_key, phase_label) in enumerate([
            ("PM_C3", "PM", "PM_C3"),
            ("SVP_C3", "SVP", "SVP_C3"),
        ]):
            t0 = time.perf_counter()
            res, kernel, n_e = run_scf(setup, refP, kernel_HF,
                                         proj_key, seed_key, n_cm12)
            eigs = physical_fock_eigs(res, kernel, refP, setup,
                                        V_H_ref, Sigma_ref)
            E_F = float(find_chemical_potential(eigs, np.asarray(setup.weights),
                                                  n_e, T,
                                                  method="bisection"))
            # Use contimod's proper FS contour extractor: for each band
            # whose surface crosses E_F, ax.contour at level=E_F draws the
            # closed/open Fermi pockets.
            from contimod.plotting.meshgrid import plot_fermisurface
            bounds = np.asarray(setup.h_template.kmesh.bounds)
            plot_fermisurface(
                kx, ky, eigs, energies=[E_F],
                ax=axes[row, col], bounds=bounds,
                linewidths=1.2,
            )
            axes[row, col].set_title(
                f"{phase_label}: (n={n_cm12}, D={D_Vnm}) — {label}\n"
                f"iters={res.n_iter}, conv={int(res.converged)}, "
                f"E_F={E_F:.1f} meV"
            )
            print(f"  {phase_label:<8s}  iters={res.n_iter} conv={int(res.converged)} "
                  f"E_F={E_F:+.2f}  t={time.perf_counter()-t0:.1f}s", flush=True)

    fig.suptitle(
        f"Fermi-surface maps: ε={HARTREE_KIND}, nk={NK}, kmax={KMAX}, "
        f"T={T} meV, η={ETA} meV"
    )
    out = REPO_ROOT / "examples" / "outputs" / "quartermetal_fermi_surfaces.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
