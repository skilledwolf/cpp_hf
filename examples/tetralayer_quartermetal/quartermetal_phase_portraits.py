"""Combined Fermi-surface + band-structure portraits at representative
(n, D) cells, both PM_C3 and SVP_C3 phases.

Outputs three plots:

  1. Fermi-surface map (one row per cell, 2 columns for PM_C3 and SVP_C3)
     using contimod's plot_fermisurface for proper iso-E contours.

  2. Band-structure plot along the k_y = 0 slice (the rhombic patch's
     mirror line), one row per cell, 2 columns.  Bands intersecting E_F
     are highlighted; the chemical potential is shown as a dashed line.

  3. Phase portrait: SVP_C3 DOS-at-E_F map with circles marking the
     chosen cells, plus FS insets attached to each cell — a single plot
     showing where each portrait sits in the (n, D) phase diagram.

All quantities use the corrected physical Fock (relative + Σ_x[refP]
+ HH-from-refP), with mu re-found against the physical eigenvalues.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parent
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import cpp_hf
from cpp_hf import SolverConfig, solve_direct_minimization
from cpp_hf.fock import build_fock
from cpp_hf.utils import selfenergy_fft, find_chemical_potential
import _common as qm


T = 0.3
NK = 96
# Slightly larger kmax than the production grid (0.30) so the
# C3-incomplete patch boundary sits further from the low-energy region
# around K (the projector only averages points whose full C3 orbit lies
# inside the patch).
KMAX = 0.35
MAX_ITER = 25
MAX_ITER_REF = 200
TOL_E = 1e-4

# Cells chosen to span the (n, D) phase diagram: low-D, transition
# boundary at D≈0.7 (the SVP_C3 horizontal-feature line), high-D, and
# the (n=0.3, D=1.0) cell sitting in the upper-left low-DOS region.
CELLS = [
    (0.20, 0.30, "low-n / low-D"),
    (0.30, 1.00, "low-n / high-D"),
    (0.40, 0.50, "boundary / mid"),
    (0.60, 0.70, "SVP transition (D~0.7)"),
    (0.80, 0.50, "metallic / mid-D"),
    (0.80, 1.00, "metallic / high-D"),
    (1.20, 0.80, "high-n / high-D"),
]


def run_scf(setup, refP, projector_key, seed_key, n_cm12):
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


def physical_fock_eigs(res, kernel, V_H_ref, Sigma_ref):
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
    return fock, eigs


def k_y0_slice(eigs, ks):
    """Extract bands along the k_y=0 mirror line of the rhombic patch.

    Our k-mesh has ks[i, j] = (i/nk - 0.5) b1 + (j/nk - 0.5) b2 with
    b1 = (s/2, -h), b2 = (s/2, +h).  Then k_y = (j - i) * h / nk vanishes
    on the diagonal i = j, where k_x runs from -s/2 to ~+s/2.
    """
    nk = eigs.shape[0]
    diag_idx = np.arange(nk)
    bands = eigs[diag_idx, diag_idx, :]  # shape (nk, n_orb)
    kxs = ks[diag_idx, diag_idx, 0]      # shape (nk,)
    return kxs, bands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eps", type=int, default=2,
                        choices=(2, 3, 4, 5, 10, 20),
                        help="Dual-gate dielectric constant (selects "
                             "hartree_kind=dualgate_eps<eps>).")
    parser.add_argument(
        "--svp-grid", type=Path, default=None,
        help="Override path to the precomputed SVP_C3 DOS grid npz "
             "used for the phase-portrait background.  Defaults to "
             "examples/outputs/quartermetal_grid_svp_c3[_eps<eps>].npz.",
    )
    args = parser.parse_args(argv)

    hartree_kind = f"dualgate_eps{args.eps}"
    suffix = "" if args.eps == 2 else f"_eps{args.eps}"
    out_dir = REPO_ROOT / "examples" / "outputs"
    grid_npz = args.svp_grid if args.svp_grid is not None else (
        out_dir / f"quartermetal_grid_svp_c3{suffix}.npz"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from contimod.plotting.meshgrid import plot_fermisurface

    setup_cache = {}
    cell_data = []  # list of dict with FS + bands per (cell, phase)

    print(f"Running SCFs at representative cells (hartree_kind={hartree_kind})...",
          flush=True)
    for n_cm12, D_Vnm, label in CELLS:
        print(f"  ({n_cm12}, {D_Vnm}) — {label}", flush=True)
        if D_Vnm not in setup_cache:
            setup = qm.build_setup(
                D_Vnm=D_Vnm, nk=NK, kmax=KMAX, T=T, bz_kind="rhombic",
                init_scale=50.0, hartree_kind=hartree_kind,
            )
            refP = qm.pm_c3_cn_reference_density(
                setup, T=T, max_iter=MAX_ITER_REF, tol_E=TOL_E,
            )
            kernel0 = cpp_hf.HartreeFockKernel(
                weights=setup.weights,
                hamiltonian=np.asarray(setup.h_template.hs),
                coulomb_q=setup.Vq, T=T,
                include_hartree=True, include_exchange=True,
                reference_density=refP, hartree_matrix=setup.hartree_matrix,
            )
            w2d = np.asarray(setup.weights)
            Sigma_ref = selfenergy_fft(
                np.asarray(kernel0._VR_shifted), np.asarray(refP),
                _apply_ifftshift=False,
                hermitian_channel_packing=kernel0.exchange_hermitian_channel_packing,
            )
            diag_R_ref = np.einsum("ij,ijaa->a", w2d, np.asarray(refP)).real
            V_H_ref = setup.hartree_matrix @ diag_R_ref
            # Drop the orbital-uniform component of V_H_ref (the "divergent
            # background" — at ε=2 it's ~+280,000 meV, dual-gate q=0 is
            # finite but unphysically large).  Layer differences (the
            # screening signal) are preserved by the centering.
            V_H_ref = V_H_ref - V_H_ref.mean()
            # Same idea for Σ_x[refP]: subtract its k-and-orbital-averaged
            # diagonal trace so absolute band energies sit near zero.
            n_orb_total = Sigma_ref.shape[-1]
            sigma_diag = np.einsum("ijaa->", Sigma_ref).real / (
                Sigma_ref.shape[0] * Sigma_ref.shape[1] * n_orb_total
            )
            Sigma_ref = Sigma_ref - np.eye(n_orb_total)[None, None] * sigma_diag
            setup_cache[D_Vnm] = (setup, refP, V_H_ref, Sigma_ref)
        setup, refP, V_H_ref, Sigma_ref = setup_cache[D_Vnm]
        ks = np.asarray(setup.h_template.kmesh.ks)
        bounds = np.asarray(setup.h_template.kmesh.bounds)

        results_per_phase = {}
        for proj_key, seed_key in [("PM_C3", "PM"), ("SVP_C3", "SVP")]:
            t0 = time.perf_counter()
            res, kernel, n_e = run_scf(setup, refP, proj_key, seed_key, n_cm12)
            fock, eigs = physical_fock_eigs(res, kernel, V_H_ref, Sigma_ref)
            E_F = float(find_chemical_potential(
                eigs, np.asarray(setup.weights), n_e, T, method="bisection"
            ))
            kxs, bands = k_y0_slice(eigs, ks)
            results_per_phase[proj_key] = dict(
                ks=ks, bounds=bounds,
                eigs=eigs, E_F=E_F,
                kxs=kxs, bands=bands,
                converged=bool(res.converged), n_iter=int(res.n_iter),
            )
            print(f"    {proj_key}: it={res.n_iter} conv={int(res.converged)} "
                  f"E_F={E_F:.1f}  t={time.perf_counter()-t0:.1f}s", flush=True)
        cell_data.append((n_cm12, D_Vnm, label, results_per_phase))

    # ---- Plot 1: Fermi surfaces ----
    fig_fs, axes_fs = plt.subplots(
        len(CELLS), 2, figsize=(11, 4.5 * len(CELLS)),
        constrained_layout=True,
    )
    for row, (n_cm12, D_Vnm, label, phases) in enumerate(cell_data):
        for col, proj_key in enumerate(["PM_C3", "SVP_C3"]):
            d = phases[proj_key]
            plot_fermisurface(
                d["ks"][..., 0], d["ks"][..., 1],
                d["eigs"], energies=[d["E_F"]],
                ax=axes_fs[row, col], bounds=d["bounds"],
                linewidths=1.4,
            )
            axes_fs[row, col].set_title(
                f"{proj_key}: (n={n_cm12}, D={D_Vnm}) — {label}\n"
                f"iters={d['n_iter']}, conv={int(d['converged'])}, "
                f"E_F={d['E_F']:.1f} meV",
                fontsize=9,
            )
    fig_fs.suptitle(
        f"Fermi surfaces at representative cells (ε={hartree_kind}, "
        f"nk={NK}, kmax={KMAX}, T={T} meV)",
        fontsize=11,
    )
    out_fs = out_dir / f"quartermetal_fermi_surfaces{suffix}.png"
    fig_fs.savefig(out_fs, dpi=140)
    plt.close(fig_fs)
    print(f"\nSaved {out_fs}")

    # ---- Plot 2: Band structures (k_y = 0 slice, centered on E_F) ----
    # Use scatter (not lines) to avoid false connections at avoided
    # crossings — eigvalsh returns sorted eigenvalues per k, so plotting
    # band b across k as a connected line creates spurious ridges.
    fig_bs, axes_bs = plt.subplots(
        len(CELLS), 2, figsize=(11, 3.0 * len(CELLS)),
        constrained_layout=True,
    )
    yzoom = 50.0  # meV around E_F
    for row, (n_cm12, D_Vnm, label, phases) in enumerate(cell_data):
        for col, proj_key in enumerate(["PM_C3", "SVP_C3"]):
            d = phases[proj_key]
            ax = axes_bs[row, col]
            kxs = d["kxs"]
            bands = d["bands"]
            E_F = d["E_F"]
            de = bands - E_F
            in_window = np.abs(de) <= yzoom * 1.5
            for b in range(bands.shape[1]):
                mask = in_window[:, b]
                ax.scatter(kxs[mask], de[mask, b],
                            s=6, color="k", alpha=0.5, edgecolors="none")
            ax.axhline(0.0, color="tab:red", lw=0.8, linestyle="dashed")
            ax.set_ylim(-yzoom, yzoom)
            ax.set_xlim(kxs.min(), kxs.max())
            ax.set_xlabel(r"$k_x$ (1/lat)")
            ax.set_ylabel(r"$E - E_F$ (meV)")
            ax.set_title(
                f"{proj_key}: (n={n_cm12}, D={D_Vnm}) — {label}",
                fontsize=9,
            )
            ax.grid(True, alpha=0.3)
    fig_bs.suptitle(
        f"Band structure scatter at $k_y=0$, centered on $E_F$ "
        f"(ε={hartree_kind}, nk={NK}, ±{yzoom:.0f} meV window)",
        fontsize=11,
    )
    out_bs = out_dir / f"quartermetal_band_structures{suffix}.png"
    fig_bs.savefig(out_bs, dpi=140)
    plt.close(fig_bs)
    print(f"Saved {out_bs}")

    # ---- Plot 3: Phase portrait — SVP_C3 DOS map with FS insets ----
    grid = np.load(grid_npz, allow_pickle=True)
    n_grid = grid["n_grid"]
    D_grid = grid["D_grid"]
    extent = (n_grid[0], n_grid[-1], D_grid[0], D_grid[-1])
    dos_arr = np.where(grid["dos_at_EF"] > 0, grid["dos_at_EF"], np.nan)
    p97 = float(np.nanpercentile(dos_arr, 97))

    fig_pp, ax_pp = plt.subplots(figsize=(13, 11), constrained_layout=False)
    im = ax_pp.imshow(dos_arr, origin="lower", aspect="auto",
                       extent=extent, cmap="viridis", vmin=0, vmax=p97)
    fig_pp.colorbar(
        im, ax=ax_pp,
        label=r"SVP_C3 $\rho(E_F)$ (linear, clipped at p97)",
        shrink=0.5, location="left",
    )
    ax_pp.set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    ax_pp.set_ylabel(r"$D$ (V/nm)")
    ax_pp.set_title("SVP_C3 phase diagram with Fermi-surface insets")

    # Add inset axes positioned at the marked cells.  Use ax.inset_axes with
    # data coordinates so they track the (n, D) point.
    inset_size_data = 0.08  # fraction of (n, D) extent
    n_span = float(n_grid[-1] - n_grid[0])
    D_span = float(D_grid[-1] - D_grid[0])
    for n_cm12, D_Vnm, label, phases in cell_data:
        ax_pp.plot(n_cm12, D_Vnm, "o", color="white",
                    markersize=8, markeredgecolor="black", zorder=5)
        # Choose inset position that doesn't crash into the boundary
        nx = (n_cm12 - n_grid[0]) / n_span
        ny = (D_Vnm - D_grid[0]) / D_span
        # Position inset just NE or SE of the dot, depending on quadrant
        dx = 0.06 if nx < 0.7 else -0.20
        dy = 0.06 if ny < 0.7 else -0.20
        ax_in = ax_pp.inset_axes(
            [nx + dx, ny + dy, 0.13, 0.13],
            transform=ax_pp.transAxes,
        )
        d = phases["SVP_C3"]
        plot_fermisurface(
            d["ks"][..., 0], d["ks"][..., 1],
            d["eigs"], energies=[d["E_F"]],
            ax=ax_in, bounds=d["bounds"],
            linewidths=0.8,
        )
        ax_in.set_xlabel("")
        ax_in.set_ylabel("")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title("", fontsize=0)
        # Connect inset corner back to the cell point with a thin line
        nx_in = nx + dx
        ny_in = ny + dy
        ax_pp.annotate(
            "", xy=(n_cm12, D_Vnm), xycoords="data",
            xytext=(nx_in + 0.065, ny_in + 0.065),
            textcoords=ax_pp.transAxes,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )

    out_pp = out_dir / f"quartermetal_phase_portrait{suffix}.png"
    fig_pp.savefig(out_pp, dpi=140, bbox_inches="tight")
    plt.close(fig_pp)
    print(f"Saved {out_pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
