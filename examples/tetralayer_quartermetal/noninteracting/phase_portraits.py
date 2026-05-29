"""Bare (non-interacting) phase portrait in the (n_e, Δ_tb) plane.

Three plots, mirroring quartermetal_phase_portraits.py but with no SCF
(just bare h(k) diagonalization):

  1. Fermi-surface map at representative cells.
  2. Band-structure scatter along the k_y = 0 mirror line.
  3. Phase portrait: linear DOS-at-E_F (background from
     ``noninteracting_dos_grid.npz``, produced by ``dos_grid.py``) with
     FS insets at each cell.

Conventions:
  - x-axis = per-flavor density n_e (10¹² cm⁻²).
  - y-axis = top-bottom layer-bias Δ_tb in meV (= 3 × paper V_z).

Single column (no PM_C3 / SVP_C3 split) since the bare bands are unique.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _common as qm  # noqa: E402


T = 0.3
NK = 96
KMAX = 0.30  # match the bare grid so DOS values are directly comparable

SPIN_VALLEY = 4
DELTA_TB_PER_D = 510.0  # meV per (V/nm)

# Cells in (n_e, Δ_tb) where n_e is per-flavor density (10¹² cm⁻²) and
# Δ_tb is the top-bottom layer-bias potential in meV.  Spans the same
# window as the FS-topology grid: n_e ∈ [0.25, 1.2], Δ_tb ∈ [90, 150].
CELLS = [
    (0.30, 100.0, "low n_e / low Δ_tb"),
    (0.70, 100.0, "mid n_e / low Δ_tb"),
    (1.15, 100.0, "high n_e / low Δ_tb"),
    (0.30, 123.0, "low n_e / paper V_z=41"),
    (0.70, 123.0, "mid n_e / paper V_z=41"),
    (1.15, 123.0, "high n_e / paper V_z=41"),
    (0.30, 145.0, "low n_e / high Δ_tb"),
    (0.70, 145.0, "mid n_e / high Δ_tb"),
    (1.15, 145.0, "high n_e / high Δ_tb"),
]


def k_y0_slice(eigs, ks):
    """Bands along the k_y=0 diagonal of the rhombic patch (i = j)."""
    nk = eigs.shape[0]
    diag_idx = np.arange(nk)
    bands = eigs[diag_idx, diag_idx, :]
    kxs = ks[diag_idx, diag_idx, 0]
    return kxs, bands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=None,
        help="Path to bare DOS grid npz for the phase-portrait background. "
             "Defaults to <noninteracting>/outputs/dos_grid.npz.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_npz = args.grid if args.grid is not None else (
        out_dir / "dos_grid.npz"
    )
    if not grid_npz.exists():
        raise FileNotFoundError(
            f"{grid_npz} not found — generate it first with "
            f"dos_grid.py"
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from contimod.plotting.meshgrid import plot_fermisurface

    setup_cache: dict[float, qm.QMSetup] = {}
    cell_data: list[dict] = []

    print("Computing bare bands at representative cells...", flush=True)
    for n_e, Delta_tb, label in CELLS:
        t0 = time.perf_counter()
        D_Vnm = float(Delta_tb) / DELTA_TB_PER_D
        if Delta_tb not in setup_cache:
            # Use the single-flavor (8-orbital) model so the FS insets show
            # one (spin, valley) flavor's iso-contour cleanly, not 4 overlaid.
            setup_cache[Delta_tb] = qm.build_setup(
                D_Vnm=D_Vnm, nk=NK, kmax=KMAX, T=T, bz_kind="rhombic",
                small_orbital=True, hartree_kind="dualgate_eps2",
            )
        setup = setup_cache[Delta_tb]
        ks = np.asarray(setup.h_template.kmesh.ks)
        bounds = np.asarray(setup.h_template.kmesh.bounds)

        n_cm12_total = float(n_e) * SPIN_VALLEY
        _, h_run = qm.n_electrons_for_density(setup, n_cm12_total, T)
        h_arr = np.asarray(h_run.hs)
        eigs = np.linalg.eigvalsh(
            h_arr.reshape(-1, h_arr.shape[-2], h_arr.shape[-1])
        ).reshape(*h_arr.shape[:2], -1)
        E_F = float(h_run.fermi.mu)
        kxs, bands = k_y0_slice(eigs, ks)

        cell_data.append(dict(
            n_e=n_e, Delta_tb=Delta_tb, D_Vnm=D_Vnm, label=label,
            ks=ks, bounds=bounds, eigs=eigs, E_F=E_F,
            kxs=kxs, bands=bands,
        ))
        print(f"  (n_e={n_e}, Δ_tb={Delta_tb} meV) — {label}: "
              f"E_F={E_F:.2f} meV  t={time.perf_counter()-t0:.1f}s",
              flush=True)

    # ---- Plot 1: Fermi surfaces ----
    n_rows = len(CELLS)
    fig_fs, axes_fs = plt.subplots(
        n_rows, 1, figsize=(5.5, 4.5 * n_rows), constrained_layout=True,
    )
    for row, d in enumerate(cell_data):
        ax = axes_fs[row]
        plot_fermisurface(
            d["ks"][..., 0], d["ks"][..., 1],
            d["eigs"], energies=[d["E_F"]],
            ax=ax, bounds=d["bounds"], linewidths=1.4,
        )
        ax.set_title(
            f"(n_e={d['n_e']}, Δ_tb={d['Delta_tb']:.0f} meV) — {d['label']}\n"
            f"E_F={d['E_F']:.2f} meV",
            fontsize=9,
        )
    fig_fs.suptitle(
        f"Bare Fermi surfaces (nk={NK}, kmax={KMAX}, T={T} meV)", fontsize=11,
    )
    out_fs = out_dir / "fermi_surfaces.png"
    fig_fs.savefig(out_fs, dpi=140)
    plt.close(fig_fs)
    print(f"\nSaved {out_fs}")

    # ---- Plot 2: Band structures along k_y = 0 (centered on E_F) ----
    fig_bs, axes_bs = plt.subplots(
        n_rows, 1, figsize=(5.5, 3.0 * n_rows), constrained_layout=True,
    )
    yzoom = 50.0  # meV around E_F
    for row, d in enumerate(cell_data):
        ax = axes_bs[row]
        kxs = d["kxs"]
        de = d["bands"] - d["E_F"]
        in_window = np.abs(de) <= yzoom * 1.5
        for b in range(d["bands"].shape[1]):
            mask = in_window[:, b]
            ax.scatter(kxs[mask], de[mask, b],
                        s=6, color="k", alpha=0.5, edgecolors="none")
        ax.axhline(0.0, color="tab:red", lw=0.8, linestyle="dashed")
        ax.set_ylim(-yzoom, yzoom)
        ax.set_xlim(kxs.min(), kxs.max())
        ax.set_xlabel(r"$k_x$ (1/lat)")
        ax.set_ylabel(r"$E - E_F$ (meV)")
        ax.set_title(
            f"(n_e={d['n_e']}, Δ_tb={d['Delta_tb']:.0f} meV) — {d['label']}",
            fontsize=9,
        )
        ax.grid(True, alpha=0.3)
    fig_bs.suptitle(
        f"Bare band structure scatter at $k_y=0$, centered on $E_F$ "
        f"(nk={NK}, ±{yzoom:.0f} meV window)", fontsize=11,
    )
    out_bs = out_dir / "band_structures.png"
    fig_bs.savefig(out_bs, dpi=140)
    plt.close(fig_bs)
    print(f"Saved {out_bs}")

    # ---- Plot 3: Phase portrait — linear DOS map with FS insets ----
    grid = np.load(grid_npz, allow_pickle=True)
    n_grid = grid["n_e_grid"]
    Dtb_grid = grid["Delta_tb_grid"]
    extent = (n_grid[0], n_grid[-1], Dtb_grid[0], Dtb_grid[-1])
    dos_arr = np.where(grid["dos_at_EF"] > 0, grid["dos_at_EF"], np.nan)
    p99 = float(np.nanpercentile(dos_arr, 99))

    fig_pp, ax_pp = plt.subplots(figsize=(13, 11), constrained_layout=False)
    im = ax_pp.imshow(dos_arr, origin="lower", aspect="auto",
                       extent=extent, cmap="viridis", vmin=0, vmax=p99)
    fig_pp.colorbar(
        im, ax=ax_pp,
        label=r"Bare $\rho(E_F)$ (linear, clipped at p99)",
        shrink=0.5, location="left",
    )
    ax_pp.set_xlabel(r"$n_e$ per flavor (× $10^{12}$ cm$^{-2}$)")
    ax_pp.set_ylabel(r"$\Delta_{tb}$ (meV)")
    ax_pp.set_title("Bare (non-interacting) phase diagram with Fermi-surface insets")

    n_span = float(n_grid[-1] - n_grid[0])
    Dtb_span = float(Dtb_grid[-1] - Dtb_grid[0])
    inset_size = 0.09
    pad = 0.03
    # Place insets on a fixed 3-row grid: bottom row of cells -> insets above
    # the dot; middle and top rows -> insets below the dot.
    Dtb_levels = sorted({d["Delta_tb"] for d in cell_data})
    n_levels = sorted({d["n_e"] for d in cell_data})
    for d in cell_data:
        n_e = d["n_e"]
        Delta_tb = d["Delta_tb"]
        ax_pp.plot(n_e, Delta_tb, "o", color="white",
                    markersize=8, markeredgecolor="black", zorder=5)
        nx = (n_e - n_grid[0]) / n_span
        ny = (Delta_tb - Dtb_grid[0]) / Dtb_span
        # Insets above the dot for the bottom Δ_tb row, below otherwise.
        dy = pad if Delta_tb == Dtb_levels[0] else -(pad + inset_size)
        # Horizontal: left column -> right of dot; middle column -> centred
        # on dot; right column -> left of dot.
        col_idx = n_levels.index(n_e)
        if col_idx == 0:
            dx = pad
        elif col_idx == len(n_levels) - 1:
            dx = -(pad + inset_size)
        else:
            dx = -inset_size / 2.0
        ax_in = ax_pp.inset_axes(
            [nx + dx, ny + dy, inset_size, inset_size],
            transform=ax_pp.transAxes,
        )
        plot_fermisurface(
            d["ks"][..., 0], d["ks"][..., 1],
            d["eigs"], energies=[d["E_F"]],
            ax=ax_in, bounds=d["bounds"], linewidths=0.8,
        )
        ax_in.set_xlabel("")
        ax_in.set_ylabel("")
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        ax_in.set_title("", fontsize=0)
        nx_in = nx + dx
        ny_in = ny + dy
        ax_pp.annotate(
            "", xy=(n_e, Delta_tb), xycoords="data",
            xytext=(nx_in + inset_size / 2, ny_in + inset_size / 2),
            textcoords=ax_pp.transAxes,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )

    out_pp = out_dir / "phase_portrait.png"
    fig_pp.savefig(out_pp, dpi=140, bbox_inches="tight")
    plt.close(fig_pp)
    print(f"Saved {out_pp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
