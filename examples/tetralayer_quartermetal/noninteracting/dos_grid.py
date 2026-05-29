#!/usr/bin/env python
"""(n_e, Δ_tb) DOS-at-E_F map for the BARE non-interacting band structure.

No Hartree, no Fock, no SCF — just diagonalize the bare h(k) of
``contimod.graphene.NlayerABC`` at each layer-bias Δ_tb, find μ for the
requested per-flavor density n_e, and compute the DOS at μ via
``bztetra.twod``.

Conventions:

  - y-axis is Δ_tb in meV (top minus bottom layer-mean of diag h on the
    rhombic patch).  Our Δ_tb is the *full* top-bottom potential range,
    which is 3× the V_z that Huang & Wolf 2026 use in eq. 52 (their V_z
    sets the per-step layer increment).  The defaults span Δ_tb ∈
    [90, 150] meV ≡ V_z ∈ [30, 50] meV.

  - x-axis is the per-flavor electron density n_e in 10¹² cm⁻².  The
    paper projects onto a single conduction band, so its n_e is per
    spin-valley flavor.  Internally we still pass total density (= 4·n_e)
    to ``compute_chemicalpotential`` since the contimod Hamiltonian is
    spinful and valleyful at the SCF level.  Defaults span n_e ∈
    [0.25, 1.2].

Defaults: 30×30 cells at nk=128, kmax=0.30.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "outputs"
for p in (str(SRC_DIR), str(EXAMPLES_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


# Per-flavor → total density factor.  Spin × valley = 4.
SPIN_VALLEY = 4
# Δ_tb (meV) ↔ D (V/nm) for tetralayer at our default ε_zz = 2.0:
# Δ_tb = D × (N-1) × layer_spacing / ε_zz × 1000 = D × 510 meV/(V/nm).
DELTA_TB_PER_D = 510.0


def _run_dtb_row(Delta_tb_meV, n_e_grid_per_flavor, nk, kmax, T):
    """Bare DOS at E_F across the per-flavor density grid for one Δ_tb.

    The bare Hamiltonian only depends on Δ_tb (which sets the layer
    bias), so we diagonalise once per row and reuse the eigenvalues for
    every n_e.
    """
    import _common as qm

    D_Vnm = float(Delta_tb_meV) / DELTA_TB_PER_D
    setup = qm.build_setup(
        D_Vnm=D_Vnm, nk=int(nk), kmax=float(kmax), T=float(T),
        bz_kind="rhombic", hartree_kind="dualgate_eps2",  # HH unused (no SCF)
    )
    h_arr = np.asarray(setup.h_template.hs)
    weights = np.asarray(setup.weights)

    eigs = np.linalg.eigvalsh(
        h_arr.reshape(-1, h_arr.shape[-2], h_arr.shape[-1])
    ).reshape(*h_arr.shape[:2], -1)

    s = 2.0 * float(kmax)
    h_tri = s * np.sqrt(3.0) / 2.0
    Bmat = np.array([[s / 2.0, s / 2.0], [-h_tri, h_tri]])

    import bztetra.twod as twod

    rows = []
    for n_e in n_e_grid_per_flavor:
        n_cm12_total = float(n_e) * SPIN_VALLEY
        _, h_run = qm.n_electrons_for_density(setup, n_cm12_total, T)
        mu = float(h_run.fermi.mu)
        W_dos = twod.density_of_states_weights(Bmat, eigs, np.array([mu]))
        area_prefactor = float(weights.sum())
        dos_at_EF = float(W_dos.sum()) * area_prefactor
        rows.append({
            "n_e": float(n_e), "Delta_tb": float(Delta_tb_meV),
            "D_Vnm": D_Vnm, "n_cm12_total": n_cm12_total,
            "mu": mu, "dos_at_EF": dos_at_EF,
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-grid", type=int, default=30,
                    help="# samples along n_e (per-flavor) axis.")
    p.add_argument("--Delta-grid", type=int, default=30,
                    help="# samples along Δ_tb axis.")
    p.add_argument("--n-min", type=float, default=0.25,
                    help="Min n_e per flavor (10¹² cm⁻²).")
    p.add_argument("--n-max", type=float, default=1.2,
                    help="Max n_e per flavor (10¹² cm⁻²).")
    p.add_argument("--Delta-min", type=float, default=90.0,
                    help="Min Δ_tb in meV.")
    p.add_argument("--Delta-max", type=float, default=150.0,
                    help="Max Δ_tb in meV.")
    p.add_argument("--nk", type=int, default=128)
    p.add_argument("--kmax", type=float, default=0.30)
    p.add_argument("--T", type=float, default=0.3)
    p.add_argument("--output", type=Path,
                    default=OUT_DIR / "dos_grid.npz")
    args = p.parse_args()

    n_e_grid = np.linspace(args.n_min, args.n_max, args.n_grid)
    Delta_grid = np.linspace(args.Delta_min, args.Delta_max, args.Delta_grid)

    print(f"Running {args.Delta_grid}×{args.n_grid} BARE grid:", flush=True)
    print(f"  n_e (per flavor) ∈ [{args.n_min:.3f}, {args.n_max:.3f}]")
    print(f"  Δ_tb (meV)        ∈ [{args.Delta_min:.1f}, {args.Delta_max:.1f}]")
    print(f"  nk={args.nk}, kmax={args.kmax}, T={args.T}")

    t_start = time.perf_counter()
    results = {}
    n_total = len(Delta_grid) * len(n_e_grid)
    for i, Dtb in enumerate(Delta_grid):
        rows = _run_dtb_row(float(Dtb), n_e_grid, int(args.nk),
                             float(args.kmax), float(args.T))
        results[float(Dtb)] = rows
        n_done = sum(len(rs) for rs in results.values())
        t_elapsed = time.perf_counter() - t_start
        t_eta = t_elapsed / max(n_done, 1) * (n_total - n_done)
        print(f"  done Δ_tb={Dtb:6.2f} meV: {n_done}/{n_total} "
              f"t={t_elapsed:.1f}s ETA={t_eta:.1f}s", flush=True)

    Delta_grid_arr = np.asarray(sorted(results.keys()))
    keys = ("mu", "dos_at_EF", "n_cm12_total", "D_Vnm")
    arrays = {}
    for k in keys:
        a = np.full((len(Delta_grid_arr), len(n_e_grid)), np.nan)
        for i, Dtb in enumerate(Delta_grid_arr):
            for j, r in enumerate(results[Dtb]):
                a[i, j] = r[k]
        arrays[k] = a

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             n_e_grid=n_e_grid, Delta_tb_grid=Delta_grid_arr,
             nk=args.nk, kmax=args.kmax, T=args.T, **arrays)
    print(f"Saved {out_path}")

    _plot(out_path)
    return 0


def _plot(npz_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    n_e_grid = data["n_e_grid"]
    Delta_grid = data["Delta_tb_grid"]
    dos = data["dos_at_EF"]
    mu = data["mu"]

    extent = (n_e_grid[0], n_e_grid[-1], Delta_grid[0], Delta_grid[-1])
    dos_lin = np.where(dos > 0, dos, np.nan)
    p99 = float(np.nanpercentile(dos_lin, 99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)

    im0 = axes[0].imshow(dos_lin, origin="lower", aspect="auto",
                          extent=extent, cmap="viridis", vmin=0, vmax=p99)
    fig.colorbar(im0, ax=axes[0],
                  label=r"$\rho(E_F)$ (linear, clipped at p99)")
    axes[0].set_xlabel(r"$n_e$ per flavor (× $10^{12}$ cm$^{-2}$)")
    axes[0].set_ylabel(r"$\Delta_{tb}$ (meV)")
    axes[0].set_title("Bare: DOS at $E_F$ (linear)")

    im2 = axes[1].imshow(mu, origin="lower", aspect="auto",
                          extent=extent, cmap="cividis")
    fig.colorbar(im2, ax=axes[1], label=r"$\mu$ (meV)")
    axes[1].set_xlabel(r"$n_e$ per flavor (× $10^{12}$ cm$^{-2}$)")
    axes[1].set_ylabel(r"$\Delta_{tb}$ (meV)")
    axes[1].set_title("Bare: chemical potential")

    out = npz_path.with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    raise SystemExit(main())
