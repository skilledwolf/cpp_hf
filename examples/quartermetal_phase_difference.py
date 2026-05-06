"""Plot E(SVP_C3) − E(PM_C3) across the (n, D) grid.

Pure post-processing: loads both NPZ outputs and shows where SVP_C3 is
the ground state (negative ΔE → SVP_C3 wins).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "examples" / "outputs"

PM_PATH = OUT / "quartermetal_grid_pm_c3.npz"
SVP_PATH = OUT / "quartermetal_grid_svp_c3.npz"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = np.load(PM_PATH, allow_pickle=True)
    svp = np.load(SVP_PATH, allow_pickle=True)

    n_grid = pm["n_grid"]
    D_grid = pm["D_grid"]
    if not (np.allclose(n_grid, svp["n_grid"])
            and np.allclose(D_grid, svp["D_grid"])):
        raise ValueError("PM_C3 and SVP_C3 grids do not match")

    extent = (n_grid[0], n_grid[-1], D_grid[0], D_grid[-1])

    E_pm = pm["energy"]
    E_svp = svp["energy"]
    conv_pm = pm["converged"]
    conv_svp = svp["converged"]

    dE = E_svp - E_pm  # positive = SVP_C3 above PM_C3, negative = SVP_C3 wins

    # Mask cells where either phase didn't converge
    both_conv = conv_pm & conv_svp
    dE_masked = np.where(both_conv, dE, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    # Panel 1: ΔE = E_svp - E_pm
    vmax = float(np.nanmax(np.abs(dE_masked)))
    im0 = axes[0].imshow(dE_masked, origin="lower", aspect="auto",
                          extent=extent, cmap="RdBu_r",
                          vmin=-vmax, vmax=+vmax)
    fig.colorbar(im0, ax=axes[0],
                  label=r"$E_{\rm SVP\_C3} - E_{\rm PM\_C3}$ (meV)")
    axes[0].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[0].set_ylabel(r"$D$ (V/nm)")
    axes[0].set_title(r"Phase split: $\Delta E$"
                       "\n(blue = SVP_C3 lower)")
    not_both = ~both_conv
    axes[0].contour(np.linspace(extent[0], extent[1], both_conv.shape[1]),
                     np.linspace(extent[2], extent[3], both_conv.shape[0]),
                     not_both.astype(float), levels=[0.5],
                     colors="black", linewidths=0.5, linestyles="dashed")

    # Panel 2: which is ground state — sign of ΔE
    ground = np.where(both_conv,
                       np.where(dE < 0, -1.0, 1.0),  # -1 = SVP_C3, +1 = PM_C3
                       np.nan)
    im1 = axes[1].imshow(ground, origin="lower", aspect="auto",
                          extent=extent, cmap="RdBu",
                          vmin=-1.5, vmax=+1.5)
    cb1 = fig.colorbar(im1, ax=axes[1], ticks=[-1, 1])
    cb1.set_ticklabels(["SVP_C3 wins", "PM_C3 wins"])
    axes[1].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[1].set_ylabel(r"$D$ (V/nm)")
    axes[1].set_title("Ground-state phase (PM_C3 vs SVP_C3)")

    # Panel 3: SVP order parameter (re-plot for context)
    svp_op = svp["svp_op"]
    svp_op_masked = np.where(both_conv, svp_op, np.nan)
    im2 = axes[2].imshow(svp_op_masked, origin="lower", aspect="auto",
                          extent=extent, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im2, ax=axes[2], label="SVP order parameter")
    axes[2].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
    axes[2].set_ylabel(r"$D$ (V/nm)")
    axes[2].set_title("SVP order parameter\n(0 = uniform, 1 = polarized)")

    fig.suptitle("PM_C3 vs SVP_C3 phase competition (ε=2 dualgate)")
    out = OUT / "quartermetal_phase_difference.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")

    # Summary statistics
    valid = dE_masked[both_conv]
    print(f"\nΔE = E_SVP - E_PM stats (over {valid.size} cells where both converged):")
    print(f"  min:    {valid.min():+.3f} meV  (SVP_C3 most favored)")
    print(f"  median: {np.median(valid):+.3f} meV")
    print(f"  max:    {valid.max():+.3f} meV  (PM_C3 most favored)")
    print(f"  fraction with SVP_C3 ground state: "
          f"{(valid < 0).sum() / valid.size * 100:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
