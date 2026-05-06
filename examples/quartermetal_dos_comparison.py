"""Compare DOS-at-E_F renderings: log10 vs linear-with-percentile-cutoff,
for both PM_C3 and SVP_C3 grids.  Pure post-processing of saved NPZs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "examples" / "outputs"

PM_PATH = OUT / "quartermetal_grid_pm_c3.npz"
SVP_PATH = OUT / "quartermetal_grid_svp_c3.npz"

PERCENTILE_CAPS = (95.0, 99.0)  # plot at these percentile clips


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pm = np.load(PM_PATH, allow_pickle=True)
    svp = np.load(SVP_PATH, allow_pickle=True)
    n_grid = pm["n_grid"]
    D_grid = pm["D_grid"]
    extent = (n_grid[0], n_grid[-1], D_grid[0], D_grid[-1])

    pm_dos = np.where(pm["dos_at_EF"] > 0, pm["dos_at_EF"], np.nan)
    svp_dos = np.where(svp["dos_at_EF"] > 0, svp["dos_at_EF"], np.nan)

    n_rows = 2
    n_cols = 1 + len(PERCENTILE_CAPS)  # log10 + each percentile linear plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                              constrained_layout=True)

    for row, (label, dos_arr) in enumerate([("PM_C3", pm_dos),
                                              ("SVP_C3", svp_dos)]):
        # log10
        log_arr = np.log10(dos_arr)
        im = axes[row, 0].imshow(log_arr, origin="lower", aspect="auto",
                                  extent=extent, cmap="viridis")
        fig.colorbar(im, ax=axes[row, 0],
                     label=r"$\log_{10}(\rho(E_F))$")
        axes[row, 0].set_title(f"{label}: log$_{{10}}$ DOS")
        axes[row, 0].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
        axes[row, 0].set_ylabel(r"$D$ (V/nm)")

        # linear with percentile clip
        for col, q in enumerate(PERCENTILE_CAPS, start=1):
            vmax = float(np.nanpercentile(dos_arr, q))
            im = axes[row, col].imshow(dos_arr, origin="lower", aspect="auto",
                                        extent=extent, cmap="viridis",
                                        vmin=0, vmax=vmax)
            fig.colorbar(im, ax=axes[row, col],
                         label=r"$\rho(E_F)$ (linear)")
            axes[row, col].set_title(
                f"{label}: linear DOS, ≤{q:.0f}-th percentile = {vmax:.2e}"
            )
            axes[row, col].set_xlabel(r"$n$ (× $10^{12}$ cm$^{-2}$)")
            axes[row, col].set_ylabel(r"$D$ (V/nm)")

    fig.suptitle("DOS-at-E_F: log10 vs linear (clipped at percentile)")
    out = OUT / "quartermetal_dos_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")

    # Also print summary stats
    print("\nDOS summary (per cell, ε=2 grid):")
    for label, dos_arr in [("PM_C3 ", pm_dos), ("SVP_C3", svp_dos)]:
        v = dos_arr[~np.isnan(dos_arr)]
        print(f"  {label}: min={v.min():.3e}  median={np.median(v):.3e}  "
              f"p95={np.percentile(v, 95):.3e}  p99={np.percentile(v, 99):.3e}  "
              f"max={v.max():.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
